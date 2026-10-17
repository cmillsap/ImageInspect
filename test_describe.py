"""
Tests for choosing hardware, downloading models and describing photos,
with onnxruntime, the network and the language model stood in for.
"""

import hashlib
import io
import os
import sys
import types
import warnings

import pytest
from PIL import Image

import describe
import models


# ============================================================================
# Hardware
# ============================================================================

class FakeOnnxruntime(types.ModuleType):
    """onnxruntime whose CUDA provider works, quietly falls back, or fails."""

    def __init__(self, cuda_build=True, cuda="works"):
        super().__init__("onnxruntime")
        self.cuda_build, self.cuda = cuda_build, cuda
        self.preloaded = False
        self.sessions = []
        fake = self

        class InferenceSession:
            def __init__(self, path, options, providers):
                fake.sessions.append(providers)
                if models.CUDA in providers and fake.cuda == "raises":
                    raise RuntimeError("CUDA failure")
                self.providers = (providers if fake.cuda == "works"
                                  else [models.CPU])

            def get_providers(self):
                return self.providers

        self.InferenceSession = InferenceSession
        self.SessionOptions = types.SimpleNamespace

    def get_available_providers(self):
        return ([models.CUDA, models.CPU] if self.cuda_build else [models.CPU])

    def preload_dlls(self):
        self.preloaded = True


@pytest.fixture
def hardware(monkeypatch):
    """Install a fake onnxruntime and GPU; returns a function that sets them."""
    models._reset_for_tests()

    def setup(gpu="Test GPU", **ort):
        fake = FakeOnnxruntime(**ort)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake)
        monkeypatch.setattr(models, "nvidia_gpu", lambda: gpu)
        return fake
    yield setup
    models._reset_for_tests()


SPEC = models.ModelFile("m.onnx", "https://example.com/m.onnx", None, 1)


class TestHardware:

    def test_cuda_build_with_a_gpu_uses_it(self, hardware):
        ort = hardware()
        session = models.create_session(SPEC)
        assert session.get_providers()[0] == models.CUDA
        assert ort.preloaded
        assert models.device() == "GPU (Test GPU)"
        assert describe.preferred_variant() == "cuda"

    def test_cpu_build_never_tries_cuda(self, hardware):
        ort = hardware(cuda_build=False)
        models.create_session(SPEC)
        assert ort.sessions == [[models.CPU]]
        assert not ort.preloaded
        assert models.device() == "CPU"
        assert describe.preferred_variant() == "cpu"

    def test_no_gpu_runs_on_the_cpu(self, hardware):
        ort = hardware(gpu=None)
        models.create_session(SPEC)
        assert ort.sessions == [[models.CPU]]
        assert models.device() == "CPU (no NVIDIA GPU found)"
        assert describe.preferred_variant() == "cpu"

    def test_quiet_fallback_is_noticed_and_cuda_not_tried_again(self, hardware):
        ort = hardware(cuda="falls back")
        with pytest.warns(RuntimeWarning, match="CUDA could not start"):
            models.create_session(SPEC)
        models.create_session(SPEC)
        assert ort.sessions == [[models.CUDA, models.CPU], [models.CPU]]
        assert models.device().startswith("CPU (CUDA could not start")
        assert describe.preferred_variant() == "cpu"

    def test_cuda_error_retries_on_the_cpu(self, hardware):
        ort = hardware(cuda="raises")
        with pytest.warns(RuntimeWarning):
            session = models.create_session(SPEC)
        assert session.get_providers() == [models.CPU]
        assert ort.sessions[-1] == [models.CPU]

    def test_failure_is_warned_about_once(self, hardware):
        hardware()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models.cuda_failed("first")
            models.cuda_failed("second")
        assert len(caught) == 1
        assert models.device() == "CPU (first)"


# ============================================================================
# Downloads
# ============================================================================

class FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A fake web server holding files, which may honour Range requests."""
    monkeypatch.setattr(models, "MODEL_DIR", str(tmp_path / "models"))
    files, requests = {}, []
    server = types.SimpleNamespace(files=files, requests=requests,
                                   ranges=True)

    def urlopen(request, timeout):
        data = files[request.full_url]
        requests.append(request)
        header = request.get_header("Range")
        if header and server.ranges:
            start = int(header.split("=")[1].rstrip("-"))
            return FakeResponse(data[start:], 206)
        return FakeResponse(data, 200)
    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    return server


def host(server, name, data, sha=True):
    url = f"https://example.com/{name}"
    server.files[url] = data
    digest = hashlib.sha256(data).hexdigest() if sha else None
    return models.ModelFile(name, url, digest, len(data))


DATA = os.urandom(2_500_000)            # Arrives in three chunks.


class TestDownloads:

    def test_downloads_into_subfolders(self, server):
        spec = host(server, "phi/cpu/a.onnx", DATA)
        models.ensure_models([spec])
        with open(models.model_path(spec), "rb") as f:
            assert f.read() == DATA
        assert models.is_present(spec)

    def test_progress_counts_across_all_the_files(self, server):
        specs = [host(server, "a", DATA), host(server, "b", DATA[:1000])]
        seen = []
        models.ensure_models(specs, lambda m, d, t: seen.append((m, d, t)),
                             label="Model")
        assert seen[-1] == ("Downloading Model", len(DATA) + 1000, len(DATA) + 1000)
        assert all(t == len(DATA) + 1000 for _, _, t in seen)

    def test_cancelling_keeps_what_arrived(self, server):
        spec = host(server, "a", DATA)
        with pytest.raises(models.Cancelled):
            models.ensure_models([spec], should_cancel=lambda: True)
        assert not models.is_present(spec)
        assert models.bytes_to_download([spec]) == len(DATA) - (1 << 20)

    def test_resumes_where_it_stopped(self, server):
        spec = host(server, "a", DATA)
        with pytest.raises(models.Cancelled):
            models.ensure_models([spec], should_cancel=lambda: True)
        models.ensure_models([spec])
        assert server.requests[-1].get_header("Range") == f"bytes={1 << 20}-"
        with open(models.model_path(spec), "rb") as f:
            assert f.read() == DATA

    def test_starts_again_if_the_server_will_not_resume(self, server):
        spec = host(server, "a", DATA)
        with pytest.raises(models.Cancelled):
            models.ensure_models([spec], should_cancel=lambda: True)
        server.ranges = False
        models.ensure_models([spec])
        with open(models.model_path(spec), "rb") as f:
            assert f.read() == DATA

    def test_corrupt_download_is_thrown_away(self, server):
        spec = host(server, "a", DATA)._replace(sha256="0" * 64)
        with pytest.raises(ValueError, match="corrupt"):
            models.ensure_models([spec])
        assert models.bytes_to_download([spec]) == len(DATA)
        assert os.listdir(models.MODEL_DIR) == []

    def test_present_files_are_not_fetched(self, server):
        spec = host(server, "a", DATA)
        models.ensure_models([spec])
        models.ensure_models([spec])
        assert len(server.requests) == 1


# ============================================================================
# The description model
# ============================================================================

class TestModelFiles:

    @pytest.mark.parametrize("variant, size", [("cuda", "2.6 GB"), ("cpu", "3.2 GB")])
    def test_each_build_is_complete_and_pinned(self, variant, size):
        files = describe.model_files(variant)
        assert len(files) == 11
        assert all("/672d73375fa86f3d7787e40ac593e33a4f04a055/" in f.url
                   and f.url.endswith(os.path.basename(f.name))
                   and len(f.sha256) == 64 for f in files)
        assert describe.format_size(sum(f.size for f in files)) == size

    def test_builds_download_to_their_own_folders(self):
        names = {os.path.dirname(f.name) for v in describe.VARIANTS
                 for f in describe.model_files(v)}
        assert names == {"phi-3.5-vision/cuda", "phi-3.5-vision/cpu"}

    def test_image_is_upright_and_small_enough(self, tmp_path):
        path = tmp_path / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6                    # Rotated 90 degrees
        Image.new("RGB", (3000, 1000)).save(path, exif=exif)
        image = Image.open(io.BytesIO(describe.image_bytes(str(path))))
        assert image.size == (448, 1344)


class TestCache:

    def test_remembers_until_the_file_changes(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(b"x")
        cache = describe.DescriptionCache(str(tmp_path / "index.sqlite"))
        assert cache.get(str(photo)) is None
        cache.put(str(photo), "A cat.")
        assert cache.get(str(photo)) == "A cat."
        os.utime(photo, (1, 1))
        assert cache.get(str(photo)) is None

    def test_missing_file(self, tmp_path):
        cache = describe.DescriptionCache(str(tmp_path / "index.sqlite"))
        assert cache.get(str(tmp_path / "gone.jpg")) is None


class FakeDescriber:
    """Stands in for the model; the GPU build fails when gpu_works is off."""

    gpu_works = True
    loaded = []

    def __init__(self, variant):
        FakeDescriber.loaded.append(variant)
        if variant == "cuda" and not FakeDescriber.gpu_works:
            raise describe.GpuUnavailable("no")
        self.variant = variant

    def describe(self, path, on_text, should_cancel):
        if should_cancel():
            return None
        on_text("A")
        return "A photo."


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    FakeDescriber.gpu_works, FakeDescriber.loaded = True, []
    monkeypatch.setattr(describe, "Describer", FakeDescriber)
    downloads = []
    monkeypatch.setattr(describe, "download",
                        lambda variant, *a: downloads.append(variant))
    available = {"cuda": True, "cpu": True}
    monkeypatch.setattr(describe, "downloaded", lambda v: available[v])
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    state = {"descriptions": describe.DescriptionCache(str(tmp_path / "db"))}
    return types.SimpleNamespace(state=state, photo=str(photo),
                                 downloads=downloads, available=available)


class TestDescribeFile:

    def test_describes_and_remembers(self, fake_model):
        texts = []
        assert describe.describe_file(fake_model.state, fake_model.photo, "cuda",
                                      on_text=texts.append) == "A photo."
        assert texts == ["A"]
        assert fake_model.downloads == ["cuda"]
        assert describe.describe_file(fake_model.state, fake_model.photo,
                                      "cuda") == "A photo."
        assert FakeDescriber.loaded == ["cuda"]

    def test_model_is_loaded_once(self, fake_model, tmp_path):
        other = tmp_path / "b.jpg"
        other.write_bytes(b"y")
        describe.describe_file(fake_model.state, fake_model.photo, "cuda")
        describe.describe_file(fake_model.state, str(other), "cuda")
        assert FakeDescriber.loaded == ["cuda"]

    def test_cancelled_is_not_remembered(self, fake_model):
        assert describe.describe_file(fake_model.state, fake_model.photo, "cpu",
                                      should_cancel=lambda: True) is None
        assert fake_model.state["descriptions"].get(fake_model.photo) is None

    def test_gpu_failure_uses_a_downloaded_cpu_build(self, fake_model):
        FakeDescriber.gpu_works = False
        describe.describe_file(fake_model.state, fake_model.photo, "cuda")
        assert FakeDescriber.loaded == ["cuda", "cpu"]
        assert fake_model.state["describer"].variant == "cpu"

    def test_gpu_failure_without_a_cpu_build_says_so(self, fake_model):
        FakeDescriber.gpu_works = False
        fake_model.available["cpu"] = False
        with pytest.raises(describe.GpuUnavailable):
            describe.describe_file(fake_model.state, fake_model.photo, "cuda")
        assert "describer" not in fake_model.state

    def test_asking_for_the_cpu_build_replaces_the_gpu_one(self, fake_model, tmp_path):
        describe.describe_file(fake_model.state, fake_model.photo, "cuda")
        other = tmp_path / "b.jpg"
        other.write_bytes(b"y")
        describe.describe_file(fake_model.state, str(other), "cpu")
        assert FakeDescriber.loaded == ["cuda", "cpu"]
