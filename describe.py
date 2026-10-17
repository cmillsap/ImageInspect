"""
Describe

Writes a description of a photo with Microsoft's Phi-3.5-vision, a small
vision language model that runs on this computer through
onnxruntime-genai, so photos never leave it.

The model comes in two builds, int4-quantised for the GPU (CUDA) or for
the CPU, and only the one this computer can use is downloaded (about
2.6 GB or 3.2 GB) - after asking, since it is large. On an RTX-class GPU a
description takes a few seconds; on the CPU, about half a minute. If the
GPU build will not load after all, the CPU build is offered instead.

Descriptions are cached in the index database, keyed on path and
modification time, so asking again is instant.
"""

import io
import os
import shutil
import sqlite3
from typing import Callable, Dict, List, Optional

import models
from image_io import open_image
from models import ModelFile, Progress

MODEL_NAME = "phi-3.5-vision"
MODEL_LABEL = "Phi-3.5-vision"
_BASE_URL = ("https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx/"
             "resolve/672d73375fa86f3d7787e40ac593e33a4f04a055/")
_FOLDERS = {"cuda": "gpu/gpu-int4-rtn-block-32/",
            "cpu": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/"}

# name -> {variant: (size, sha256)}
_FILES = {
    "genai_config.json": {
        "cpu": (2121, "aab58b61ca85a8bf2d4781fe1b4299e3d0f20a4cd902bc5c02adfe361b62aad9"),
        "cuda": (2121, "aab58b61ca85a8bf2d4781fe1b4299e3d0f20a4cd902bc5c02adfe361b62aad9")},
    "processor_config.json": {
        "cpu": (752, "b6f5a3ceb16ad076b31430e358d68f0ebb53d9ddeb91270e9e5c4ff8fefed67f"),
        "cuda": (752, "b6f5a3ceb16ad076b31430e358d68f0ebb53d9ddeb91270e9e5c4ff8fefed67f")},
    "special_tokens_map.json": {
        "cpu": (670, "f9535fada19315e0d8eb89521cebe111c008fee98d16e7c6744acfb50505d8c3"),
        "cuda": (706, "8aa9f4bae46e5280787c8979252ffbf81f411a9aae79650f33a03d2a4b282514")},
    "tokenizer.json": {
        "cpu": (1851388, "478fb6fd0bf1403eb2ffda892e285f5ae968b9e0c06eb2815552ebedacb8ca93"),
        "cuda": (1851388, "478fb6fd0bf1403eb2ffda892e285f5ae968b9e0c06eb2815552ebedacb8ca93")},
    "tokenizer_config.json": {
        "cpu": (9547, "d637971d627e4f8ce0422401bd0a305d9f763657ad21e8205afe8d36b6672524"),
        "cuda": (9960, "7743c1615083d279af0170c5550dd7f40ea4de275fb3a99a5cb09bbe43e50662")},
    "phi-3.5-v-instruct-embedding.onnx": {
        "cpu": (6091, "83bb330b0a319812eccdf6b6bc5ce17eb6a120bcf76120f6af27c19c3d396ea1"),
        "cuda": (6091, "b707f7f659d8585a9253f3b95b1740522f3062a7eb4465f09d93c984743e399d")},
    "phi-3.5-v-instruct-embedding.onnx.data": {
        "cpu": (394002432, "2dcbf0fcb14433ac8fa82b35c39675c01fb552512b4c0e5d54e8f247046ef393"),
        "cuda": (197001216, "e0af51a7c122ed63878262191faa905c1e317c7153185a6f7275e61d22fda63a")},
    "phi-3.5-v-instruct-text.onnx": {
        "cpu": (52123846, "f7580bd589693e87e3825ffe8eda3f5fda4158d2d82d27803c263aa0e7454a81"),
        "cuda": (291883, "d598bf368ed0b84b9d9db91a6a4843a2c92d071bf51df6763daadb3213922204")},
    "phi-3.5-v-instruct-text.onnx.data": {
        "cpu": (2327285760, "214fd2e8a09b8c7b141f9095171aed6f994768a36803b7eeb9ea3be16da99dd6"),
        "cuda": (2120189952, "de9df6b7062a3d133af7674041e583534dff5a95758572c15468eb1bf602e576")},
    "phi-3.5-v-instruct-vision.onnx": {
        "cpu": (414021, "e06ccb2d2c35c2565f0abe507241f33cae9d7f63a1d79f6170b317d12f35c3bf"),
        "cuda": (422406, "30d5d0dc644db606e42cf0d759acb2a22aef4ef572c77689e910ad9fa3b5ad1e")},
    "phi-3.5-v-instruct-vision.onnx.data": {
        "cpu": (444916232, "45af0e626ba0eff702d9c1873c84f9e60f1f17bb2036bf1abc045968b5e645df"),
        "cuda": (282245640, "68aafcb815850a4982644fc81efec6800054865bb6dbc4ff41f275cc73533eac")},
}

VARIANTS = ("cuda", "cpu")


def model_files(variant: str) -> List[ModelFile]:
    files = []
    for name, builds in _FILES.items():
        size, sha256 = builds[variant]
        files.append(ModelFile(f"{MODEL_NAME}/{variant}/{name}",
                               _BASE_URL + _FOLDERS[variant] + name, sha256, size))
    return files


def model_dir(variant: str) -> str:
    return os.path.join(models.MODEL_DIR, MODEL_NAME, variant)


def preferred_variant() -> str:
    """The build this computer should run."""
    return "cuda" if models.use_cuda() else "cpu"


def downloaded(variant: str) -> bool:
    return all(models.is_present(f) for f in model_files(variant))


def download_size(variant: str) -> int:
    return models.bytes_to_download(model_files(variant))


def free_space() -> int:
    """Free bytes on the drive the models go to."""
    folder = models.MODEL_DIR
    while not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return shutil.disk_usage(folder).free


def download(variant: str, progress: Progress = None,
             should_cancel: Callable[[], bool] = lambda: False) -> None:
    models.ensure_models(model_files(variant), progress, should_cancel,
                         label=f"{MODEL_LABEL} ({format_size(download_size(variant))})")


def format_size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{max(n, 1e6) / 1e6:.0f} MB"


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------

PROMPT = ("<|user|>\n<|image_1|>\nDescribe this photo in detail: who or what "
          "is in it, the setting, what is happening, any visible text, and "
          "notable details.<|end|>\n<|assistant|>\n")
# Phi-3.5-vision sees a photo as at most four 336-pixel crops plus an
# overview, so a larger image only costs decoding time.
INPUT_SIZE = 1344
MAX_TOKENS = 600


class GpuUnavailable(Exception):
    """The GPU build of the model could not be loaded."""


class Describer:
    """One loaded build of the model. Loading takes a few seconds and a few
    GB of memory, so keep it for later descriptions."""

    def __init__(self, variant: str):
        self.variant = variant
        if variant == "cuda" and not models.use_cuda():
            raise GpuUnavailable("CUDA is not available")
        import onnxruntime_genai as og
        self._og = og
        try:
            config = og.Config(model_dir(variant))
            config.clear_providers()
            if variant == "cuda":
                config.append_provider("cuda")
            self._model = og.Model(config)
        except Exception as e:                              # noqa: BLE001
            if variant == "cuda":
                models.cuda_failed(f"the description model could not use "
                                   f"the GPU ({e})")
                raise GpuUnavailable(str(e)) from e
            raise
        if variant == "cuda":
            models.note_cuda_used()
        self._processor = self._model.create_multimodal_processor()

    def describe(self, path: str, on_text: Callable[[str], None] = lambda t: None,
                 should_cancel: Callable[[], bool] = lambda: False) -> Optional[str]:
        """A description of the photo at path, passing on_text the text so
        far as it is written. None if cancelled."""
        og = self._og
        images = og.Images.open_bytes(image_bytes(path))
        inputs = self._processor(PROMPT, images=images)
        params = og.GeneratorParams(self._model)
        params.set_search_options(max_length=7680)
        generator = og.Generator(self._model, params)
        generator.set_inputs(inputs)
        stream = self._processor.create_stream()
        text = ""
        for _ in range(MAX_TOKENS):
            if generator.is_done():
                break
            if should_cancel():
                return None
            generator.generate_next_token()
            text += stream.decode(generator.get_next_tokens()[0])
            on_text(text)
        return text.strip()


def image_bytes(path: str) -> bytes:
    """The photo as a PNG the model's processor can decode: upright, RGB,
    and no larger than it needs. Going through open_image also covers
    HEIC and DNG, which the processor cannot read itself."""
    with open_image(path, draft_size=(INPUT_SIZE, INPUT_SIZE)) as img:
        img = img.convert("RGB")
        img.thumbnail((INPUT_SIZE, INPUT_SIZE))
        out = io.BytesIO()
        img.save(out, "PNG")
    return out.getvalue()


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

class DescriptionCache:
    """Descriptions already written, in the index database."""

    def __init__(self, cache_path: str = os.path.join(models.APP_DIR, "index.sqlite")):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS descriptions (path TEXT NOT NULL, "
            "model TEXT NOT NULL, mtime REAL NOT NULL, text TEXT NOT NULL, "
            "PRIMARY KEY (path, model))")
        self._db.commit()

    def get(self, path: str) -> Optional[str]:
        path = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        row = self._db.execute(
            "SELECT mtime, text FROM descriptions WHERE path = ? AND model = ?",
            (path, MODEL_NAME)).fetchone()
        return row[1] if row and row[0] == mtime else None

    def put(self, path: str, text: str) -> None:
        path = os.path.abspath(path)
        self._db.execute(
            "INSERT OR REPLACE INTO descriptions VALUES (?, ?, ?, ?)",
            (path, MODEL_NAME, os.path.getmtime(path), text))
        self._db.commit()


def describe_file(state: Dict, path: str, variant: str,
                  progress: Progress = None,
                  on_text: Callable[[str], None] = lambda t: None,
                  should_cancel: Callable[[], bool] = lambda: False
                  ) -> Optional[str]:
    """Describe path with variant's build, downloading it if need be and
    keeping the loaded model and cache in state. None if cancelled.

    Raises GpuUnavailable if the GPU build will not load and the CPU build
    is not downloaded; with the CPU build at hand it is used instead.
    """
    cache = state.get("descriptions")
    if cache is None:
        cache = state["descriptions"] = DescriptionCache()
    cached = cache.get(path)
    if cached is not None:
        return cached
    describer = state.get("describer")
    if describer is None or (describer.variant != variant and variant == "cpu"):
        download(variant, progress, should_cancel)
        if progress:
            progress(f"Loading {MODEL_LABEL}", 0, 0)
        try:
            describer = Describer(variant)
        except GpuUnavailable:
            if not downloaded("cpu"):
                raise
            describer = Describer("cpu")
        state["describer"] = describer
    if progress:
        progress("Describing", 0, 0)
    text = describer.describe(path, on_text, should_cancel)
    if text is not None:
        cache.put(path, text)
    return text
