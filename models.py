"""
Models

Downloading model files and choosing the hardware they run on, shared by
the CLIP, face and description pipelines.

Every onnxruntime session is created by create_session(), so the choice of
hardware lives in one place. When the installed onnxruntime is the CUDA
build (onnxruntime-gpu, from requirements-cuda.txt) and the machine has an
NVIDIA GPU that CUDA 13 supports, sessions run on the GPU; otherwise on the
CPU. onnxruntime does not fail when CUDA cannot be used, it quietly runs on
the CPU instead, so each session is checked for the hardware it actually
got. The first failure turns CUDA off for the rest of the run, so it is
tried once, and device() says which is in use and why.

DirectML is not offered: onnxruntime-genai's DirectML build lags well
behind the CUDA and CPU ones and needs an older onnxruntime.

Downloads resume where they stopped, so a cancelled or interrupted
multi-gigabyte download is not started again from scratch.
"""

import ctypes
import hashlib
import os
import sys
import urllib.request
import warnings
from typing import Callable, Iterable, List, NamedTuple, Optional

APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                       "ImageInspect")
MODEL_DIR = os.path.join(APP_DIR, "models")

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
# The oldest GPUs CUDA 13 supports are Turing (compute capability 7.5).
MIN_COMPUTE_CAPABILITY = (7, 5)

Progress = Optional[Callable[[str, int, int], None]]


class Cancelled(Exception):
    """A download was cancelled; what has arrived so far is kept."""


class ModelFile(NamedTuple):
    name: str                          # path under MODEL_DIR
    url: str
    sha256: Optional[str]
    size: int
    shared_path: Optional[str] = None  # an existing copy to use instead


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------

def model_path(spec: ModelFile) -> str:
    """Where a model file is, or will be once downloaded."""
    if (spec.shared_path and os.path.isfile(spec.shared_path)
            and os.path.getsize(spec.shared_path) == spec.size):
        return spec.shared_path
    return os.path.join(MODEL_DIR, spec.name)


def is_present(spec: ModelFile) -> bool:
    path = model_path(spec)
    return os.path.isfile(path) and os.path.getsize(path) == spec.size


def bytes_to_download(specs: Iterable[ModelFile]) -> int:
    """How much of specs is still to download, allowing for partial files."""
    total = 0
    for spec in specs:
        if not is_present(spec):
            part = os.path.join(MODEL_DIR, spec.name) + ".part"
            done = os.path.getsize(part) if os.path.isfile(part) else 0
            total += max(spec.size - done, 0)
    return total


def _download(spec: ModelFile, progress: Callable[[int], None],
              should_cancel: Callable[[], bool]) -> None:
    """Fetch spec, carrying on from a .part file left by an earlier try.

    progress is told the bytes of spec there are so far.
    """
    final_path = os.path.join(MODEL_DIR, spec.name)
    part_path = final_path + ".part"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    digest = hashlib.sha256()
    done = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
    if done > spec.size:
        os.remove(part_path)
        done = 0
    request = urllib.request.Request(spec.url)
    if done:
        request.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if done and response.status == 206:
            with open(part_path, "rb") as f:
                while chunk := f.read(1 << 20):
                    digest.update(chunk)
            mode = "ab"
        else:
            done, mode = 0, "wb"       # The server sent it all again.
        with open(part_path, mode) as out:
            progress(done)
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done)
                if should_cancel():
                    raise Cancelled()
    if done != spec.size or (spec.sha256 and digest.hexdigest() != spec.sha256):
        os.remove(part_path)
        raise ValueError(f"Downloaded {spec.name} is corrupt; please retry.")
    os.replace(part_path, final_path)


def ensure_models(specs: Iterable[ModelFile], progress: Progress = None,
                  should_cancel: Callable[[], bool] = lambda: False,
                  label: Optional[str] = None) -> None:
    """Download whichever of specs are missing.

    progress is told the bytes done and in all, as one total across the
    files; label names the download, and is otherwise the file's name.
    Raises Cancelled if should_cancel() turns true.
    """
    missing = [s for s in specs if not is_present(s)]
    total = sum(s.size for s in missing)
    finished = 0
    for spec in missing:
        message = f"Downloading {label or spec.name}"
        report = ((lambda done, m=message, f=finished:
                   progress(m, f + done, total)) if progress
                  else (lambda done: None))
        _download(spec, report, should_cancel)
        finished += spec.size


# ----------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------

_gpu: Optional[str] = None      # The GPU's name, once looked for
_gpu_checked = False
_cuda_problem: Optional[str] = None   # Why CUDA is not used, if it is not
_cuda_used = False              # A session has run on the GPU


def nvidia_gpu() -> Optional[str]:
    """The name of the GPU CUDA would use (device 0), or None if there is
    no NVIDIA driver, no GPU it can see, or only ones too old for CUDA 13.

    Asks the driver directly, which needs no CUDA libraries, so it is
    cheap and answers the same way on either onnxruntime build.
    """
    global _gpu, _gpu_checked, _cuda_problem
    if _gpu_checked:
        return _gpu
    _gpu_checked = True
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32"
                             else "libcuda.so.1")
        count, device = ctypes.c_int(), ctypes.c_int()
        if (driver.cuInit(0) != 0
                or driver.cuDeviceGetCount(ctypes.byref(count)) != 0
                or count.value < 1
                or driver.cuDeviceGet(ctypes.byref(device), 0) != 0):
            return None
        name = ctypes.create_string_buffer(256)
        driver.cuDeviceGetName(name, len(name), device)
        major, minor = ctypes.c_int(), ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
        driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
        driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
    except (OSError, AttributeError):
        return None
    gpu = name.value.decode(errors="replace").strip() or "NVIDIA GPU"
    if (major.value, minor.value) < MIN_COMPUTE_CAPABILITY:
        _cuda_problem = f"the {gpu} is too old for CUDA 13"
        return None
    _gpu = gpu
    return _gpu


def cuda_installed() -> bool:
    """Whether the installed onnxruntime is the CUDA build."""
    import onnxruntime
    return CUDA in onnxruntime.get_available_providers()


_libraries_loaded = False


def use_cuda() -> bool:
    """Whether to run models on the GPU, loading the CUDA libraries the
    first time it says yes."""
    global _libraries_loaded
    if _cuda_problem is not None or not cuda_installed() or nvidia_gpu() is None:
        return False
    if not _libraries_loaded:
        import onnxruntime
        # CUDA and cuDNN come from the nvidia-* pip packages, which
        # onnxruntime does not find by itself.
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()
        _libraries_loaded = True
    return _cuda_problem is None


def cuda_failed(reason: str) -> None:
    """Stop using CUDA for the rest of the run, saying why once."""
    global _cuda_problem
    if _cuda_problem is None:
        _cuda_problem = reason
        warnings.warn(f"Running on the CPU: {reason}", RuntimeWarning,
                      stacklevel=2)


def device() -> str:
    """Which hardware the models run on, for showing to the user."""
    if _cuda_used and _cuda_problem is None:
        return f"GPU ({nvidia_gpu()})"
    if not cuda_installed():
        return "CPU"
    if nvidia_gpu() is None and _cuda_problem is None:
        return "CPU (no NVIDIA GPU found)"
    return f"CPU ({_cuda_problem})" if _cuda_problem else "CPU"


def note_cuda_used() -> None:
    global _cuda_used
    _cuda_used = True


def providers() -> List[str]:
    return [CUDA, CPU] if use_cuda() else [CPU]


def create_session(spec: ModelFile, quiet: bool = False):
    """An onnxruntime session for spec on the best available hardware.

    quiet hides onnxruntime's warnings, for models that trigger a harmless
    one on every run.
    """
    import onnxruntime
    options = onnxruntime.SessionOptions()
    if quiet:
        options.log_severity_level = 3
    path = model_path(spec)
    if use_cuda():
        try:
            session = onnxruntime.InferenceSession(path, options,
                                                   providers=[CUDA, CPU])
        except Exception as e:                              # noqa: BLE001
            cuda_failed(f"CUDA could not start ({e})")
        else:
            if CUDA in session.get_providers():
                note_cuda_used()
                return session
            cuda_failed("CUDA could not start (are the CUDA libraries "
                        "installed? see requirements-cuda.txt)")
            return session
    return onnxruntime.InferenceSession(path, options, providers=[CPU])


def _reset_for_tests() -> None:
    global _gpu, _gpu_checked, _cuda_problem, _cuda_used, _libraries_loaded
    _gpu, _gpu_checked, _cuda_problem = None, False, None
    _cuda_used = _libraries_loaded = False
