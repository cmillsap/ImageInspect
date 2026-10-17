"""
Models

Downloading model files and opening them with onnxruntime, shared by the
CLIP and face pipelines.

Every inference session is created by create_session(), so the choice of
hardware lives in one place. It prefers a GPU execution provider when the
installed onnxruntime build offers one (onnxruntime-gpu for CUDA,
onnxruntime-directml for DirectML) and otherwise runs on the CPU; the
callers keep decoding and preprocessing apart from inference and feed the
models fixed-size batches, which is what a GPU wants anyway.
"""

import hashlib
import os
import urllib.request
from typing import Callable, Iterable, List, NamedTuple, Optional

APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                       "ImageInspect")
MODEL_DIR = os.path.join(APP_DIR, "models")

# Most preferred first; only those the installed onnxruntime has are used.
PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider",
                       "CPUExecutionProvider")

Progress = Optional[Callable[[str, int, int], None]]


class ModelFile(NamedTuple):
    name: str                          # local filename
    url: str
    sha256: Optional[str]
    size: int
    shared_path: Optional[str] = None  # an existing copy to use instead


def model_path(spec: ModelFile) -> str:
    """Where a model file is, or will be once downloaded."""
    if (spec.shared_path and os.path.isfile(spec.shared_path)
            and os.path.getsize(spec.shared_path) == spec.size):
        return spec.shared_path
    return os.path.join(MODEL_DIR, spec.name)


def _download(spec: ModelFile, progress: Progress) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_DIR, spec.name)
    part_path = final_path + ".part"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response, \
                open(part_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(f"Downloading {spec.name}", done, total)
        if spec.sha256 and digest.hexdigest() != spec.sha256:
            raise ValueError(f"Downloaded {spec.name} is corrupt; please retry.")
        os.replace(part_path, final_path)
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)


def ensure_models(specs: Iterable[ModelFile], progress: Progress = None) -> None:
    """Download whichever of specs are missing."""
    for spec in specs:
        path = model_path(spec)
        if not (os.path.isfile(path) and os.path.getsize(path) == spec.size):
            _download(spec, progress)


def providers() -> List[str]:
    import onnxruntime
    available = onnxruntime.get_available_providers()
    return [p for p in PREFERRED_PROVIDERS if p in available]


def create_session(spec: ModelFile, quiet: bool = False):
    """An onnxruntime session for spec on the best available hardware.

    quiet hides onnxruntime's warnings, for models that trigger a harmless
    one on every run.
    """
    import onnxruntime
    chosen = providers()
    options = onnxruntime.SessionOptions()
    if quiet:
        options.log_severity_level = 3
    if "DmlExecutionProvider" in chosen:
        # DirectML does not support these (see the onnxruntime DirectML docs).
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(model_path(spec), options,
                                        providers=chosen)
