"""
CLIP Search

Indexes a folder of images with OpenAI's CLIP ViT-B/32 and ranks them
against a text prompt. Both halves of the model run through onnxruntime,
from the same pinned HuggingFace checkpoint ImageMosaic uses, so the image
and text embeddings share one space.

Each image's embedding and a small JPEG thumbnail are cached in SQLite,
keyed on path and modification time, so only new or edited files are
decoded on later searches.
"""

import hashlib
import io
import os
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from image_io import open_image

# Same formats as ImageMosaic (tile_database.DEFAULT_IMAGE_EXTENSIONS).
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff',
                    '.cr2', '.heic', '.heif', '.dng')

_BASE_URL = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
             "d15189d7028b43f1d3e65039190477f6af591c2a/")

# (local filename, url, sha256 or None, size). The vision model is the very
# file ImageMosaic downloads, so an existing copy of it is reused.
VISION_MODEL = ("clip-vit-b32-vision-fp16.onnx",
                _BASE_URL + "onnx/vision_model_fp16.onnx",
                "35c4e0fb0aeee527dcde1693520b214a34424a786babd530f35366bad5844efd",
                176080659)
TEXT_MODEL = ("clip-vit-b32-text-fp16.onnx",
              _BASE_URL + "onnx/text_model_fp16.onnx",
              "df587ffbf248bf20d44fa6e16adc5ebc27ead691860e5333dbdaab5fd6bf3f6e",
              127339794)
TOKENIZER = ("clip-vit-b32-tokenizer.json", _BASE_URL + "tokenizer.json",
             None, 2224119)

APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                       "ImageInspect")
MODEL_DIR = os.path.join(APP_DIR, "models")
IMAGEMOSAIC_MODEL_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "ImageMosaic", "models")
CACHE_PATH = os.path.join(APP_DIR, "index.sqlite")

CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
CLIP_MAX_TOKENS = 77

THUMB_SIZE = 256
BATCH_SIZE = 32

Progress = Optional[Callable[[str, int, int], None]]


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def _model_file(spec) -> str:
    """Where a model file is, or will be once downloaded."""
    name, _, _, size = spec
    if spec is VISION_MODEL:
        shared = os.path.join(IMAGEMOSAIC_MODEL_DIR, name)
        if os.path.isfile(shared) and os.path.getsize(shared) == size:
            return shared
    return os.path.join(MODEL_DIR, name)


def _download(spec, progress: Progress) -> None:
    name, url, sha256, _ = spec
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_DIR, name)
    part_path = final_path + ".part"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(part_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(f"Downloading {name}", done, total)
        if sha256 and digest.hexdigest() != sha256:
            raise ValueError(f"Downloaded {name} is corrupt; please retry.")
        os.replace(part_path, final_path)
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)


def ensure_models(progress: Progress = None) -> None:
    """Download whichever model files are missing."""
    for spec in (VISION_MODEL, TEXT_MODEL, TOKENIZER):
        path = _model_file(spec)
        if not (os.path.isfile(path) and os.path.getsize(path) == spec[3]):
            _download(spec, progress)


class Clip:
    """The CLIP image and text encoders. Embeddings are unit length."""

    def __init__(self):
        import onnxruntime
        from tokenizers import Tokenizer

        providers = ["CPUExecutionProvider"]
        self._vision = onnxruntime.InferenceSession(
            _model_file(VISION_MODEL), providers=providers)
        self._text = onnxruntime.InferenceSession(
            _model_file(TEXT_MODEL), providers=providers)
        self._tokenizer = Tokenizer.from_file(_model_file(TOKENIZER))
        self._tokenizer.enable_truncation(CLIP_MAX_TOKENS)

    def embed_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Embed a batch of clip_input arrays."""
        batch = np.stack(images).astype(np.float32) / 255.0
        batch = ((batch - CLIP_MEAN) / CLIP_STD).transpose(0, 3, 1, 2)
        name = self._vision.get_inputs()[0].name
        return _normalize(self._vision.run(None, {name: batch})[0])

    def embed_text(self, text: str) -> np.ndarray:
        ids = np.array([self._tokenizer.encode(text).ids], np.int64)
        return _normalize(self._text.run(None, {"input_ids": ids})[0])[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------

def find_images(folder: str, recursive: bool) -> List[str]:
    if recursive:
        return [os.path.join(root, f)
                for root, _, files in os.walk(folder) for f in files
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            and os.path.isfile(os.path.join(folder, f))]


def clip_input(image: Image.Image) -> np.ndarray:
    """The image as CLIP sees it: short side resized to 224, centre-cropped."""
    image = image.convert("RGB")
    width, height = image.size
    scale = CLIP_SIZE / min(width, height)
    size = (max(CLIP_SIZE, round(width * scale)),
            max(CLIP_SIZE, round(height * scale)))
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = (size[0] - CLIP_SIZE) // 2
    top = (size[1] - CLIP_SIZE) // 2
    image = image.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    return np.asarray(image, dtype=np.uint8)


def _prepare(path: str) -> Optional[Tuple[np.ndarray, bytes]]:
    """Decode one image into (CLIP input, JPEG thumbnail), or None if unreadable."""
    try:
        with open_image(path, draft_size=(2 * THUMB_SIZE, 2 * THUMB_SIZE)) as img:
            img = img.convert("RGB")
            pixels = clip_input(img)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=85)
            return pixels, buffer.getvalue()
    except Exception as e:                                   # noqa: BLE001
        print(f"Skipping {path}: {e}")
        return None


# ----------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------

class Index:
    """Embeddings for a set of image files, backed by the SQLite cache."""

    def __init__(self, cache_path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, "
            "mtime REAL NOT NULL, embedding BLOB NOT NULL, thumb BLOB NOT NULL)")
        self.paths: List[str] = []
        self.embeddings = np.zeros((0, 512), np.float32)

    def build(self, clip: Clip, paths: List[str], progress: Progress = None,
              should_cancel: Callable[[], bool] = lambda: False) -> None:
        """Load cached embeddings for paths and compute the missing ones."""
        mtimes = {}
        for p in paths:
            try:
                mtimes[os.path.abspath(p)] = os.path.getmtime(p)
            except OSError:
                pass

        cached = {}
        for path, mtime, blob in self._db.execute(
                "SELECT path, mtime, embedding FROM images"):
            if mtimes.get(path) == mtime:
                cached[path] = np.frombuffer(blob, np.float32)

        missing = [p for p in mtimes if p not in cached]
        with ThreadPoolExecutor() as pool:
            for start in range(0, len(missing), BATCH_SIZE):
                if should_cancel():
                    break
                if progress:
                    progress("Indexing images", start, len(missing))
                chunk = missing[start:start + BATCH_SIZE]
                prepared = [(p, r) for p, r in zip(chunk, pool.map(_prepare, chunk))
                            if r is not None]
                if not prepared:
                    continue
                vectors = clip.embed_images([r[0] for _, r in prepared])
                rows = []
                for (path, (_, thumb)), vector in zip(prepared, vectors):
                    cached[path] = vector
                    rows.append((path, mtimes[path], vector.tobytes(), thumb))
                with self._db:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?)", rows)

        self.paths = list(cached)
        self.embeddings = (np.stack(list(cached.values())) if cached
                           else np.zeros((0, 512), np.float32))

    def search(self, query: np.ndarray, limit: int) -> List[Tuple[str, float]]:
        """The limit best (path, cosine similarity) matches, best first."""
        scores = self.embeddings @ query
        best = np.argsort(-scores)[:limit]
        return [(self.paths[i], float(scores[i])) for i in best]

    def thumbnail(self, path: str) -> Optional[bytes]:
        row = self._db.execute("SELECT thumb FROM images WHERE path = ?",
                               (path,)).fetchone()
        return row[0] if row else None
