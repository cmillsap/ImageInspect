"""
Face Search

Finds photos of a particular person. CLIP sees a face as part of the
scene, so it matches look-alike scenes rather than the person; this uses a
dedicated face pipeline instead, InsightFace's "buffalo_l" models:

- SCRFD-10G finds each face and five landmarks on it (eyes, nose tip,
  mouth corners).
- The face is warped so those landmarks sit where ArcFace expects them,
  and ArcFace (ResNet-50 trained on WebFace600K) turns it into a 512-d
  embedding. Two photos of the same person score a high cosine
  similarity even across age, pose and lighting; different people score
  near zero.

Both run through onnxruntime from the original ONNX exports, so the
`insightface` package (which needs a C++ build on Windows) is not needed;
the anchor decoding, non-maximum suppression and alignment it would do are
reimplemented here with numpy and Pillow.

The models are licensed by InsightFace for non-commercial research use only.

Faces are cached in the same SQLite file as the CLIP index, in tables of
their own, keyed on path and modification time like it. A photo can hold
any number of faces; one is scored by its best-matching face.
"""

import io
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

import models
from clip_search import CACHE_PATH
from image_io import open_image
from models import ModelFile, Progress

_BASE_URL = ("https://huggingface.co/public-data/insightface/resolve/"
             "33c1063c49c785b7652d3fd529f86fa4f149392b/models/buffalo_l/")

DETECTOR = ModelFile(
    "buffalo_l-det_10g.onnx", _BASE_URL + "det_10g.onnx",
    "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    16923827)
RECOGNIZER = ModelFile(
    "buffalo_l-w600k_r50.onnx", _BASE_URL + "w600k_r50.onnx",
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    174383860)

# Photos are analysed at up to this size. The detector itself looks at
# DETECT_SIZE, but faces are cut out of the larger image for recognition.
ANALYZE_SIZE = 1280
DETECT_SIZE = 640
DETECT_STRIDES = (8, 16, 32)
DETECT_ANCHORS = 2               # anchors per position in det_10g
DETECT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
# Faces smaller than this (in ANALYZE_SIZE pixels) are too blurry to
# recognise reliably and mostly add false matches.
MIN_FACE_SIZE = 24

FACE_SIZE = 112
# Where ArcFace expects the five landmarks in its 112 x 112 input.
ARCFACE_LANDMARKS = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                              [56.0252, 71.7366], [41.5493, 92.3655],
                              [70.7299, 92.2041]], np.float32)

# Photos of one person decades apart still score 0.35-0.8, while different
# people stay below about 0.2, so this sits between the two.
DEFAULT_MATCH_THRESHOLD = 0.30

CROP_SIZE = 128
BATCH_SIZE = 32                  # images decoded per batch
EMBED_BATCH = 64                 # faces per recognizer run


def ensure_models(progress: Progress = None) -> None:
    """Download whichever model files are missing."""
    models.ensure_models((DETECTOR, RECOGNIZER), progress)


class Face(NamedTuple):
    box: np.ndarray        # x1, y1, x2, y2 as fractions of the image size
    score: float           # detector confidence
    embedding: np.ndarray  # unit length, 512-d
    crop: Optional[bytes]  # JPEG of the face, when asked for


# ----------------------------------------------------------------------
# Preprocessing (pure numpy/Pillow, run on worker threads)
# ----------------------------------------------------------------------

def load_for_analysis(path: str) -> Image.Image:
    """path decoded upright, RGB, at most ANALYZE_SIZE on its long side."""
    with open_image(path, draft_size=(ANALYZE_SIZE, ANALYZE_SIZE)) as img:
        img = img.convert("RGB")
        img.thumbnail((ANALYZE_SIZE, ANALYZE_SIZE), Image.Resampling.BILINEAR)
        return img


def detector_input(image: Image.Image) -> Tuple[np.ndarray, float]:
    """image scaled to fit DETECT_SIZE, padded at the bottom right, and
    the scale applied."""
    width, height = image.size
    scale = DETECT_SIZE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = np.zeros((DETECT_SIZE, DETECT_SIZE, 3), np.uint8)
    canvas[:size[1], :size[0]] = np.asarray(
        image.resize(size, Image.Resampling.BILINEAR))
    return canvas, scale


def _prepare(path: str) -> Optional[Tuple[Image.Image, np.ndarray, float]]:
    try:
        image = load_for_analysis(path)
        return (image, *detector_input(image))
    except Exception as e:                                   # noqa: BLE001
        print(f"Skipping {path}: {e}")
        return None


def align(image: Image.Image, landmarks: np.ndarray) -> np.ndarray:
    """The face with these landmarks, warped to ArcFace's 112 x 112 layout.

    Fits the similarity transform (rotation, uniform scale, shift) taking
    the template landmarks to the detected ones, by least squares, and
    samples the image through it.
    """
    u, v = ARCFACE_LANDMARKS[:, 0], ARCFACE_LANDMARKS[:, 1]
    ones, zeros = np.ones_like(u), np.zeros_like(u)
    # x = a*u - b*v + c ;  y = b*u + a*v + d
    system = np.concatenate([np.stack([u, -v, ones, zeros], 1),
                             np.stack([v, u, zeros, ones], 1)])
    target = np.concatenate([landmarks[:, 0], landmarks[:, 1]])
    a, b, c, d = np.linalg.lstsq(system, target, rcond=None)[0]
    face = image.transform((FACE_SIZE, FACE_SIZE), Image.Transform.AFFINE,
                           (a, -b, c, b, a, d), Image.Resampling.BILINEAR)
    return np.asarray(face)


def face_crop(image: Image.Image, box: np.ndarray) -> bytes:
    """A square JPEG around box (in pixels), with some margin, for display."""
    x1, y1, x2, y2 = box
    side = 1.5 * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop = image.crop((round(cx - side / 2), round(cy - side / 2),
                       round(cx + side / 2), round(cy + side / 2)))
    crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _nms(boxes: np.ndarray, scores: np.ndarray) -> List[int]:
    """Indices of the boxes to keep, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]])
                       - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]])
                       - np.maximum(y1[i], y1[order[1:]]) + 1)
        overlap = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][overlap <= NMS_THRESHOLD]
    return keep


def _anchor_centers(stride: int) -> np.ndarray:
    side = DETECT_SIZE // stride
    ys, xs = np.mgrid[:side, :side]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
    return np.repeat(centers, DETECT_ANCHORS, axis=0).astype(np.float32)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

class FaceModel:
    """The SCRFD detector and ArcFace recognizer."""

    def __init__(self):
        self._detector = models.create_session(DETECTOR)
        # It declares a batch size of 1 but takes any, and onnxruntime
        # warns about that on every batch.
        self._recognizer = models.create_session(RECOGNIZER, quiet=True)
        self._anchors = {s: _anchor_centers(s) for s in DETECT_STRIDES}

    def detect(self, pixels: np.ndarray, scale: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Faces in a detector_input image, as (boxes, landmarks, scores),
        best first, in the coordinates of the image before scaling."""
        blob = ((pixels.astype(np.float32) - 127.5) / 128.0)
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        name = self._detector.get_inputs()[0].name
        outputs = self._detector.run(None, {name: blob})
        count = len(DETECT_STRIDES)
        boxes, landmarks, scores = [], [], []
        for i, stride in enumerate(DETECT_STRIDES):
            score = outputs[i][:, 0]
            keep = score >= DETECT_THRESHOLD
            if not keep.any():
                continue
            centers = self._anchors[stride][keep]
            distance = outputs[i + count][keep] * stride
            points = outputs[i + 2 * count][keep].reshape(-1, 5, 2) * stride
            boxes.append(np.concatenate([centers - distance[:, :2],
                                         centers + distance[:, 2:]], axis=1))
            landmarks.append(points + centers[:, np.newaxis])
            scores.append(score[keep])
        if not scores:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0, 5, 2), np.float32), np.zeros(0, np.float32))
        boxes = np.concatenate(boxes) / scale
        landmarks = np.concatenate(landmarks) / scale
        scores = np.concatenate(scores)
        keep = _nms(boxes, scores)
        return boxes[keep], landmarks[keep], scores[keep]

    def embed(self, faces: List[np.ndarray]) -> np.ndarray:
        """Embed aligned 112 x 112 faces, in batches."""
        name = self._recognizer.get_inputs()[0].name
        out = []
        for start in range(0, len(faces), EMBED_BATCH):
            batch = np.stack(faces[start:start + EMBED_BATCH]).astype(np.float32)
            batch = ((batch - 127.5) / 127.5).transpose(0, 3, 1, 2)
            out.append(self._recognizer.run(None, {name: batch})[0])
        if not out:
            return np.zeros((0, 512), np.float32)
        vectors = np.concatenate(out).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def find_faces(self, image: Image.Image, pixels: np.ndarray, scale: float
                   ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """(boxes in pixels, scores, aligned faces) for faces big enough to
        recognise."""
        boxes, landmarks, scores = self.detect(pixels, scale)
        big = np.minimum(boxes[:, 2] - boxes[:, 0],
                         boxes[:, 3] - boxes[:, 1]) >= MIN_FACE_SIZE
        boxes, landmarks, scores = boxes[big], landmarks[big], scores[big]
        return boxes, scores, [align(image, points) for points in landmarks]

    def analyze(self, image: Image.Image, crops: bool = False) -> List[Face]:
        """Every recognisable face in image, most confident first.

        An image cropped tightly around a face shows the detector too
        little around it to find, so when nothing is found the image is
        tried again with a border.
        """
        image = image.convert("RGB")
        size = np.array(image.size * 2, np.float32)
        boxes, scores, aligned = self.find_faces(image, *detector_input(image))
        if not len(boxes):
            pad = max(image.size) // 2
            padded = Image.new("RGB", (image.width + 2 * pad,
                                       image.height + 2 * pad))
            padded.paste(image, (pad, pad))
            boxes, scores, aligned = self.find_faces(padded,
                                                     *detector_input(padded))
            image, boxes = padded, boxes - pad
        embeddings = self.embed(aligned)
        return [Face(np.clip(box / size, 0, 1), float(score), embedding,
                     face_crop(image, box) if crops else None)
                for box, score, embedding in zip(boxes, scores, embeddings)]

    def analyze_file(self, path: str, crops: bool = False) -> List[Face]:
        return self.analyze(load_for_analysis(path), crops)


def person_template(embeddings: np.ndarray) -> np.ndarray:
    """One embedding standing for the person in several photos of them."""
    mean = embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)


# ----------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------

class FaceIndex:
    """The faces in a set of image files, backed by the SQLite cache.

    Like clip_search.Index, the cache is read into memory once. After
    build(), paths lists the files holding at least one face, and the faces
    of paths[i] are rows starts[i]:starts[i + 1] of embeddings and boxes.
    """

    def __init__(self, cache_path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        # ok is 0 for files that could not be decoded.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS face_files (path TEXT PRIMARY KEY, "
            "mtime REAL NOT NULL, ok INTEGER NOT NULL)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS faces (path TEXT NOT NULL, "
            "face INTEGER NOT NULL, box BLOB NOT NULL, score REAL NOT NULL, "
            "embedding BLOB NOT NULL, PRIMARY KEY (path, face))")
        # path -> (mtime, ok, boxes, embeddings)
        self._known: Optional[dict] = None
        self.paths: List[str] = []
        self.starts = np.zeros(1, np.int64)
        self.embeddings = np.zeros((0, 512), np.float32)
        self.boxes = np.zeros((0, 4), np.float32)
        self._row = {}                          # path -> i in self.paths

    def _load(self, progress: Progress) -> None:
        if self._known is not None:
            return
        if progress:
            progress("Loading face index", 0, 0)
        faces = {}
        for path, box, embedding in self._db.execute(
                "SELECT path, box, embedding FROM faces ORDER BY path, face"):
            boxes, embeddings = faces.setdefault(path, ([], []))
            boxes.append(np.frombuffer(box, np.float32))
            embeddings.append(np.frombuffer(embedding, np.float32))
        self._known = {}
        for path, mtime, ok in self._db.execute(
                "SELECT path, mtime, ok FROM face_files"):
            boxes, embeddings = faces.get(path, ([], []))
            self._known[path] = (mtime, bool(ok), *_stack(boxes, embeddings))

    def build(self, model: FaceModel, paths: List[str], progress: Progress = None,
              should_cancel: Callable[[], bool] = lambda: False) -> None:
        """Use cached faces for paths and find them in the rest."""
        self._load(progress)
        mtimes = {}
        for p in paths:
            try:
                mtimes[os.path.abspath(p)] = os.path.getmtime(p)
            except OSError:
                pass
        missing = [path for path, mtime in mtimes.items()
                   if self._known.get(path, (None,))[0] != mtime]

        done = 0
        with ThreadPoolExecutor() as pool:
            for start in range(0, len(missing), BATCH_SIZE):
                if should_cancel():
                    break
                chunk = missing[start:start + BATCH_SIZE]
                found = []        # (path, boxes, scores, first face, count)
                aligned = []
                for path, prepared in zip(chunk, pool.map(_prepare, chunk)):
                    done += 1
                    if progress:
                        progress(f"Finding faces ({os.path.basename(path)})",
                                 done, len(missing))
                    if prepared is None:
                        found.append((path, None, None, 0, 0))
                        continue
                    image, pixels, scale = prepared
                    boxes, scores, faces = model.find_faces(image, pixels, scale)
                    size = np.array(image.size * 2, np.float32)
                    found.append((path, np.clip(boxes / size, 0, 1).astype(np.float32),
                                  scores, len(aligned), len(faces)))
                    aligned.extend(faces)
                embeddings = model.embed(aligned)

                file_rows, face_rows = [], []
                for path, boxes, scores, first, count in found:
                    ok = boxes is not None
                    if not ok:
                        boxes = np.zeros((0, 4), np.float32)
                    vectors = embeddings[first:first + count]
                    self._known[path] = (mtimes[path], ok, boxes, vectors)
                    file_rows.append((path, mtimes[path], int(ok)))
                    face_rows.extend(
                        (path, i, box.tobytes(), float(score), vector.tobytes())
                        for i, (box, score, vector)
                        in enumerate(zip(boxes, scores, vectors)))
                with self._db:
                    self._db.executemany("DELETE FROM faces WHERE path = ?",
                                         [(r[0],) for r in file_rows])
                    self._db.executemany(
                        "INSERT OR REPLACE INTO face_files VALUES (?, ?, ?)",
                        file_rows)
                    self._db.executemany(
                        "INSERT INTO faces VALUES (?, ?, ?, ?, ?)", face_rows)

        self.paths, boxes, embeddings, counts = [], [], [], []
        for path, mtime in mtimes.items():
            known = self._known.get(path)
            if known is not None and known[0] == mtime and len(known[3]):
                self.paths.append(path)
                boxes.append(known[2])
                embeddings.append(known[3])
                counts.append(len(known[3]))
        self.boxes, self.embeddings = _stack(boxes, embeddings, concat=True)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._row = {path: i for i, path in enumerate(self.paths)}

    def person_scores(self, template: np.ndarray, paths: List[str]) -> np.ndarray:
        """Each of paths' best face similarity to template, or -1 where the
        file has no face (or has not been indexed)."""
        scores = np.full(len(paths), -1.0, np.float32)
        if not self.paths:
            return scores
        best = np.maximum.reduceat(self.embeddings @ template, self.starts[:-1])
        for i, path in enumerate(paths):
            row = self._row.get(path)
            if row is not None:
                scores[i] = best[row]
        return scores

    def best_box(self, template: np.ndarray, path: str) -> Optional[np.ndarray]:
        """The box of path's face most like template, if it has faces."""
        row = self._row.get(path)
        if row is None:
            return None
        start, end = self.starts[row], self.starts[row + 1]
        return self.boxes[start + int(np.argmax(self.embeddings[start:end]
                                                @ template))]


def _stack(boxes, embeddings, concat=False):
    """Lists of boxes and embeddings as arrays, allowing for none."""
    join = np.concatenate if concat else np.stack
    if not len(embeddings):
        return np.zeros((0, 4), np.float32), np.zeros((0, 512), np.float32)
    return join(boxes).astype(np.float32), join(embeddings).astype(np.float32)
