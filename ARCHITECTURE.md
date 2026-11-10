# How Image Inspect Works

A guide to the code, for someone about to change it. It covers what each
module does, how they fit together, how data flows through a search, and what
is stored where. The user-facing behaviour is in `README.md`; this document
explains the code behind it.

Version described: **1.1.1** (`version.py`).

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Repository map](#2-repository-map)
3. [Module dependencies](#3-module-dependencies)
4. [What is stored where](#4-what-is-stored-where)
5. [A search, end to end](#5-a-search-end-to-end)
6. [`models.py`: downloads and hardware](#6-modelspy-downloads-and-hardware)
7. [`image_io.py`: decoding every format](#7-image_iopy-decoding-every-format)
8. [`clip_search.py`: CLIP indexing and ranking](#8-clip_searchpy-clip-indexing-and-ranking)
9. [`face_search.py`: finding a person](#9-face_searchpy-finding-a-person)
10. [`describe.py`: local photo descriptions](#10-describepy-local-photo-descriptions)
11. [`ai_prompt.py`: prompts saved in AI images](#11-ai_promptpy-prompts-saved-in-ai-images)
12. [`app.py`: the user interface](#12-apppy-the-user-interface)
13. [`ui_support.py`: accessibility helpers](#13-ui_supportpy-accessibility-helpers)
14. [`diagnose.py`: checking an installed copy](#14-diagnosepy-checking-an-installed-copy)
15. [Threading model](#15-threading-model)
16. [Tests](#16-tests)
17. [Setup, running and packaging](#17-setup-running-and-packaging)
18. [Common changes and where to make them](#18-common-changes-and-where-to-make-them)

---

## 1. The big picture

Image Inspect is a PyQt6 desktop application for Windows that searches folders
of photos in three ways:

| Feature | Model | Runs through | Module |
| --- | --- | --- | --- |
| **Things & scenes:** match a text prompt and/or example ("guide") images | OpenAI CLIP ViT-B/32 (fp16 ONNX) | `onnxruntime` | `clip_search.py` |
| **Person:** find photos of the person in the guide images | InsightFace *buffalo_l*: SCRFD-10G detector + ArcFace R50 recogniser | `onnxruntime` | `face_search.py` |
| **Describe:** write a description of one photo | Microsoft Phi-3.5-vision-instruct (int4 ONNX) | `onnxruntime-genai` | `describe.py` |

A fourth feature reads text instead of running a model. **AI prompts**: if
an image was made by a generator such as Stable Diffusion, ComfyUI or
Midjourney, the prompt saved in its metadata is extracted and shown
(`ai_prompt.py`).

The design follows a few principles:

- **Everything runs locally.** Models are downloaded once from pinned
  Hugging Face revisions, checked against SHA-256 hashes, and run on the
  computer itself. No photo leaves the machine.
- **No PyTorch.** Every model is an ONNX export run by onnxruntime. The
  pre- and post-processing that libraries like `transformers` or `insightface`
  would normally do (CLIP normalisation, SCRFD anchor decoding, NMS, face
  alignment) is reimplemented in a few lines of numpy and Pillow. This keeps
  the install small and avoids C++ build steps on Windows.
- **Incremental indexing.** Embeddings, thumbnails, faces and descriptions
  are cached in one SQLite file, keyed on `(path, mtime)`. A second search
  over the same folder only processes files that are new or changed.
- **One place for each concern.** All image decoding goes through
  `image_io.open_image()`, all model sessions through
  `models.create_session()`, and all downloads through
  `models.ensure_models()`.
- **GPU when possible, CPU otherwise, without errors.** CUDA is used if the
  CUDA build of onnxruntime is installed and a recent enough NVIDIA GPU is
  present. A CUDA failure is detected and the app falls back quietly to the
  CPU.
- **Accessible by default.** Every control has a spoken name and keyboard
  access. Events away from the focus are announced to screen readers through
  Windows UI Automation. Colours meet WCAG contrast, and sizes follow the
  system font.

---

## 2. Repository map

```
ImageInspect/
├── app.py              Qt GUI: main window, workers, dialogs, entry point (main)
├── clip_search.py      CLIP model wrapper, folder scanning, embedding index, ranking
├── face_search.py      Face detection/recognition, face index, person scoring
├── describe.py         Phi-3.5-vision download, loading, generation, description cache
├── ai_prompt.py        Reads generator prompts from PNG/JPEG/WebP metadata
├── models.py           Model file specs, resumable verified downloads, CPU/CUDA choice
├── image_io.py         open_image(): Pillow + HEIC (pillow-heif) + DNG (rawpy), EXIF rotation
├── ui_support.py       Accessibility/theming helpers (shared with ImageMosaic)
├── diagnose.py         `--diagnose` self-test writing a report file
├── version.py          __version__
│
├── test_app.py         GUI tests (accessibility, keyboard, announcements, dialogs)
├── test_search.py      Folder scanning, Index.search, FaceIndex, geometry helpers
├── test_describe.py    Hardware selection, downloads, model files, describe cache/flow
├── test_ai_prompt.py   Every prompt format, plus damaged/absent metadata
├── test_ui_support.py  Window sizing, scaled fonts, labels, contrast, announcements
│
├── requirements-base.txt   Shared pinned deps (PyQt6, Pillow, pillow-heif, rawpy, numpy, tokenizers, pytest)
├── requirements.txt        base + onnxruntime + onnxruntime-genai          (CPU)
├── requirements-cuda.txt   base + onnxruntime-gpu[cuda,cudnn] + genai-cuda  (NVIDIA)
├── setup.bat               Creates venv/, picks CPU or CUDA build, installs
├── run.bat                 Runs app.py from the venv
│
└── packaging/
    ├── build.ps1           Builds both installers (venv → pytest → PyInstaller → Inno Setup)
    ├── ImageInspect.spec   PyInstaller one-folder spec (bundles CUDA DLLs for the GPU variant)
    ├── installer.iss       Inno Setup script (per-user install, CPU/GPU variants share AppId)
    ├── make_icon.py        Generates icon.ico with Pillow
    ├── icon.ico
    └── requirements-build.txt  PyInstaller
```

`ui_support.py` and `image_io.py` are copied from the sister project
[ImageMosaic](https://github.com/cmillsap/ImageMosaic). Keep them in sync
with it where possible. The CLIP vision model file is also shared:
if ImageMosaic has already downloaded it, Image Inspect uses that copy
instead of downloading its own.

---

## 3. Module dependencies

```
                         app.py  ──────────────► ui_support.py
              ┌───────────┼──────────┬───────────────┐
              ▼           ▼          ▼               ▼
        clip_search   face_search  describe      ai_prompt
              │   ▲       │  │        │          (Pillow only)
              │   └───────┘  │        │
              │  (CACHE_PATH)│        │
              ▼              ▼        ▼
            models ◄─────────┴────────┘
              │
        image_io (used by clip_search, face_search, describe, app)

        diagnose ──► models, clip_search, face_search, describe  (imported lazily)
```

- `models` and `image_io` are the leaves; they import no other project modules.
- `face_search` imports `CACHE_PATH` from `clip_search` so both write to the
  same SQLite file.
- `ai_prompt` uses only Pillow and the standard library. It does not need
  `image_io` because it reads metadata rather than pixels.
- `onnxruntime` and `onnxruntime_genai` are imported **inside functions**, not
  at the top of modules. Importing the app, or running most tests, therefore
  does not load the large native libraries.

---

## 4. What is stored where

### Application data: `%LOCALAPPDATA%\ImageInspect\`

Defined as `models.APP_DIR`, which falls back to `~` when `LOCALAPPDATA` is
unset.

```
%LOCALAPPDATA%\ImageInspect\
├── index.sqlite                           all caches (see below)
└── models\
    ├── clip-vit-b32-vision-fp16.onnx      ~176 MB (or ImageMosaic's copy is used)
    ├── clip-vit-b32-text-fp16.onnx        ~127 MB
    ├── clip-vit-b32-tokenizer.json        ~2 MB
    ├── buffalo_l-det_10g.onnx             ~17 MB
    ├── buffalo_l-w600k_r50.onnx           ~174 MB
    └── phi-3.5-vision\
        ├── cuda\…                         ~2.6 GB (GPU build)
        └── cpu\…                          ~3.2 GB (CPU build)
```

A download in progress is written to `<name>.part` and renamed only once its
size and hash have been checked.

### `index.sqlite` schema

Three classes open their own connection to the same file, each with
`check_same_thread=False`, and each creates its own tables:

| Table | Owner | Columns | Purpose |
| --- | --- | --- | --- |
| `images` | `clip_search.Index` | `path TEXT PK, mtime REAL, embedding BLOB, thumb BLOB` | 512-d float32 CLIP embedding (unit length) and a 256 px JPEG thumbnail |
| `failed` | `clip_search.Index` | `path TEXT PK, mtime REAL` | Files that would not decode; skipped until their mtime changes |
| `face_files` | `face_search.FaceIndex` | `path TEXT PK, mtime REAL, ok INTEGER` | Which files have had faces looked for (`ok=0`: could not decode) |
| `faces` | `face_search.FaceIndex` | `path, face INTEGER, box BLOB, score REAL, embedding BLOB` (PK `path, face`) | Each face: box as 4 float32 fractions of the image, detector score, 512-d ArcFace embedding |
| `descriptions` | `describe.DescriptionCache` | `path, model TEXT, mtime REAL, text TEXT` (PK `path, model`) | Descriptions already written, per model |

All paths are stored as `os.path.abspath(path)`. A cache entry is valid only
when the stored `mtime` equals the file's current `os.path.getmtime()`
exactly. Deleting `index.sqlite` resets every cache.

### User settings: `QSettings("ImageInspect", "ImageInspect")`

On Windows these are stored in `HKCU\Software\ImageInspect\ImageInspect`.

| Key | Meaning |
| --- | --- |
| `folders` | Every folder in the list |
| `unticked_folders` | The folders that are listed but unticked |
| `folder` | Legacy single-folder setting, read only if `folders` is absent |
| `recent_folders` | Up to 10 most recently searched folders, for the *+* button's menu |
| `recursive` | *Include subfolders* |
| `match` | `"look"` or `"person"` (the last mode used) |
| `image_weight` | Text/visual balance slider, 0–100 |
| `face_threshold` | Person strictness, 0.05–0.95 |
| `limit` | The *Show* count |
| `guide_folder` | Folder the guide-image file dialog last opened in |

The uninstaller offers to delete both the data folder and
`HKCU\Software\ImageInspect`.

---

## 5. A search, end to end

The main path through the program, from a click on *Search* to the results
grid:

```
MainWindow.search()                                   [GUI thread]
  │  validates folders/prompt/guides, saves settings,
  │  clears results, creates SearchWorker(self.state, …)
  ▼
SearchWorker.run()                                    [worker QThread]
  1. First search only: clip_search.ensure_models()  → downloads if needed
                        state["clip"]  = Clip()       → 2 onnxruntime sessions + tokenizer
                        state["index"] = Index()      → opens index.sqlite
  2. Person mode only:  person_template()             → see §9; may stop early to ask
                                                        which face is meant
  3. find_images_in(folders, recursive)               → list of image paths, unreachable folders
  4. index.build(clip, paths)                         → embeds only new/changed files
  5. text = clip.embed_text(prompt)                   (if a prompt was given)
  6a. Look mode:   guide embeddings (cached, or computed) → index.search(text, guides, …)
  6b. Person mode: face_index.build(…) → person_scores → index.search(text, None, …, person=…)
  7. emit finished_ok([(path, scores, thumbnail JPEG bytes, face box or None)])
  │
  ▼
MainWindow.show_results()                             [GUI thread]
  builds QListWidgetItems from cached thumbnails, writes the summary,
  announces it to screen readers
MainWindow.on_worker_done()
  clears self.worker, re-enables controls; if the worker set needs_choice,
  shows FacePickerDialog and starts the search again
```

Progress is reported through the `progress(str message, int done, int total)`
signal. `total == 0` means "busy, no fraction", and the progress bar then
shows an indeterminate animation. Cancelling calls `requestInterruption()`.
The indexing loops check `should_cancel` (bound to `isInterruptionRequested`)
between batches. A cancelled search still ranks the images indexed so far,
and the summary says so.

---

## 6. `models.py`: downloads and hardware

### Model file specs

```python
class ModelFile(NamedTuple):
    name: str                 # relative path under MODEL_DIR
    url: str                  # pinned to a Hugging Face commit hash
    sha256: Optional[str]     # None = size check only (the CLIP tokenizer)
    size: int                 # exact byte count
    shared_path: Optional[str] = None   # use this file instead if it exists with the right size
```

- `model_path(spec)` returns `shared_path` if that file exists with the
  expected size, and otherwise `MODEL_DIR/name`. This is how ImageMosaic's
  CLIP vision model is reused.
- `is_present(spec)` checks that the file exists with the exact expected size.
- `bytes_to_download(specs)` is the total still to fetch, minus whatever is
  already in each `.part` file.

### Resumable, verified downloads

`ensure_models(specs, progress, should_cancel, label)` downloads each missing
file in turn and reports one combined byte count across all of them.
`_download()` does the following:

1. If `<name>.part` exists, sends `Range: bytes=<done>-`.
2. On `206 Partial Content`, re-hashes the existing part, then appends.
   On any other status, starts again from zero.
3. Streams in 1 MB chunks, updating SHA-256 and progress. If `should_cancel()`
   returns true it raises `models.Cancelled` and keeps the `.part` file for
   next time.
4. When finished, checks the size and hash. If either is wrong it deletes the
   part and raises `ValueError`. Otherwise it renames the part into place with
   `os.replace`.

### Choosing hardware

The module-level state records whether a GPU was found and whether CUDA has
failed:

| Variable | Meaning |
| --- | --- |
| `_gpu`, `_gpu_checked` | Name of CUDA device 0, cached after the first lookup |
| `_cuda_problem` | Why CUDA is not used (once set, CUDA is never tried again this run) |
| `_cuda_used` | Some session actually ran on the GPU |
| `_libraries_loaded` | `onnxruntime.preload_dlls()` has been called |

- `nvidia_gpu()` calls the driver (`nvcuda.dll`) directly through `ctypes`:
  `cuInit`, `cuDeviceGetCount`, `cuDeviceGetName`, and the compute
  capability attributes 75 and 76. This needs no CUDA toolkit, so it works
  with either onnxruntime build. GPUs older than compute capability 7.5
  (Turing, the RTX 20-series) are rejected, because CUDA 13 does not
  support them.
- `cuda_installed()` checks whether `CUDAExecutionProvider` is among
  onnxruntime's available providers. It is only there in the `onnxruntime-gpu`
  build.
- `use_cuda()` returns true when CUDA is installed, a suitable GPU exists and
  CUDA has not failed. The first time it does, it calls
  `onnxruntime.preload_dlls()` so that the CUDA/cuDNN DLLs from the `nvidia-*`
  pip packages can be found.
- `create_session(spec, quiet)` is **the only place an
  `onnxruntime.InferenceSession` is created.** With CUDA it requests
  `[CUDA, CPU]`, then checks `session.get_providers()`: onnxruntime does
  not raise an error when CUDA fails to start, and quietly runs on the CPU
  instead. If CUDA is missing from the providers, or creating the session
  raised, `cuda_failed(reason)` records the problem and emits one
  `RuntimeWarning`, and every later session uses only the CPU.
- `device()` gives the human-readable string shown in the status bar, such
  as `GPU (NVIDIA GeForce RTX 5060 Ti)`, `CPU (no NVIDIA GPU found)` or
  `CPU (CUDA could not start …)`.
- `_reset_for_tests()` clears all of this state between tests.

DirectML (for AMD and Intel GPUs) is deliberately not supported, because the
DirectML build of onnxruntime-genai lags behind.

---

## 7. `image_io.py`: decoding every format

`open_image(path, draft_size=None) -> PIL.Image` is the only function in the
project that decodes pixels.

- **HEIC/HEIF:** `pillow_heif.register_heif_opener()` runs at import time,
  so Pillow can open these files directly. The registration is in this module
  rather than in `app.py` so that any process that imports `image_io` gets it.
- **DNG (`RAW_EXTENSIONS`):** Pillow would return only the small embedded
  preview. `_open_raw()` develops the sensor data with
  `rawpy.postprocess(use_camera_wb=True, half_size=True, output_bps=8)`.
  `half_size` skips demosaicing and gives a quarter-resolution image, which is
  far more than any caller needs and much faster.
- **`draft_size`** asks the JPEG decoder to decode at a reduced scale
  (`Image.draft`). It must be applied before the pixels are loaded, which is
  why it is a parameter here rather than something callers do afterwards.
  Callers pass a square so that the bound holds whichever way the photo is
  rotated.
- **EXIF orientation:** `_apply_exif_orientation()` rotates or flips the
  image upright when the Orientation tag is 2–8. Pillow does not do this by
  itself. pillow-heif and rawpy already apply the orientation, so only
  Pillow-native formats need it. A malformed EXIF block is ignored rather than
  making the photo unreadable.

The returned image works as a context manager (`with open_image(p) as img:`).

Supported extensions are listed in `clip_search.IMAGE_EXTENSIONS`:
`.png .jpg .jpeg .bmp .gif .tif .tiff .cr2 .heic .heif .dng`. These are the
same as ImageMosaic's.

---

## 8. `clip_search.py`: CLIP indexing and ranking

### The model: `Clip`

- The vision and text encoders are two fp16 ONNX files from
  `Xenova/clip-vit-base-patch32` at a pinned commit, so the image and text
  embeddings share one 512-d space. The tokenizer is that repo's
  `tokenizer.json`, loaded with the `tokenizers` library and truncated to 77
  tokens.
- `clip_input(img)` resizes the short side to 224 with bicubic resampling,
  centre-crops to 224×224, and returns a `uint8` array.
- `embed_images(batch)` scales to 0–1, normalises with CLIP's mean and std,
  converts to NCHW, runs the model, and L2-normalises the result.
- `embed_text(text)` tokenises, runs the text model with `input_ids`, and
  L2-normalises.
- `embed_image_file(path)` opens with `draft_size=(448, 448)` and embeds.

All embeddings are unit length, so a dot product is a cosine similarity.

### Finding files

- `find_images(folder, recursive)` uses `os.walk` or `os.listdir`, filtered
  by extension.
- `find_images_in(folders, recursive, progress)` scans several folders. It:
  - removes duplicates by `normcase(abspath(path))`, so overlapping folders, or
    one folder spelt two ways, list each file once;
  - returns `(paths, unreachable)`, where a missing folder or one that raises
    `OSError` is added to `unreachable` instead of failing the search.

### The index: `Index`

`Index` loads `index.sqlite` into memory once (`_load`):
`_known: {abspath: (mtime, embedding)}` and `_failed: {abspath: mtime}`.
Later searches in the same session only need to stat each file.

`build(clip, paths, progress, should_cancel)`:

1. Stats every path for its mtime.
2. Splits the paths into **current** (cached mtime matches) and **missing**
   (new, changed, or previously failed with a different mtime).
3. Processes `missing` in batches of 32:
   - `ThreadPoolExecutor.map(_prepare, chunk)` decodes the images in
     parallel. `_prepare` opens each image once and produces both the CLIP
     input array and a 256 px JPEG thumbnail at quality 85. A failure returns
     `None` and prints the reason.
   - Progress is reported for each file, not each batch, because a batch
     can take minutes when files are on cloud storage.
   - One `embed_images` call per batch. The new rows go into `images` and the
     failures into `failed`, in one transaction per batch, so a cancelled
     search keeps the work it finished.
4. Sets `self.paths` and `self.embeddings` (an `N×512` matrix) to exactly the
   files in this search. The in-memory cache can hold more than that, from
   other folders.

`embedding(path)` returns a cached, up-to-date embedding if there is one.
`SearchWorker` uses it for guide images that are inside the indexed folders,
and only embeds guides from elsewhere.

`thumbnail(path)` reads a thumbnail from SQLite on demand. Thumbnails are not
held in memory.

### Ranking: `Index.search`

```python
search(text, guides, image_weight, limit, person=None, person_threshold=0.0)
```

1. Builds up to two score vectors over `self.paths`:
   - `text`: `embeddings @ text_embedding`.
   - `image`: `embeddings @ normalize(mean(guide_embeddings))`. Several
     guides are averaged into one direction, which favours what they have in
     common.
   - `person` (person mode) is passed in already computed, and **replaces**
     `image`.
2. With one part, ranks by its raw cosine similarity.
3. With two parts, ranks by
   `image_weight · z(visual) + (1 − image_weight) · z(text)`, where `z`
   standardises each vector across the index (`(s − mean) / std`). Without
   this, image-to-image similarities (roughly 0.6–1.0) would swamp
   text-to-image ones (roughly 0.15–0.35).
4. In person mode, drops results whose person score is below
   `person_threshold`. Photos with no faces score −1, so they are always
   dropped.
5. Returns the top `limit` results as `[(path, {"text": …, "image"/"person": …})]`.
   These are the raw cosine scores, not the z-scores, so the UI can display
   them.

---

## 9. `face_search.py`: finding a person

CLIP treats a face as part of the scene, so for people it finds similar
scenes rather than the same person. Person mode uses a dedicated face
pipeline: InsightFace's *buffalo_l* ONNX models, with the `insightface`
package's preprocessing reimplemented here.

### Pipeline for one image

```
open_image → RGB, longest side ≤ 1280 (ANALYZE_SIZE)          load_for_analysis
   │
   ├─ detector_input: scale to fit 640×640, pad bottom/right with black → (pixels, scale)
   │
   ▼
FaceModel.detect                     SCRFD det_10g
   • input (x − 127.5)/128, NCHW
   • 9 outputs = 3 strides (8, 16, 32) × {scores, box distances, 5-point landmarks}
   • per stride: keep score ≥ 0.5, decode against anchor centres
     (2 anchors per grid cell, precomputed in _anchor_centers)
   • divide by scale → coordinates in the 1280 px image
   • _nms (IoU > 0.4 suppressed) → boxes, landmarks, scores, best first
   │
   ▼
FaceModel.find_faces
   • drop faces whose shorter side is < 24 px (MIN_FACE_SIZE): too blurry, cause false matches
   • align(): least-squares similarity transform taking ArcFace's 5 template
     landmarks (112×112) to the detected ones, then Image.transform(AFFINE)
   │
   ▼
FaceModel.embed                      ArcFace w600k_r50
   • (x − 127.5)/127.5, NCHW, batches of 64 → L2-normalised 512-d vectors
```

`FaceModel.analyze(image, crops)` is used for **guide** images. It does the
same as above and also:

- retries with the image padded by half its size on every side if no face is
  found. A photo cropped tightly to a face gives the detector too little
  context, and the padding fixes that;
- returns `Face(box_as_fractions, score, embedding, crop_jpeg)`, where
  `face_crop` produces a 128 px square crop with 1.5× margin, shown in the
  face picker and the guide list.

### The face index: `FaceIndex`

`FaceIndex` mirrors `clip_search.Index`: it loads `face_files` and `faces`
into memory once, and `build()` processes only new or changed files, 32
images per batch. Each batch decodes in a thread pool, then runs the detector
on each image and the recogniser once over all the batch's aligned faces.
Each batch's rows are replaced in one transaction: `DELETE` the old faces,
then insert the new ones.

After `build()`, the faces of every file in this search are stored as flat
arrays:

```
paths       = [p0, p1, p2, …]                (only files with ≥ 1 face)
starts      = [0, 3, 4, 6, …]                faces of paths[i] are rows starts[i]:starts[i+1]
embeddings  = (total_faces × 512)
boxes       = (total_faces × 4)
_row        = {path: i}
```

- `person_scores(template, paths)` computes `embeddings @ template` and
  takes the best face per photo with `np.maximum.reduceat(…, starts[:-1])`.
  The result is aligned to the given `paths` (the CLIP index's order), with −1
  for files that have no face.
- `best_box(template, path)` is the box of the photo's best-matching face.
  The enlarged view draws it, and screen readers hear where it is.
- `person_template(embeddings)` is the L2-normalised mean of the guide
  faces' embeddings, so several photos of a person cover different ages,
  angles and lighting.

`DEFAULT_MATCH_THRESHOLD = 0.30`: the same person typically scores
0.35–0.8, and different people below about 0.2. The *Strictness* slider
changes the threshold.

### Choosing the face in a guide (`SearchWorker.person_template`)

Person mode needs one face per guide image. The worker and window agree on
the face like this:

1. `SearchWorker.known_faces` (`{path_key: embedding or None}`) is seeded
   from the window's `guide_faces`, so guides that have been resolved are not
   analysed again.
2. For each unresolved guide, `analyze_file(path, crops=True)`:
   - 1 face: resolved;
   - 0 faces: recorded as `None` (the guide is ignored, shown with ⚠ and
     counted in the summary);
   - several faces: left unresolved.
3. Everything analysed is sent to the window through the `guide_faces`
   signal. If any guide had several faces, the worker sets
   `needs_choice = True` and returns **before** indexing.
4. `MainWindow.on_worker_done` sees `needs_choice` and shows a
   `FacePickerDialog` for each such guide. The faces are ordered left to
   right, and focus starts on the largest. The window stores the chosen face
   in `guide_faces` and calls `search()` again. This time every guide is
   resolved.
5. If no guide has a face, the worker fails with an explanatory message.

The pretrained InsightFace models are licensed for **non-commercial research
use only**.

---

## 10. `describe.py`: local photo descriptions

### Model files

Phi-3.5-vision-instruct ONNX has two int4 builds in the same Hugging Face
repo, each pinned to a commit:

- `cuda` from `gpu/gpu-int4-rtn-block-32/` (about 2.6 GB)
- `cpu` from `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/` (about 3.2 GB)

`_FILES` lists each of the 10 files with its size and SHA-256 per variant.
`model_files(variant)` turns that into `ModelFile`s under
`models/phi-3.5-vision/<variant>/`. Only the variant this computer will use is
downloaded (`preferred_variant()` returns `"cuda"` if `models.use_cuda()`).

Helpers: `downloaded()`, `download_size()`, `free_space()` (walks up to an
existing folder to find the drive), and `format_size()`.

### Generation: `Describer`

- `__init__(variant)` builds an `og.Config` from the model folder, sets the
  provider explicitly (CUDA or none, which means CPU), and loads `og.Model`.
  If the CUDA variant fails, it calls `models.cuda_failed(...)` and raises
  `GpuUnavailable`. Loading takes a few seconds and several GB of memory, so
  the loaded `Describer` is kept in `state["describer"]` for the rest of the
  session.
- `describe(path, on_text, should_cancel)`:
  1. `image_bytes(path)` decodes through `open_image`, which also handles
     HEIC and DNG, converts to RGB, limits the size to 1344 px (the most the
     model's crops can use), and encodes to PNG in memory.
  2. The multimodal processor combines that with `PROMPT`. This is the
     Phi-3.5 chat template with `<|image_1|>` and an instruction to describe
     the people or subjects, the setting, the action, any text, and notable
     details.
  3. Generates one token at a time, up to `MAX_TOKENS = 600`, with
     `max_length=7680`. Each token is decoded through a stream so `on_text`
     receives the text so far, which shows it appearing live in the UI. The
     method returns `None` if cancelled.

### Cache and orchestration

- `DescriptionCache` stores rows in the `descriptions` table, keyed by
  `(abspath, MODEL_NAME)` and validated by mtime.
- `describe_file(state, path, variant, progress, on_text, should_cancel)` is
  what the worker calls:
  1. If the description is cached, returns it.
  2. Otherwise, if no model is loaded (or the CPU build was requested while
     the GPU build is loaded), downloads the variant and loads it.
  3. If loading the GPU build raises `GpuUnavailable` and the CPU build is
     already downloaded, loads that instead. Otherwise the exception
     propagates, so the UI can ask before downloading 3.2 GB.
  4. Generates, caches the result unless cancelled, and returns it.

---

## 11. `ai_prompt.py`: prompts saved in AI images

`find_prompt(path) -> Optional[Prompt]` **never raises.** A damaged or
unreadable file has no prompt.

```python
@dataclass
class Prompt:
    generator: str                 # e.g. "ComfyUI", "Midjourney"
    prompt: str
    negative: str = ""
    settings: List[Tuple[str, str]]  # (label, value) in display order: Model, Steps, Seed…
    def as_text(self) -> str       # for the clipboard / screen readers
```

### Step 1: collect text metadata (`read_text_metadata`)

- **PNG:** `_png_text_chunks` reads the chunk list directly, without
  decoding pixels. It handles `tEXt`, `zTXt` (deflate) and `iTXt` (optionally
  compressed UTF-8). It reads chunks after `IDAT` too; Pillow would only see
  those once the pixels were decoded. Values over 16 MB are skipped, and
  decompression is bounded (`_inflate`), which protects against
  decompression bombs. Many generators write UTF-8 into `tEXt` even though
  the specification says Latin-1, so UTF-8 is tried first.
- **Other formats (JPEG, WebP…):** `_pillow_metadata` collects string
  `img.info` entries, XMP and comments, plus the EXIF `ImageDescription`,
  `Software`, `Make` and `UserComment`. `decode_user_comment` handles the
  8-byte charset prefix (`UNICODE` in either byte order, `ASCII`, `JIS`, or
  undefined).

The result is a `{key: text}` dict.

### Step 2: try each parser in order (`PARSERS`)

Most specific first. The first parser that returns a non-empty prompt wins,
and an exception inside one parser just moves on to the next.

| Order | Parser | Looks at | Recognises |
| --- | --- | --- | --- |
| 1 | `from_comfyui` | `prompt` (API graph JSON), else `workflow` | ComfyUI |
| 2 | `from_invokeai` | `invokeai_metadata`, `sd-metadata`, `Dream` | InvokeAI (new and old) |
| 3 | `from_novelai` | `Comment` JSON with `uc`, or `Software == "NovelAI"` | NovelAI |
| 4 | `from_fooocus_chunk` | `fooocus` JSON | Fooocus |
| 5 | `from_parameters` | `parameters`: JSON means Fooocus, text means A1111 style | A1111, Forge, SD.Next… |
| 6 | `from_user_comment` | EXIF `UserComment`: JSON with `c` means Draw Things, with `prompt` means Fooocus, otherwise A1111 text | JPEG/WebP from the above |
| 7 | `from_midjourney` | `Description` / `ImageDescription` / XMP `dc:description` ending in `Job ID: <uuid>` | Midjourney |

ComfyUI is first because a ComfyUI file can also contain an A1111-style
`parameters` chunk written by a custom node, and the graph is the reliable
source.

### Notable parsing details

- **A1111 text** (`parse_a1111`): the last line, if it matches
  `^Steps: \d+`, is the settings line, parsed by `SETTING`, a regex that
  allows quoted values containing commas. Everything after
  `Negative prompt:` is the negative prompt. Only the keys in `SHOWN_A1111`
  are displayed. A `Version` starting with `f` means Forge.
- **ComfyUI API graph** (`_comfy_api_graph`): finds the *terminal* samplers,
  which are nodes with a linked `positive` input whose output does not feed
  another node's `positive` or `negative`. Nodes such as *ControlNet Apply*
  pass conditioning along; samplers are where it ends. Then it walks back
  along `positive`/`negative` links (`_comfy_texts`) through the inputs
  named in `TEXT_INPUTS`, collecting distinct strings. Pass-through nodes
  that have both a positive and a negative input are followed by output
  index (0 is positive, 1 is negative), and so are encoders that take both a
  `prompt` and a `negative_prompt` (such as *TextEncodeQwenImage21*). A
  switch node (`on_true`/`on_false`) is followed along the branch its
  `switch` value picks, or along both if that value is wired in from
  another node. The walk is limited to depth 30, with
  a `seen` set to stop cycles. The model name is found by following the
  `model` links to a `ckpt_name`, `unet_name` or `model_name`.
- **ComfyUI workflow only** (`_comfy_workflow`): this format has no links to
  follow, so it takes the first `*TextEncode*` node's text as the positive
  prompt and the second as the negative.
- **InvokeAI legacy:** text in `[brackets]` is negative (`_split_brackets`).

---

## 12. `app.py`: the user interface

`main()` handles `--diagnose` (see §14). Otherwise it creates the
`QApplication`, sets the application name, version and icon, calls
`apply_windows_appearance`, and shows `MainWindow`. `BUNDLE_DIR` is the
folder containing the app, or PyInstaller's `_MEIPASS` when frozen, and is
used to find `packaging/icon.ico`.

### Shared state

`MainWindow.state` is one plain dict shared by every worker:

| Key | Value | Created by |
| --- | --- | --- |
| `clip` | `clip_search.Clip` | first search |
| `index` | `clip_search.Index` | first search |
| `faces` | `face_search.FaceModel` | first person search |
| `face_index` | `face_search.FaceIndex` | first person search |
| `describer` | `describe.Describer` | first description |
| `descriptions` | `describe.DescriptionCache` | first description request |

Each model is loaded the first time a feature needs it and kept until the app
closes.

### Classes

| Class | Role |
| --- | --- |
| `MainWindow` | The sidebar (folders, mode, description, guides, sliders, *Search*), the results grid, and the status bar (message, progress, elapsed time, *Cancel*, device). Owns the settings and starts workers. |
| `SearchWorker(QThread)` | One search (see §5). Signals: `progress`, `finished_ok(list)`, `failed(str)`, `guide_faces(list)`. |
| `DescriptionService(QObject)` | Runs one `DescribeWorker` at a time for any number of enlarged views. If a new request arrives while a worker is stopping, it is held in `pending` and started when the worker finishes. Handles the GPU→CPU fallback prompt. Emits `device_changed`. |
| `DescribeWorker(QThread)` | Calls `describe.describe_file`. Signals: `progress`, `text` (partial), `finished_ok`, `failed`. Sets `needs_cpu_model` on `GpuUnavailable`. `view` is the panel to update, or `None` once that panel has closed. |
| `ImageDialog(QDialog)` | The enlarged photo, scaled to 90%×85% of the screen, with the matched face boxed in amber. Buttons: *Find similar*, *Find this person*, *Prompt* (only if `ai_prompt.find_prompt` finds one), *Describe* (Ctrl+D), *Show in folder*, *Open in default viewer*. Side panels appear in a column next to the photo (`show_beside`), and the photo shrinks to make room. |
| `DescriptionPanel` | The live description text, status, progress, the AI accuracy note, *Stop* and *Copy*. Announces the finished description. |
| `PromptPanel` | The prompt, negative prompt, settings and generator, and *Copy prompt* (copies the prompt only). |
| `FacePickerDialog` | A grid of face crops from a multi-person guide, ordered left to right. Each is named for screen readers, e.g. "Person 2 of 3, top left, largest face". |
| `ThumbnailList(QListWidget)` | Icon grid used for folders, guides and results. Opens its context menu on the current item when asked from the keyboard, and paints a hint when empty. The hint is also set as the accessible description. |

### Data on list items

Results and guides store data on their `QListWidgetItem`s:

| Role | Results list | Guide list | Folder list |
| --- | --- | --- | --- |
| `UserRole` | path | path | folder |
| `UserRole + 1` | face box (fractions) or `None` | original square thumbnail `QPixmap` | — |
| `UserRole + 2` | whether it is a guide image (highlighted, ★) | — | — |
| `AccessibleTextRole` | spoken summary: rank, name, scores, face position | spoken name | — |

### `MainWindow.guide_faces`

`{path_key: (embedding, crop_pixmap)}`, or `(None, None)` for a guide with
no face. It survives between searches, so faces are not found again and a
face picked from a multi-person photo is remembered. Removing or clearing
guides removes their entries. `update_guide_items()` shows the chosen face
crop in person mode, or the photo with a ⚠ badge if no face was found.

### `update_controls()`

The single function that sets the state of every control. It runs after any
input changes and when a search starts or ends. Principles:

- Controls that do not apply are **disabled, not hidden**, so the layout does
  not move. Their labels are set to secondary colour with `set_tone` rather
  than disabled, because screen readers announce a disabled label as
  "unavailable".
- Labels, placeholders, accessible names and hints change with the mode
  ("Looks like" vs "Who", etc.).
- *Search* is enabled when not busy and either (person mode) there are guides,
  or (look mode) there is a prompt or guides.

### Keyboard

- `SHORTCUTS`: Ctrl+O (add folder), Ctrl+Shift+O (add guides),
  Ctrl+Enter (search), Esc (cancel). Each one is a `QAction` that **clicks
  the button**, so a shortcut can do nothing that clicking could not. The
  shortcut is added to the button's tooltip.
- Ctrl+D (`DESCRIBE_SHORTCUT`) describes the current result, both in the
  main window and in the enlarged view.
- Delete or Backspace removes the current folder or guide (widget-scoped
  actions).
- Every input has a buddy label with an `&` mnemonic for Alt access.

### Sizing: `apply_metrics()`

Every size is derived from the font height (`em`): gaps, sidebar width,
folder list height (four rows, measured using a temporary probe item),
guide list height, result cell size, and progress bar width. It runs again on
`FontChange`, together with `refresh_scaled_fonts`. On a palette or theme
change, `recolor_results()` reapplies the guide highlight colour.

### Announcements

`announce()` (see §13) is called for guides added or removed, folders added or
removed, search start, cancelling, results (with `important=True` so they can
interrupt), and failures. Long-running progress is announced at most every
15 s (`ANNOUNCE_PROGRESS_EVERY`), so a screen reader does not read out every
file.

---

## 13. `ui_support.py`: accessibility helpers

This module is shared with ImageMosaic. Its main parts:

- **Contrast:** `relative_luminance`, `contrast_ratio` (WCAG), and
  `readable_color(color, background)`, which moves a colour towards black or
  white in 5% steps until it reaches 4.5:1. `secondary_text_color` fades body
  text 35% towards the background, then applies the same contrast floor.
- **Tones:** `set_tone(label, "secondary" | "done" | "problem" | None)`
  colours a label through its palette (never a stylesheet) and installs a
  `_ToneKeeper` event filter that recolours it on palette, theme or parent
  changes. In high-contrast mode every tone is plain body text, and symbols
  carry the meaning instead.
- **Labels:**
  - `WidestTextLabel` reserves the width of its longest possible value, so a
    readout such as `0.00` does not shift the layout.
  - `WrapLabel` reserves the height its wrapped text needs.
  - `HintLabel` is a one-line tip that isn't drawn when it doesn't fit, but
    remains available as a tooltip and to screen readers.
  - `StatusLabel` shows ✓ or ⚠ plus a colour, while screen readers hear
    "Done:" or "Warning:" instead of the symbol.
- **Font scaling:** `scale_font(widget, factor, bold)` tags a widget so that
  `refresh_scaled_fonts(window)` can recompute its font from the window's font
  when the font changes.
- **Window placement:** `initial_window_rect` opens the window at the
  preferred size, but at most 90% of the available screen area, centred.
- **Windows appearance:** `apply_windows_appearance` switches to the Fusion
  style in dark mode when the default style is `windowsvista`, which would
  otherwise stay light. It also applies the *Accessibility > Text size*
  registry setting, which Qt ignores.
- **Screen reader announcements:** `announce(widget, text, important)` calls
  `UiaRaiseNotificationEvent` from `uiautomationcore.dll` through `ctypes`,
  on the window's HWND. It does nothing if no UIA client is listening. It
  frees the BSTRs and releases the provider through its vtable, and catches
  every error. `set_announcer()` replaces the sink, which tests use to record
  announcements.

---

## 14. `diagnose.py`: checking an installed copy

The installed app has no console, so `ImageInspect.exe --diagnose photo.jpg
report.txt` (handled in `app.main()`) writes a report instead of starting the
GUI. The report contains:

- the version, Python version, whether the app is frozen, the NVIDIA GPU name,
  and whether the CUDA build is installed;
- for CLIP, faces and description: run on the photo and record the time and
  the provider actually used, or `FAILED` with a traceback. Models that are
  not downloaded are skipped, never fetched;
- the final `models.device()`.

---

## 15. Threading model

- **GUI thread:** all widgets, `QSettings`, dialogs, and the signal handlers
  in `MainWindow` and `DescriptionService`.
- **`SearchWorker`:** at most one at a time (`MainWindow.worker`). While it
  runs, the sidebar and the *Show* box are disabled, and context menus,
  drops and removals are refused.
- **`DescribeWorker`:** at most one at a time (`DescriptionService.worker`).
  It can run **at the same time as** a search. Both use the shared `state`
  dict, but they touch different keys, and each SQLite connection is separate
  and opened with `check_same_thread=False`.
- **Thread pools:** `Index.build` and `FaceIndex.build` decode images with a
  `ThreadPoolExecutor`. The model inference itself runs on the worker thread,
  one batch at a time.
- **Cancellation** is cooperative: `requestInterruption()` sets the flag, and
  the code checks it through the `should_cancel` callable between batches,
  between download chunks, and between generated tokens.
- **Closing the window** interrupts and waits for the search worker, and
  `DescriptionService.shutdown()` does the same for the describe worker.
- Worker signals cross threads through Qt's queued connections. Note that
  `DescriptionService.run` connects the worker's signals to lambdas that check
  `worker.view`. A closed view sets this to `None`, so late signals are
  ignored safely.

---

## 16. Tests

```powershell
venv\Scripts\python -m pytest
```

The tests do **not** need the models: model sessions, onnxruntime, HTTP
downloads and the describer are replaced with fakes.

| File | Covers |
| --- | --- |
| `test_search.py` | `find_images_in` (several folders, overlap, missing folders, progress); `Index.search` with hand-made unit vectors (text alone, person replacing guides, threshold with text, limit); `FaceIndex` scoring and `best_box`; `person_template`; `_nms`, `align`, `detector_input` geometry. |
| `test_describe.py` | `FakeOnnxruntime` for the hardware choice (CUDA used or not, quiet fallback detected, retry on CPU, one warning); a fake `urlopen` for downloads (subfolders, combined progress, cancel keeps the part, resume, server that will not resume, corrupt file deleted, present files skipped); `_FILES` complete and pinned for both variants; image preparation; `DescriptionCache`; `describe_file` with a `FakeDescriber` (caching, loading once, cancel not cached, GPU→CPU fallback). |
| `test_ai_prompt.py` | Builds synthetic PNG/JPEG/WebP files for each generator: A1111/Forge (tEXt, zTXt, iTXt, after IDAT), ComfyUI (graph, precedence over `parameters`, combined conditioning, switch and two-prompt encoder, cycles, workflow only), InvokeAI new and old, NovelAI, Fooocus, Draw Things, Midjourney (description and XMP); camera photos, plain PNGs, damaged files, bad JSON; `as_text`. |
| `test_ui_support.py` | Window rect, scaled fonts, `WidestTextLabel`, `WrapLabel`, `HintLabel`, contrast maths, tone colours in light, dark and high contrast, `StatusLabel`, the announcer hook, and that the UIA path never raises. |
| `test_app.py` | A real `MainWindow` with temporary `QSettings`, `FakeWorker` and `FakeIndex` to show results without searching. Every input has a distinct accessible name; names follow the mode; result and guide items are spoken in full; tab order matches the screen; shortcuts and tooltips; announcements (nothing at startup, guides, results, cancel, throttled progress); `describe_place`; `FacePickerDialog`; `ImageDialog` description; *Prompt* button. |

`packaging/build.ps1` runs the whole suite in each build venv before
freezing, so a failing test stops the build.

---

## 17. Setup, running and packaging

### Development

```powershell
setup.bat            # venv with Python 3.12; CUDA build if nvidia-smi works, otherwise CPU
setup.bat cpu|cuda   # choose explicitly; running it again switches
run.bat              # venv\Scripts\python app.py
```

The CPU and CUDA builds of onnxruntime use the same module name and cannot be
installed together. `setup.bat` uninstalls the other build first.
Dependencies are pinned exactly in `requirements-base.txt` and the two
variant files.

### Installers (`packaging/build.ps1`)

For each variant (`cpu`, `cuda`, or both):

1. Create or update `build\venv-<variant>` with that variant's requirements
   plus PyInstaller.
2. Run pytest.
3. Freeze with `packaging\ImageInspect.spec` into
   `dist\<variant>\ImageInspect\`. This is a **one-folder** build, because a
   one-file build would unpack hundreds of MB, or GBs, on every launch.
   The spec:
   - detects the GPU variant by whether `site-packages\nvidia` exists, and if
     so bundles the CUDA 13 and cuDNN runtime DLLs and their licences, in the
     same layout so `onnxruntime.preload_dlls()` finds them;
   - bundles onnxruntime and onnxruntime-genai native libraries, `icon.ico`
     and `README.md`;
   - excludes heavy optional imports (other Qt bindings, torch, pytest,
     transformers…) and the TensorRT provider;
   - writes a Windows version resource from `version.py`.
4. Compile `packaging\installer.iss` with Inno Setup 6 into
   `dist\ImageInspect-<version>[-GPU]-Setup.exe`. The installer:
   - installs per user by default (no UAC prompt), with an option to install
     for all users;
   - uses the same `AppId` for both variants, so either upgrades or replaces
     the other. `{app}\_internal` is deleted first, so switching from GPU to
     CPU removes the CUDA libraries;
   - refuses to install before Windows 10, and warns when installing the GPU
     variant without `nvcuda.dll`;
   - offers on uninstall to delete `%LOCALAPPDATA%\ImageInspect` and the
     registry settings. The default is to keep them.

To change the version, edit `version.py`. The spec, `build.ps1` and the
installer all read it from there.

---

## 18. Common changes and where to make them

| Change | Where |
| --- | --- |
| Support a new image format | Add the extension to `clip_search.IMAGE_EXTENSIONS`. If Pillow can't decode it, add a branch in `image_io.open_image` (as for DNG). |
| Support a new AI generator's prompt | Write a `from_xxx(texts)` parser in `ai_prompt.py` returning a `Prompt`, and add it to `PARSERS` in the right precedence order. If its metadata lives somewhere new, extend `read_text_metadata`. Add a case to `test_ai_prompt.py`. |
| Swap or upgrade a model | Update the `ModelFile` (URL pinned to a commit, `sha256`, exact `size`). If the embedding size or preprocessing changes, the cached embeddings in `index.sqlite` are no longer valid. Rename the table or add a model column, as `descriptions` does. |
| Change how text and image scores are combined | `clip_search.Index.search` and `_standardize`. |
| Tune face matching | `face_search` constants: `DEFAULT_MATCH_THRESHOLD`, `DETECT_THRESHOLD`, `NMS_THRESHOLD`, `MIN_FACE_SIZE`, `ANALYZE_SIZE`. |
| Change the description prompt or length | `describe.PROMPT`, `MAX_TOKENS`. Cached descriptions are keyed by `MODEL_NAME`, so change it too if old descriptions should not be reused. |
| Add a keyboard shortcut | `SHORTCUTS` (bound to a button) in `app.py`, or a `QAction` like `describe_action`. Update the table in `README.md` and `TestKeyboard`. |
| Add a sidebar control | Create it in `MainWindow.__init__` with a buddy label (`label_for` / `heading`), add it to the sidebar layout, set its state in `update_controls`, save it in `search()` if it should persist, and check `TestScreenReaders` / `TestKeyboard` (every input must have a unique accessible name and a sensible tab position). |
| Add a new model-backed feature | Create its sessions with `models.create_session`, download with `models.ensure_models`, decode with `image_io.open_image`, cache in `index.sqlite` keyed on `(abspath, mtime)`, run it in a `QThread` that reports through `progress(str, int, int)` and checks `isInterruptionRequested`, and keep the loaded model in `MainWindow.state`. |

### Things to watch for

- **Cache validity depends on exact mtime equality.** Tools that preserve
  mtimes while changing content (some sync clients, `touch -r`) leave stale
  embeddings until `index.sqlite` is deleted.
- **Paths are compared in more than one way.** The index uses
  `os.path.abspath`, while the UI's `same_path` / `path_key` also apply
  `normcase`. On Windows, a path that differs only in letter case can
  therefore be a cache miss but a UI match.
- **`Index.paths` is per search, but the in-memory cache is global.**
  `_known` keeps everything loaded from the database, and `paths` and
  `embeddings` are rebuilt on every `build()` to cover only the files in the
  current search.
- **CUDA state is process-wide and only goes one way.** Once
  `models.cuda_failed` has been called, nothing uses the GPU again until the
  app restarts. Tests must call `models._reset_for_tests()`.
- **Heavy imports stay inside functions.** `onnxruntime`,
  `onnxruntime_genai` and `tokenizers` are imported inside functions, so
  startup and tests stay fast. Keep it that way.
