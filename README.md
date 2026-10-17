# Image Inspect

Search a folder of photos by description, by example, or both. Type a prompt
such as "a dog on the beach", add one or more guide images that look like what
you want, or combine the two, and the app shows thumbnails of the
best-matching images (top 100 by default), ranked by
[CLIP](https://openai.com/research/clip) similarity. Guide images can also
pick out a person, found by face recognition. Click a thumbnail to enlarge it.

## Searching

The sidebar on the left holds the search; results fill the rest of the
window.

- **Folder:** type or pick a folder; the list remembers the last ten
  searched, and *…* browses for another.
- **Search for:** *Things & scenes* matches what images look like (CLIP);
  *Person* finds photos of the person in the guide photos (face recognition).
- **Description:** what you are looking for, e.g. "a dog on the beach". Press
  Enter or Search. In *Person* mode it is optional and narrows the photos of
  the person, e.g. "at the beach".
- **Guide images** (*Looks like* / *Who*): click *+*, drop image files on
  the window, or right-click a result and choose *Add to guide images*. The
  file dialog opens where guides were last picked from. Double-click a guide
  to remove it. Several guide images are averaged, which in *Things & scenes*
  helps pick out what they have in common, such as a style of artwork, and in
  *Person* covers the person at different ages, angles and lighting.
- **Text / Images (or Person) slider:** with both a description and guide
  images, how much each counts.
- **Strictness** (*Person* only): how alike a face must be to count. Lower it
  if photos of the person are missing, raise it if strangers turn up.
- **Show:** the most results to list.
- Right-click a result, or use the buttons in its enlarged view, to *Find
  similar images* or *Find this person* in it. If a photo shows several
  people you are asked which one you mean, and the guide then shows that
  face; a guide in which no face was found is marked ⚠ and left out.
- Each thumbnail shows one score (the guide images' when there are any);
  hover for the full breakdown. Guide images that appear in the results are
  highlighted and marked ★. In *Person* mode the enlarged view boxes the face
  that matched.
- Progress and *Cancel* are in the status bar, if indexing new files is
  taking too long.

*Things & scenes* matches overall content and style well but does not
recognise individuals: guide photos of a person find similar scenes, clothes
and poses. Use *Person* for that.

## Keyboard and screen readers

Everything works from the keyboard, and every control has a name a screen
reader such as NVDA or Narrator reads out.

| Key | Action |
| --- | --- |
| Ctrl+O | Choose the folder to search |
| Ctrl+Shift+O | Add guide images |
| Enter (in the folder or description) or Ctrl+Enter | Search |
| Esc | Cancel a search |
| Alt + underlined letter | Move to that control (e.g. Alt+D for Description) |
| Enter on a result | Enlarge it |
| Menu key or Shift+F10 on a result or guide | Its menu (*Find similar*, *Find this person*, …) |
| Delete on a guide | Remove it |

Results and guides are read with their file name, scores, and where in the
photo the matching face is; the face picker names each person by their place
in the photo, left to right. Screen readers also hear what happens away from
the focus: guides added or removed, indexing progress (every 15 seconds or
so), the results, and warnings such as a guide with no face.

The window follows the Windows light or dark theme, high contrast, and the
*Accessibility > Text size* setting (read at start-up, so restart the app
after changing it). `ui_support.py`, which provides much of this, is copied
from ImageMosaic.

Supports the same formats as [ImageMosaic](https://github.com/cmillsap/ImageMosaic):
JPEG, PNG, TIFF, BMP, GIF, Canon `.cr2`, HEIC/HEIF and DNG (`image_io.py` is
copied from that project).

## Setup

```powershell
py -3.12 -m venv venv
venv\Scripts\pip install -r requirements.txt
run.bat                       # or: venv\Scripts\python app.py
venv\Scripts\python -m pytest  # run the tests (no models needed)
```

## How it works

- The first search downloads CLIP ViT-B/32 (fp16 ONNX, about 300 MB in total)
  to `%LOCALAPPDATA%\ImageInspect\models`. If ImageMosaic has already
  downloaded its vision model, that copy is reused.
- Each image's CLIP embedding and a 256 px thumbnail are cached in
  `%LOCALAPPDATA%\ImageInspect\index.sqlite`, keyed on path and modification
  time. The first search over a folder indexes it; later searches only process
  new or changed files, so they take moments. Files that cannot be decoded
  are remembered and skipped until they change. Cloud-only files (e.g. iCloud
  Photos) are downloaded the first time they are indexed, which can be slow.
  Delete the file to reset the cache.
- The number under each thumbnail is the cosine similarity between the image
  and the query. Text scores are compressed, typically 0.15-0.35; image scores
  run much higher, typically 0.6-1.0. Compare them with each other rather than
  reading them as absolute values. When combining text and images, each score
  list is standardised across the folder before weighting, so neither swamps
  the other.
- Person search uses InsightFace's *buffalo_l* models (`face_search.py`):
  SCRFD finds each face and its landmarks, and ArcFace turns the aligned face
  into an embedding. They are downloaded (about 190 MB) the first time you
  search for a person. A photo's score is the cosine similarity of its
  best-matching face to the guide faces (averaged): photos of the same person
  typically score 0.35-0.8, different people below about 0.2, and the default
  cut-off is 0.30. Faces are found in a separate pass, the first time a folder
  is searched for a person, at a few images a second; they are cached
  alongside the CLIP index, so later searches are quick.
- **Licence:** the InsightFace pretrained models are for non-commercial
  research use only; see
  [InsightFace's licence](https://github.com/deepinsight/insightface#license).
- All models run through onnxruntime on the CPU. `models.py` creates every
  session and prefers a GPU when the installed onnxruntime offers one, so
  swapping `onnxruntime` for `onnxruntime-directml` (any DirectX 12 GPU) or
  `onnxruntime-gpu` (NVIDIA CUDA) in `requirements.txt` should be all that
  GPU acceleration needs. That has not been tested yet.
