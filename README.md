# Image Inspect

Search a folder of photos by description, by example, or both. Type a prompt
such as "a dog on the beach", add one or more guide images that look like what
you want, or combine the two, and the app shows thumbnails of the
best-matching images (top 100 by default), ranked by
[CLIP](https://openai.com/research/clip) similarity. Guide images can also
pick out a person, found by face recognition. Click a thumbnail to enlarge it,
and ask for a written description of the photo from a language model that runs
on your computer.

## Install

Download an installer from the
[latest release](https://github.com/cmillsap/ImageInspect/releases/latest)
and run it. Python is not needed.

- **ImageInspect-x.y.z-GPU-Setup.exe** if the computer has an NVIDIA
  graphics card (RTX 20-series or newer): models run on the card, and a photo
  is described in seconds rather than half a minute. It is large because it
  carries NVIDIA's CUDA libraries.
- **ImageInspect-x.y.z-Setup.exe** otherwise. Everything works, on the
  processor.

Either installs over the other, and over an earlier version. It installs for
the current user, with no UAC prompt; choose *Install for all users* on the
first page to put it in Program Files instead. The models are downloaded the
first time each feature is used. Uninstalling asks whether to delete the
search index and downloaded models too.

The installers are not code-signed, so SmartScreen says "Windows protected
your PC" the first time: click *More info*, then *Run anyway*. Requires 64-bit
Windows 10 or later.

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

## Describing a photo

Press *Describe* in a photo's enlarged view, choose *Describe image* from a
result's menu, or press Ctrl+D on a result. The description is written beside
the photo by [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx),
a small vision language model that runs on your computer, so photos are not
sent anywhere. *Copy* puts it on the clipboard.

- The first time, you are asked before the model is downloaded: 2.6 GB for an
  NVIDIA graphics card or 3.2 GB for the processor, whichever this computer
  will use. *Stop* or closing the view cancels the download; it carries on
  from there next time.
- With a recent NVIDIA card a description takes a few seconds; on the
  processor, about half a minute (on a Ryzen 7 9700X: 2-3 s on an RTX 5060 Ti
  against 25-30 s). The model stays loaded until the app closes.
- Descriptions are remembered, so asking again is instant until the photo
  changes.
- Like any such model it can get details wrong; it often miscounts people.

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
| Ctrl+D on a result or in the enlarged view | Describe the photo |
| Menu key or Shift+F10 on a result or guide | Its menu (*Find similar*, *Find this person*, …) |
| Delete on a guide | Remove it |

Results and guides are read with their file name, scores, and where in the
photo the matching face is; the face picker names each person by their place
in the photo, left to right. Screen readers also hear what happens away from
the focus: guides added or removed, indexing progress (every 15 seconds or
so), the results, and warnings such as a guide with no face. A photo's
description is read out once it is written, and stays in the enlarged view's
*Description* box to review.

The window follows the Windows light or dark theme, high contrast, and the
*Accessibility > Text size* setting (read at start-up, so restart the app
after changing it). `ui_support.py`, which provides much of this, is copied
from ImageMosaic.

Supports the same formats as [ImageMosaic](https://github.com/cmillsap/ImageMosaic):
JPEG, PNG, TIFF, BMP, GIF, Canon `.cr2`, HEIC/HEIF and DNG (`image_io.py` is
copied from that project).

## Setup

```powershell
setup.bat                      # creates venv and installs the packages
run.bat                        # or: venv\Scripts\python app.py
venv\Scripts\python -m pytest  # run the tests (no models needed)
```

`setup.bat` installs the NVIDIA GPU (CUDA) build of onnxruntime if the
computer has an NVIDIA driver (`requirements-cuda.txt`, which brings about
1.6 GB of CUDA and cuDNN libraries with it) and the CPU build otherwise
(`requirements.txt`). `setup.bat cpu` or `setup.bat cuda` chooses for itself,
and running it again switches. The two builds cannot be installed side by
side.

## Building the installers

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1               # both
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Variant cpu  # or one
```

Needs Python 3.12 (`py -3.12`) and Inno Setup 6
(`winget install JRSoftware.InnoSetup`). For each variant it makes a build
venv under `build\`, runs the tests, freezes the app with PyInstaller and
packs it with Inno Setup into `dist\`. The version comes from `version.py`.

`ImageInspect.exe --diagnose photo.jpg report.txt` checks that each
downloaded model loads and runs on the photo, and on which hardware, writing
the result to `report.txt`; useful as the installed app has no console.

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
- Descriptions use the int4-quantised ONNX builds of Phi-3.5-vision-instruct
  (MIT licence), run by onnxruntime-genai (`describe.py`). The photo is
  turned upright and scaled to 1344 px first, which is as much as the model
  sees, and goes through the app's own decoder, so HEIC and DNG work too.
  Descriptions are cached in `index.sqlite`.
- **GPU:** `models.py` chooses the hardware for every model. With the CUDA
  build installed and an NVIDIA GPU of the Turing generation (RTX 20-series)
  or newer, everything runs on the GPU: on an RTX 5060 Ti, CLIP indexes about
  ten times faster than on a Ryzen 7 9700X and finding faces about five times
  faster, with the same results. If CUDA cannot be used after all,
  onnxruntime quietly falls back to the CPU; the app notices, carries on on
  the CPU, and the status bar says which is in use and why (e.g. *Running on
  GPU (NVIDIA GeForce RTX 5060 Ti)*). Should only the description model's
  GPU build fail to load, the CPU build is offered instead. DirectML (AMD and
  Intel GPUs) is not supported: onnxruntime-genai's DirectML build lags well
  behind and needs an older onnxruntime.
- Downloads resume where they stopped and are checked against pinned
  SHA-256 hashes.
