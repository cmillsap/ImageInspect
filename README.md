# Image Inspect

Search a folder of photos by description. Type a prompt such as "a dog on the
beach" and the app shows thumbnails of the best-matching images (top 100 by
default), ranked by [CLIP](https://openai.com/research/clip) similarity. Click
a thumbnail to enlarge it.

Supports the same formats as [ImageMosaic](https://github.com/cmillsap/ImageMosaic):
JPEG, PNG, TIFF, BMP, GIF, Canon `.cr2`, HEIC/HEIF and DNG (`image_io.py` is
copied from that project).

## Setup

```powershell
py -3.12 -m venv venv
venv\Scripts\pip install -r requirements.txt
run.bat                       # or: venv\Scripts\python app.py
```

## How it works

- The first search downloads CLIP ViT-B/32 (fp16 ONNX, about 300 MB in total)
  to `%LOCALAPPDATA%\ImageInspect\models`. If ImageMosaic has already
  downloaded its vision model, that copy is reused.
- Each image's CLIP embedding and a 256 px thumbnail are cached in
  `%LOCALAPPDATA%\ImageInspect\index.sqlite`, keyed on path and modification
  time. The first search over a folder indexes it; later searches only process
  new or changed files, so they take moments. Delete the file to reset the cache.
- The number under each thumbnail is the cosine similarity between the image
  and the prompt. CLIP scores are compressed, typically 0.15-0.35, so compare
  them with each other rather than reading them as absolute values.
