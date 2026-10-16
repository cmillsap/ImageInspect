# Image Inspect

Search a folder of photos by description, by example, or both. Type a prompt
such as "a dog on the beach", add one or more guide images that look like what
you want, or combine the two, and the app shows thumbnails of the
best-matching images (top 100 by default), ranked by
[CLIP](https://openai.com/research/clip) similarity. Click a thumbnail to
enlarge it.

## Searching

- **Text:** type a description and press Enter or Search.
- **Guide images:** click *Add guide images…*, drop image files on the window,
  or right-click a result and choose *Add to guide images*. *Find similar
  images* (right-click menu, or the enlarged view) searches with just that one
  image. Several guide images are averaged, which helps pick out what they have
  in common, such as a style of artwork.
- **Both:** with a description and guide images, the slider sets how much each
  counts. Each thumbnail then shows both scores.
- Guide images that appear in the results are highlighted and marked ★ guide.
- To search again, edit the description or guide images and press Search; the
  old results clear as soon as the new search starts. Search turns into Cancel
  while indexing, if new files are taking too long.

CLIP matches overall content and style well, but it does not recognise
individuals: guide photos of a person find similar scenes, clothes and poses
rather than reliably finding that one person.

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
