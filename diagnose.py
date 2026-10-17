"""
Diagnose

Checks that every model loads and runs, and on which hardware, writing a
report to a file. The installed app has no console, so this is how to see
inside it:

    ImageInspect.exe --diagnose photo.jpg report.txt

Models that are not downloaded yet are skipped rather than fetched.
"""

import os
import sys
import time
import traceback

import models


def _check(report, name, run):
    start = time.perf_counter()
    try:
        result = run()
    except Exception:                                       # noqa: BLE001
        report.append(f"{name}: FAILED\n{traceback.format_exc()}")
        return
    report.append(f"{name}: ok in {time.perf_counter() - start:.1f} s{result}")


def run(photo, out_path):
    import clip_search
    import describe
    import face_search
    from version import __version__

    report = [f"Image Inspect {__version__}, Python {sys.version.split()[0]}",
              f"Frozen: {getattr(sys, 'frozen', False)}",
              f"NVIDIA GPU: {models.nvidia_gpu()}",
              f"CUDA build: {models.cuda_installed()}"]

    def clip():
        if not all(models.is_present(s) for s in (
                clip_search.VISION_MODEL, clip_search.TEXT_MODEL,
                clip_search.TOKENIZER)):
            return " (skipped: not downloaded)"
        c = clip_search.Clip()
        c.embed_image_file(photo)
        c.embed_text("a photo")
        return f" on {c._vision.get_providers()[0]}"

    def faces():
        if not all(models.is_present(s) for s in (face_search.DETECTOR,
                                                    face_search.RECOGNIZER)):
            return " (skipped: not downloaded)"
        f = face_search.FaceModel()
        found = f.analyze_file(photo)
        return f" on {f._detector.get_providers()[0]}, {len(found)} faces"

    def description():
        variant = describe.preferred_variant()
        if not describe.downloaded(variant):
            return f" (skipped: {variant} build not downloaded)"
        d = describe.Describer(variant)
        text = d.describe(photo)
        return f" with the {d.variant} build:\n{text}"

    _check(report, "CLIP", clip)
    _check(report, "Faces", faces)
    _check(report, "Description", description)
    report.append(f"Device: {models.device()}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
        os.path.join(os.getcwd(), "diagnose.txt"))
