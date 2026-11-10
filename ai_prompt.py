"""
AI Image Prompts

Many image generators save the prompt an image was made from inside the
file. find_prompt() reads it back, whichever of these wrote it:

- AUTOMATIC1111, Forge, SD.Next and others in their style: a "parameters"
  PNG text chunk, or the EXIF UserComment of a JPEG or WebP. The prompt,
  then "Negative prompt: ...", then a "Steps: 20, Sampler: ..." line.
- ComfyUI: a "prompt" PNG chunk holding the node graph as JSON. The text is
  found by following each sampler's positive and negative inputs back to
  the text-encoding nodes that feed them. Older or edited files may carry
  only the "workflow" chunk (the editor's view of the graph).
- InvokeAI: "invokeai_metadata" JSON, or the older "sd-metadata" / "Dream".
- NovelAI: "Description" for the prompt and "Comment" JSON for the rest.
- Fooocus: "parameters" (or "fooocus") as JSON.
- Draw Things: JSON in the EXIF UserComment, prompt under "c".
- Midjourney: the prompt in the image description, ending "Job ID: ...".

Services such as ChatGPT/DALL-E and Adobe Firefly do not store the prompt,
and most websites and chat apps strip metadata when an image is uploaded,
so a missing prompt does not mean an image is not AI-generated.

PNG chunks are read directly rather than through Pillow: Pillow only sees
chunks after the image data once the pixels are decoded, and reading the
chunks costs next to nothing by comparison.
"""

import json
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import ExifTags, Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A text chunk larger than this is not metadata; skip rather than inflate it.
MAX_TEXT_CHUNK = 16 << 20
# How deep to follow ComfyUI links, against cycles in a damaged graph.
MAX_LINK_DEPTH = 30


@dataclass
class Prompt:
    """The prompt an image was generated from."""
    generator: str
    prompt: str
    negative: str = ""
    # Other settings worth showing (model, seed, steps...), in file order.
    settings: List[Tuple[str, str]] = field(default_factory=list)

    def as_text(self) -> str:
        """Everything, as plain text for copying or a screen reader."""
        parts = [self.prompt]
        if self.negative:
            parts.append(f"Negative prompt: {self.negative}")
        if self.settings:
            parts.append("\n".join(f"{k}: {v}" for k, v in self.settings))
        parts.append(f"Generator: {self.generator}")
        return "\n\n".join(parts)


def find_prompt(path: str) -> Optional[Prompt]:
    """The prompt saved in the image at path, or None if there is none.

    Never raises for a damaged or unreadable file; that just has no prompt.
    """
    try:
        texts = read_text_metadata(path)
    except Exception:                                        # noqa: BLE001
        return None
    for parser in PARSERS:
        try:
            found = parser(texts)
        except Exception:                                    # noqa: BLE001
            found = None    # One generator's odd data should not hide another's.
        if found is not None and found.prompt.strip():
            return found
    return None


# ============================================================================
# Reading the metadata
# ============================================================================

def read_text_metadata(path: str) -> Dict[str, str]:
    """The text metadata in the file, keyed by name.

    PNG text chunks keep their own keywords. From EXIF come "UserComment",
    "ImageDescription", "Software" and "Make"; from XMP, "xmp"; from a JPEG
    comment, "comment".
    """
    with open(path, "rb") as f:
        if f.read(8) == PNG_SIGNATURE:
            texts = _png_text_chunks(f)
            if texts:
                return texts
    return _pillow_metadata(path)


def _png_text_chunks(f) -> Dict[str, str]:
    """tEXt, zTXt and iTXt chunks, read from just after the signature."""
    texts = {}
    while True:
        header = f.read(8)
        if len(header) < 8:
            break
        length, kind = struct.unpack(">I4s", header)
        if kind == b"IEND":
            break
        if kind in (b"tEXt", b"zTXt", b"iTXt") and length <= MAX_TEXT_CHUNK:
            data = f.read(length)
            f.seek(4, 1)                                     # CRC
            try:
                key, value = _decode_text_chunk(kind, data)
            except (ValueError, zlib.error):
                continue
            texts.setdefault(key, value)
        else:
            f.seek(length + 4, 1)
    return texts


def _decode_text_chunk(kind: bytes, data: bytes) -> Tuple[str, str]:
    key, _, rest = data.partition(b"\0")
    key = key.decode("latin-1")
    if kind == b"tEXt":
        # Latin-1 by the spec, but generators write UTF-8 there anyway.
        return key, _utf8_or_latin1(rest)
    if kind == b"zTXt":
        return key, _utf8_or_latin1(_inflate(rest[1:]))
    # iTXt: compression flag, method, language\0, translated keyword\0, text
    compressed = rest[:1] == b"\1"
    _, _, rest = rest[2:].partition(b"\0")
    _, _, text = rest.partition(b"\0")
    if compressed:
        text = _inflate(text)
    return key, text.decode("utf-8", "replace")


def _inflate(data: bytes) -> bytes:
    d = zlib.decompressobj()
    out = d.decompress(data, MAX_TEXT_CHUNK)
    if d.unconsumed_tail:
        raise ValueError("text chunk too large")
    return out


def _utf8_or_latin1(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _pillow_metadata(path: str) -> Dict[str, str]:
    """EXIF, XMP and comments of a JPEG, WebP or other non-PNG image."""
    texts = {}
    with Image.open(path) as img:
        for key, value in img.info.items():
            if isinstance(value, str):
                texts[key] = value
            elif key in ("xmp", "comment") and isinstance(value, bytes):
                texts[key] = _utf8_or_latin1(value)
        try:
            exif = img.getexif()
        except Exception:                                    # noqa: BLE001
            return texts
        for tag in (ExifTags.Base.ImageDescription, ExifTags.Base.Software,
                    ExifTags.Base.Make):
            value = exif.get(tag)
            if isinstance(value, str) and value.strip("\0 "):
                texts[ExifTags.TAGS[tag]] = value.strip("\0")
        comment = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.UserComment)
        if comment:
            texts["UserComment"] = decode_user_comment(comment)
    return texts


def decode_user_comment(value) -> str:
    """An EXIF UserComment: 8 bytes naming the character set, then text."""
    if isinstance(value, str):
        return value.strip("\0")
    prefix, body = value[:8], value[8:]
    if prefix.startswith(b"UNICODE"):
        # UTF-16, in whichever byte order the writer chose; in Latin text
        # the zero byte of each pair gives it away.
        big = body[:2] in (b"\xfe\xff",) or (
            body[:2] != b"\xff\xfe" and body[0:1] == b"\0")
        text = body.decode("utf-16-be" if big else "utf-16-le", "replace")
        return text.lstrip("﻿").strip("\0")
    if prefix.startswith((b"ASCII", b"\0\0\0\0\0\0\0\0", b"JIS")):
        return _utf8_or_latin1(body).strip("\0")
    return _utf8_or_latin1(value).strip("\0")


# ============================================================================
# AUTOMATIC1111 style text
# ============================================================================

SETTINGS_LINE = re.compile(r"^\s*Steps: \d+")
# key: value, where a value may be "quoted, with commas"
SETTING = re.compile(r'\s*([\w ./()+-]+?):\s*("(?:\\.|[^"\\])*"|[^,]*)(?:,|$)')
SHOWN_A1111 = ("Model", "Steps", "Sampler", "Schedule type", "CFG scale",
               "Seed", "Size", "Denoising strength", "Hires upscale",
               "Clip skip", "Lora hashes")


def parse_a1111(text: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    """Split A1111-style text into (prompt, negative, settings)."""
    lines = text.strip().split("\n")
    settings = {}
    if lines and SETTINGS_LINE.match(lines[-1]):
        settings = {k.strip(): _unquote(v.strip())
                    for k, v in SETTING.findall(lines.pop())}
    elif not any(line.startswith("Negative prompt:") for line in lines):
        return None
    prompt, negative, in_negative = [], [], False
    for line in lines:
        if line.startswith("Negative prompt:"):
            in_negative = True
            line = line[len("Negative prompt:"):].lstrip()
        (negative if in_negative else prompt).append(line)
    return "\n".join(prompt).strip(), "\n".join(negative).strip(), settings


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except ValueError:
            return value[1:-1]
    return value


def _a1111_prompt(text: str, fallback_generator: str) -> Optional[Prompt]:
    parsed = parse_a1111(text)
    if parsed is None:
        return None
    prompt, negative, settings = parsed
    version = settings.get("Version", "")
    if version.startswith("f"):
        generator = "Stable Diffusion WebUI Forge"
    elif "ComfyUI" in version or "ComfyUI" in text:
        generator = "ComfyUI"
    else:
        generator = fallback_generator
    shown = [(k, settings[k]) for k in SHOWN_A1111 if settings.get(k)]
    return Prompt(generator, prompt, negative, shown)


def from_parameters(texts):
    text = texts.get("parameters")
    if not text:
        return None
    data = _json(text)
    if isinstance(data, dict):
        return _fooocus(data)
    return _a1111_prompt(text, "Stable Diffusion WebUI (AUTOMATIC1111 style)")


def from_user_comment(texts):
    text = texts.get("UserComment", "")
    data = _json(text)
    if isinstance(data, dict):
        if "c" in data:                                      # Draw Things
            return Prompt("Draw Things", str(data["c"]), str(data.get("uc", "")),
                          _pick(data, ("model", "Model"), ("steps", "Steps"),
                                ("sampler", "Sampler"), ("scale", "Guidance"),
                                ("seed", "Seed"), ("size", "Size")))
        if "prompt" in data:
            return _fooocus(data)
        return None
    return _a1111_prompt(text, "Stable Diffusion WebUI (AUTOMATIC1111 style)")


def _fooocus(data):
    prompt = data.get("prompt") or data.get("full_prompt")
    if isinstance(prompt, list):
        prompt = "\n".join(map(str, prompt))
    if not isinstance(prompt, str):
        return None
    negative = data.get("negative_prompt", "")
    if isinstance(negative, list):
        negative = "\n".join(map(str, negative))
    return Prompt("Fooocus", prompt, str(negative or ""),
                  _pick(data, ("base_model", "Model"), ("styles", "Styles"),
                        ("steps", "Steps"), ("sampler", "Sampler"),
                        ("guidance_scale", "Guidance"), ("seed", "Seed"),
                        ("resolution", "Size")))


def from_fooocus_chunk(texts):
    data = _json(texts.get("fooocus", ""))
    return _fooocus(data) if isinstance(data, dict) else None


# ============================================================================
# ComfyUI
# ============================================================================

# Input names under which a node passes text or conditioning along.
TEXT_INPUTS = ("text", "text_g", "text_l", "prompt", "positive", "string",
               "value", "text_a", "text_b", "conditioning", "conditioning_1",
               "conditioning_2", "conditioning_to", "conditioning_from",
               "clip_l", "t5xxl", "source")


def from_comfyui(texts):
    graph = _json(texts.get("prompt", ""))
    if isinstance(graph, dict) and graph:
        found = _comfy_api_graph(graph)
        if found is not None:
            return found
    workflow = _json(texts.get("workflow", ""))
    if isinstance(workflow, dict):
        return _comfy_workflow(workflow)
    return None


def _comfy_api_graph(graph):
    nodes = {k: v for k, v in graph.items()
             if isinstance(v, dict) and isinstance(v.get("inputs"), dict)}
    # Samplers take conditioning; so do nodes such as ControlNet apply that
    # pass it on, but only a sampler's conditioning goes no further.
    passes_on = {str(n["inputs"][k][0]) for n in nodes.values()
                 for k in ("positive", "negative") if _is_link(n["inputs"].get(k))}
    samplers = [n for key, n in nodes.items()
                if _is_link(n["inputs"].get("positive")) and key not in passes_on]
    for sampler in samplers:
        positive = _comfy_texts(nodes, sampler["inputs"]["positive"])
        if not positive:
            continue
        negative = _comfy_texts(nodes, sampler["inputs"].get("negative"))
        settings = []
        model = _comfy_model(nodes, sampler["inputs"].get("model"))
        if model:
            settings.append(("Model", model))
        for key, label in (("steps", "Steps"), ("sampler_name", "Sampler"),
                           ("scheduler", "Scheduler"), ("cfg", "CFG scale"),
                           ("seed", "Seed"), ("noise_seed", "Seed"),
                           ("denoise", "Denoising strength")):
            value = sampler["inputs"].get(key)
            if value is not None and not _is_link(value):
                settings.append((label, str(value)))
        return Prompt("ComfyUI", "\n\n".join(positive), "\n\n".join(negative),
                      settings)
    # No sampler found: whatever text was encoded, in node order.
    encoded = [n["inputs"][key] for n in nodes.values()
               if "TextEncode" in str(n.get("class_type", ""))
               for key in ("text", "prompt")
               if isinstance(n["inputs"].get(key), str)
               and n["inputs"][key].strip()]
    return Prompt("ComfyUI", "\n\n".join(encoded)) if encoded else None


def _is_link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[1], int))


def _comfy_texts(nodes, link, depth=0, seen=None):
    """The distinct strings that feed the input at link, in order."""
    seen = set() if seen is None else seen
    if not _is_link(link) or depth > MAX_LINK_DEPTH:
        return []
    node = nodes.get(str(link[0]))
    if node is None or str(link[0]) in seen:
        return []
    seen.add(str(link[0]))
    inputs = node["inputs"]
    if _is_link(inputs.get("positive")) and _is_link(inputs.get("negative")):
        # A node such as ControlNet apply passes both on: output 0 is the
        # positive, 1 the negative.
        return _comfy_texts(nodes, inputs["negative" if link[1] == 1
                                          else "positive"], depth + 1, seen)
    if "on_true" in inputs or "on_false" in inputs:
        # A switch passes on one of two inputs. When the choice is itself
        # wired in, it cannot be known here, so take both.
        switch = inputs.get("switch")
        names = (("on_true",) if switch is True else ("on_false",)
                 if switch is False else ("on_true", "on_false"))
        return _comfy_inputs(nodes, inputs, names, depth, seen)
    if "negative_prompt" in inputs and link[1] == 1:
        # An encoder such as TextEncodeQwenImage21 encodes both prompts:
        # output 0 is the positive, 1 the negative.
        return _comfy_inputs(nodes, inputs, ("negative_prompt",), depth, seen)
    return _comfy_inputs(nodes, inputs, TEXT_INPUTS, depth, seen)


def _comfy_inputs(nodes, inputs, names, depth, seen):
    """The distinct strings given to, or feeding, the named inputs."""
    found = []
    for name in names:
        value = inputs.get(name)
        if isinstance(value, str):
            if value.strip() and value not in found:
                found.append(value)
        elif _is_link(value):
            for text in _comfy_texts(nodes, value, depth + 1, seen):
                if text not in found:
                    found.append(text)
    return found


def _comfy_model(nodes, link, depth=0):
    if not _is_link(link) or depth > MAX_LINK_DEPTH:
        return ""
    node = nodes.get(str(link[0]))
    if node is None:
        return ""
    inputs = node["inputs"]
    for key in ("ckpt_name", "unet_name", "model_name"):
        if isinstance(inputs.get(key), str):
            return inputs[key]
    return _comfy_model(nodes, inputs.get("model"), depth + 1)


def _comfy_workflow(workflow):
    texts = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or "TextEncode" not in str(node.get("type")):
            continue
        for value in node.get("widgets_values") or []:
            if isinstance(value, str) and value.strip() and value not in texts:
                texts.append(value)
                break
    if not texts:
        return None
    # Without the graph, which is positive is a guess: the first, by custom.
    return Prompt("ComfyUI", texts[0], "\n\n".join(texts[1:2]))


# ============================================================================
# InvokeAI, NovelAI, Midjourney
# ============================================================================

def from_invokeai(texts):
    data = _json(texts.get("invokeai_metadata", ""))
    if isinstance(data, dict) and isinstance(data.get("positive_prompt"), str):
        model = data.get("model")
        settings = []
        if isinstance(model, dict) and model.get("name"):
            settings.append(("Model", str(model["name"])))
        settings += _pick(data, ("steps", "Steps"), ("scheduler", "Scheduler"),
                          ("cfg_scale", "CFG scale"), ("seed", "Seed"))
        if data.get("width") and data.get("height"):
            settings.append(("Size", f"{data['width']}x{data['height']}"))
        return Prompt("InvokeAI", data["positive_prompt"],
                      str(data.get("negative_prompt") or ""), settings)
    data = _json(texts.get("sd-metadata", ""))
    if isinstance(data, dict) and isinstance(data.get("image"), dict):
        image = data["image"]
        prompt = image.get("prompt")
        if isinstance(prompt, list):
            prompt = " ".join(str(p.get("prompt", "")) for p in prompt
                              if isinstance(p, dict))
        if isinstance(prompt, str):
            positive, negative = _split_brackets(prompt)
            return Prompt("InvokeAI", positive, negative,
                          _pick(image, ("steps", "Steps"),
                                ("sampler", "Sampler"), ("cfg_scale", "CFG scale"),
                                ("seed", "Seed")))
    dream = texts.get("Dream", "")
    match = re.match(r'\s*"(.*)"', dream, re.S)
    if match:
        positive, negative = _split_brackets(match.group(1))
        return Prompt("InvokeAI", positive, negative)
    return None


def _split_brackets(prompt):
    """InvokeAI's "a cat [blurry, dark]": the bracketed parts are negative."""
    negative = re.findall(r"\[([^\]]*)\]", prompt)
    positive = re.sub(r"\s*\[[^\]]*\]", "", prompt).strip()
    return positive, ", ".join(n.strip() for n in negative)


def from_novelai(texts):
    comment = _json(texts.get("Comment", ""))
    if texts.get("Software") != "NovelAI" and not (
            isinstance(comment, dict) and "uc" in comment):
        return None
    comment = comment if isinstance(comment, dict) else {}
    prompt = comment.get("prompt") or texts.get("Description", "")
    settings = []
    if texts.get("Source"):
        settings.append(("Model", texts["Source"]))
    settings += _pick(comment, ("steps", "Steps"), ("sampler", "Sampler"),
                      ("scale", "Guidance"), ("seed", "Seed"))
    if comment.get("width") and comment.get("height"):
        settings.append(("Size", f"{comment['width']}x{comment['height']}"))
    return Prompt("NovelAI", str(prompt), str(comment.get("uc", "")), settings)


MIDJOURNEY_JOB = re.compile(r"\s*Job ID:\s*([0-9a-f-]{36})\s*$", re.I)


def from_midjourney(texts):
    candidates = [texts.get("Description", ""), texts.get("ImageDescription", "")]
    xmp = texts.get("xmp", "")
    match = re.search(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", xmp, re.S)
    if match:
        candidates.append(_unescape_xml(match.group(1)))
    for text in candidates:
        job = MIDJOURNEY_JOB.search(text or "")
        if job:
            return Prompt("Midjourney", text[:job.start()].strip(),
                          settings=[("Job ID", job.group(1))])
    return None


def _unescape_xml(text):
    for entity, char in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                         ("&apos;", "'"), ("&#xA;", "\n"), ("&#10;", "\n"),
                         ("&amp;", "&")):
        text = text.replace(entity, char)
    return text


# ============================================================================

def _json(text):
    if not isinstance(text, str) or not text.lstrip().startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _pick(data, *keys):
    """[(label, str(value))] for each (key, label) present in data."""
    return [(label, str(data[key])) for key, label in keys
            if data.get(key) not in (None, "")]


# Most specific first: a ComfyUI file can also carry A1111-style
# "parameters" written by a custom node, but its graph is the truth.
PARSERS = (from_comfyui, from_invokeai, from_novelai, from_fooocus_chunk,
           from_parameters, from_user_comment, from_midjourney)
