"""
Tests for reading the prompts AI image generators save in their images,
using small files written the way each generator writes them.
"""

import json

import pytest
from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo

import ai_prompt
from ai_prompt import find_prompt

A1111 = (
    "a lighthouse at dusk, (oil painting:1.2)\n"
    "dramatic sky\n"
    "Negative prompt: blurry, lowres\n"
    "Steps: 30, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, "
    "Seed: 1234, Size: 512x768, Model hash: abc123, Model: dreamshaper_8, "
    'Lora hashes: "detail: 1a2b, style: 3c4d", Version: v1.10.1')


def png(tmp_path, chunks=(), name="a.png", after_data=()):
    """A PNG with the given (keyword, text) chunks; zTXt/iTXt if marked."""
    info = PngInfo()
    for key, value, *kind in chunks:
        if kind == ["z"]:
            info.add_text(key, value, zip=True)
        elif kind == ["i"]:
            info.add_itxt(key, value, zip=True)
        else:
            info.add_text(key, value)
    path = tmp_path / name
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    if after_data:
        # Put chunks after the image data, as some tools do.
        import struct
        import zlib
        data = path.read_bytes()
        iend = data.rindex(b"IEND") - 4
        extra = b""
        for key, value in after_data:
            body = key.encode() + b"\0" + value.encode()
            extra += (struct.pack(">I", len(body)) + b"tEXt" + body
                      + struct.pack(">I", zlib.crc32(b"tEXt" + body)))
        path.write_bytes(data[:iend] + extra + data[iend:])
    return str(path)


def jpeg(tmp_path, user_comment=None, description=None, name="a.jpg"):
    exif = Image.Exif()
    if description is not None:
        exif[ExifTags.Base.ImageDescription] = description
    if user_comment is not None:
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.UserComment] = user_comment
    path = tmp_path / name
    Image.new("RGB", (8, 8)).save(path, exif=exif)
    return str(path)


class TestA1111:
    def test_png_parameters(self, tmp_path):
        found = find_prompt(png(tmp_path, [("parameters", A1111)]))
        assert found.generator.startswith("Stable Diffusion WebUI")
        assert found.prompt == "a lighthouse at dusk, (oil painting:1.2)\ndramatic sky"
        assert found.negative == "blurry, lowres"
        settings = dict(found.settings)
        assert settings["Model"] == "dreamshaper_8"
        assert settings["CFG scale"] == "7"
        assert settings["Lora hashes"] == "detail: 1a2b, style: 3c4d"
        assert "Model hash" not in settings

    def test_forge_is_named(self, tmp_path):
        text = A1111.replace("v1.10.1", "f2.0.1v1.10.1-previous-313")
        assert find_prompt(png(tmp_path, [("parameters", text)])).generator == \
            "Stable Diffusion WebUI Forge"

    def test_compressed_and_international_chunks(self, tmp_path):
        assert find_prompt(png(tmp_path, [("parameters", A1111, "z")])).negative
        text = A1111.replace("lighthouse", "phare 灯台")
        assert "灯台" in find_prompt(png(tmp_path, [("parameters", text, "i")])).prompt

    def test_chunk_after_image_data(self, tmp_path):
        found = find_prompt(png(tmp_path, after_data=[("parameters", A1111)]))
        assert found.negative == "blurry, lowres"

    def test_without_settings_or_negative(self, tmp_path):
        found = find_prompt(png(tmp_path, [("parameters",
                                            "a cat\nNegative prompt: dog")]))
        assert (found.prompt, found.negative) == ("a cat", "dog")

    @pytest.mark.parametrize("encode", [
        lambda t: b"UNICODE\0" + t.encode("utf-16-be"),
        lambda t: b"UNICODE\0" + t.encode("utf-16-le"),
        lambda t: b"ASCII\0\0\0" + t.encode("ascii"),
    ])
    def test_jpeg_user_comment(self, tmp_path, encode):
        found = find_prompt(jpeg(tmp_path, user_comment=encode(A1111)))
        assert found.prompt.startswith("a lighthouse")
        assert dict(found.settings)["Seed"] == "1234"

    def test_webp(self, tmp_path):
        exif = Image.Exif()
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.UserComment] = \
            b"UNICODE\0" + A1111.encode("utf-16-be")
        path = tmp_path / "a.webp"
        Image.new("RGB", (8, 8)).save(path, exif=exif)
        assert find_prompt(str(path)).negative == "blurry, lowres"


def comfy_graph():
    """txt2img, with the negative routed through a ControlNet node."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sdxl_base.safetensors"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": ["20", 0], "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "watermark, text", "clip": ["4", 1]}},
        "20": {"class_type": "PrimitiveString", "inputs": {"value": "a red fox"}},
        "11": {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                          "control_net": ["12", 0], "image": ["13", 0]}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["11", 0],
                         "negative": ["11", 1], "latent_image": ["5", 0],
                         "seed": 42, "steps": 25, "cfg": 6.5,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0}},
    }


class TestComfyUI:
    def test_prompt_graph(self, tmp_path):
        found = find_prompt(png(tmp_path, [("prompt", json.dumps(comfy_graph())),
                                           ("workflow", "{}")]))
        assert found.generator == "ComfyUI"
        assert found.prompt == "a red fox"
        assert found.negative == "watermark, text"
        settings = dict(found.settings)
        assert settings["Model"] == "sdxl_base.safetensors"
        assert settings["Seed"] == "42"

    def test_graph_takes_precedence_over_parameters(self, tmp_path):
        found = find_prompt(png(tmp_path, [("parameters", A1111),
                                           ("prompt", json.dumps(comfy_graph()))]))
        assert found.prompt == "a red fox"

    def test_combined_conditioning(self, tmp_path):
        graph = comfy_graph()
        graph["8"] = {"class_type": "CLIPTextEncode",
                      "inputs": {"text": "in snow", "clip": ["4", 1]}}
        graph["9"] = {"class_type": "ConditioningCombine",
                      "inputs": {"conditioning_1": ["6", 0],
                                 "conditioning_2": ["8", 0]}}
        graph["3"]["inputs"]["positive"] = ["9", 0]
        found = find_prompt(png(tmp_path, [("prompt", json.dumps(graph))]))
        assert found.prompt == "a red fox\n\nin snow"

    def test_cycle_does_not_hang(self, tmp_path):
        graph = {"1": {"class_type": "X", "inputs": {"conditioning": ["2", 0]}},
                 "2": {"class_type": "X", "inputs": {"conditioning": ["1", 0]}},
                 "3": {"class_type": "KSampler",
                       "inputs": {"positive": ["1", 0], "negative": ["2", 0]}}}
        assert find_prompt(png(tmp_path, [("prompt", json.dumps(graph))])) is None

    def test_workflow_only(self, tmp_path):
        workflow = {"nodes": [
            {"type": "CLIPTextEncode", "widgets_values": ["a castle"]},
            {"type": "CLIPTextEncode", "widgets_values": ["fog"]},
            {"type": "KSampler", "widgets_values": [1, "fixed", 20]}]}
        found = find_prompt(png(tmp_path, [("workflow", json.dumps(workflow))]))
        assert (found.prompt, found.negative) == ("a castle", "fog")


class TestOthers:
    def test_invokeai(self, tmp_path):
        meta = {"positive_prompt": "a teapot", "negative_prompt": "cracks",
                "model": {"name": "juggernaut"}, "seed": 7, "steps": 30,
                "width": 1024, "height": 1024}
        found = find_prompt(png(tmp_path, [("invokeai_metadata", json.dumps(meta))]))
        assert (found.generator, found.prompt, found.negative) == \
            ("InvokeAI", "a teapot", "cracks")
        assert ("Size", "1024x1024") in found.settings

    def test_old_invokeai(self, tmp_path):
        found = find_prompt(png(tmp_path, [
            ("Dream", '"a teapot [cracks, dust]" -s 50 -S 7')]))
        assert (found.prompt, found.negative) == ("a teapot", "cracks, dust")

    def test_novelai(self, tmp_path):
        comment = {"prompt": "1girl, library", "uc": "bad hands",
                   "steps": 28, "scale": 5, "seed": 99, "sampler": "k_euler"}
        found = find_prompt(png(tmp_path, [
            ("Description", "1girl, library"), ("Software", "NovelAI"),
            ("Source", "NovelAI Diffusion V4"), ("Comment", json.dumps(comment))]))
        assert (found.generator, found.prompt, found.negative) == \
            ("NovelAI", "1girl, library", "bad hands")
        assert dict(found.settings)["Model"] == "NovelAI Diffusion V4"

    def test_fooocus(self, tmp_path):
        meta = {"prompt": "a bonsai", "negative_prompt": "", "styles": "['Fooocus V2']",
                "base_model": "juggernautXL", "seed": "5"}
        found = find_prompt(png(tmp_path, [("parameters", json.dumps(meta))]))
        assert (found.generator, found.prompt) == ("Fooocus", "a bonsai")

    def test_draw_things(self, tmp_path):
        meta = {"c": "a lantern", "uc": "people", "seed": 3, "steps": 20}
        found = find_prompt(jpeg(tmp_path, user_comment=b"ASCII\0\0\0"
                                 + json.dumps(meta).encode()))
        assert (found.generator, found.prompt, found.negative) == \
            ("Draw Things", "a lantern", "people")

    def test_midjourney(self, tmp_path):
        text = ("a koi pond, ukiyo-e --ar 3:2 --v 6.1 "
                "Job ID: 0f3c2a8e-1b2c-4d5e-8f90-a1b2c3d4e5f6")
        found = find_prompt(png(tmp_path, [("Description", text),
                                           ("Author", "someone")]))
        assert found.generator == "Midjourney"
        assert found.prompt == "a koi pond, ukiyo-e --ar 3:2 --v 6.1"

    def test_midjourney_xmp(self, tmp_path):
        xmp = ('<x:xmpmeta><rdf:RDF><rdf:Description><dc:description><rdf:Alt>'
               '<rdf:li xml:lang="x-default">a fox &amp; a hare --v 7 Job ID: '
               '0f3c2a8e-1b2c-4d5e-8f90-a1b2c3d4e5f6</rdf:li></rdf:Alt>'
               '</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>')
        path = tmp_path / "mj.webp"
        Image.new("RGB", (8, 8)).save(path, xmp=xmp.encode())
        assert find_prompt(str(path)).prompt == "a fox & a hare --v 7"


class TestNoPrompt:
    def test_camera_photo(self, tmp_path):
        assert find_prompt(jpeg(tmp_path, description="OLYMPUS DIGITAL CAMERA",
                                user_comment=b"ASCII\0\0\0holiday")) is None

    def test_plain_png(self, tmp_path):
        assert find_prompt(png(tmp_path, [("Software", "GIMP")])) is None

    def test_missing_and_damaged_files(self, tmp_path):
        assert find_prompt(str(tmp_path / "gone.png")) is None
        bad = tmp_path / "bad.png"
        bad.write_bytes(ai_prompt.PNG_SIGNATURE + b"\0\0\0\x05tEXtpar")
        assert find_prompt(str(bad)) is None

    def test_malformed_json_falls_through(self, tmp_path):
        found = find_prompt(png(tmp_path, [("prompt", "{not json"),
                                           ("parameters", A1111)]))
        assert found.prompt.startswith("a lighthouse")


def test_as_text():
    text = ai_prompt.Prompt("ComfyUI", "a fox", "blur", [("Seed", "1")]).as_text()
    assert text == "a fox\n\nNegative prompt: blur\n\nSeed: 1\n\nGenerator: ComfyUI"
