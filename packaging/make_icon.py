"""Generates packaging/icon.ico: a magnifying glass over a grid of photos.

Re-run after editing it; the .ico is committed so a normal build does not
need this script.

    venv\\Scripts\\python packaging\\make_icon.py
"""
import os

from PIL import Image, ImageDraw

# Thumbnails in a few photo-like colours, so the grid reads as "pictures"
# even at 16 px.
TILES = [(86, 150, 206), (238, 170, 82), (108, 176, 116), (214, 104, 92)]
BACKGROUND = (30, 34, 44, 255)
LENS_RIM = (245, 245, 245, 255)
HANDLE = (200, 205, 215, 255)


def build(size=512):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 512

    draw.rounded_rectangle([28 * s, 28 * s, 484 * s, 484 * s],
                           radius=82 * s, fill=BACKGROUND)
    # A 2 x 2 grid of thumbnails.
    cell, gap, left, top = 170 * s, 22 * s, 70 * s, 70 * s
    for i, colour in enumerate(TILES):
        x = left + (i % 2) * (cell + gap)
        y = top + (i // 2) * (cell + gap)
        draw.rounded_rectangle([x, y, x + cell, y + cell], radius=22 * s,
                               fill=colour + (255,))

    # The magnifying glass over the bottom-right, its lens tinting what
    # it covers.
    cx, cy, r = 300 * s, 300 * s, 112 * s
    lens = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(lens).ellipse([cx - r, cy - r, cx + r, cy + r],
                                 fill=(255, 255, 255, 70))
    img = Image.alpha_composite(img, lens)
    draw = ImageDraw.Draw(img)
    width = round(30 * s)
    hx, hy = cx + r * 0.72, cy + r * 0.72
    draw.line([hx, hy, 452 * s, 452 * s], fill=HANDLE, width=round(54 * s))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=LENS_RIM, width=width)
    return img


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    build().save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                             (64, 64), (128, 128), (256, 256)])
    print(out)


if __name__ == "__main__":
    main()
