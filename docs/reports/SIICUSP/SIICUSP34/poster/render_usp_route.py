"""Render the app's current USP SVG at print resolution for the poster.

The SVG is the source of geometry, route, and order colors. This renderer only
adds the subtle map grid and rasterizes the existing visual at 300 dpi when
the image is printed 53 cm wide. No benchmark or solver data are changed.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "apps/siicusp34/data/usp-preview.svg"
OUTPUT = Path(__file__).with_name("figures") / "usp-route-print.png"
SCALE = 7.5
BG = (16, 44, 56)


def color(value: str) -> tuple[int, int, int]:
    if value.startswith("#"):
        return tuple(bytes.fromhex(value.lstrip("#")))  # type: ignore[return-value]
    match = re.fullmatch(r"hsl\((\d+) (\d+)% (\d+)% / ([.\d]+)\)", value)
    if not match:
        raise ValueError(f"Unsupported color: {value}")
    h, s, light, alpha = match.groups()
    rgb = colorsys.hls_to_rgb(int(h) / 360, int(light) / 100, int(s) / 100)
    opacity = float(alpha)
    return tuple(round(255 * component * opacity + base * (1 - opacity))
                 for component, base in zip(rgb, BG))  # type: ignore[return-value]


def points(value: str) -> list[tuple[int, int]]:
    return [tuple(round(float(v) * SCALE) for v in pair.split(","))
            for pair in value.split()]


def main() -> None:
    root = ET.parse(SOURCE).getroot()
    image = Image.new("RGB", (round(840 * SCALE), round(480 * SCALE)), BG)
    draw = ImageDraw.Draw(image)
    # Same 24-unit grid used by the app, subdued for print.
    for x in range(0, 841, 24):
        draw.line([(round(x * SCALE), 0), (round(x * SCALE), image.height)],
                  fill=(30, 59, 68), width=2)
    for y in range(0, 481, 24):
        draw.line([(0, round(y * SCALE)), (image.width, round(y * SCALE))],
                  fill=(30, 59, 68), width=2)
    for element in root:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "polygon":
            polygon = points(element.attrib["points"])
            draw.polygon(polygon, fill=color(element.attrib["fill"]))
            draw.line(polygon + polygon[:1], fill=color(element.attrib["stroke"]),
                      width=round(float(element.attrib["stroke-width"]) * SCALE),
                      joint="curve")
        elif tag == "polyline":
            route = points(element.attrib["points"])
            draw.line(route, fill=(15, 39, 48), width=round(6.8 * SCALE), joint="curve")
            draw.line(route, fill=color(element.attrib["stroke"]),
                      width=round(float(element.attrib["stroke-width"]) * SCALE),
                      joint="curve")
            radius = round(2 * SCALE)
            for x, y in route[1:-1]:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                             fill=color(element.attrib["stroke"]))
        elif tag == "circle":
            x, y = float(element.attrib["cx"]) * SCALE, float(element.attrib["cy"]) * SCALE
            radius = float(element.attrib["r"]) * SCALE
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         fill="white", outline=BG, width=round(2 * SCALE))
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), round(15 * SCALE))
        draw.text((round(349 * SCALE), round(251 * SCALE)), "IME  s = t",
                  font=font, fill="white", stroke_width=round(SCALE / 3),
                  stroke_fill=BG)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)


if __name__ == "__main__":
    main()
