"""Shared fonts for Pillow plots on macOS, Windows, and Linux."""

from pathlib import Path

from PIL import ImageFont


def plot_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use Times when available, with a scalable portable fallback."""
    times = "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf"
    candidates = [
        str(Path("/System/Library/Fonts/Supplemental") / times),
        "timesbd.ttf" if bold else "times.ttf",
        "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
