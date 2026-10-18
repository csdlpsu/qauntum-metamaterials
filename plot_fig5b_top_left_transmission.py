from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plotting_utils import plot_font as _font

from narrowband_directional_filter import compute_filter, matlab_colon


WIDTH = 1450
HEIGHT = 1170
PLOT_LEFT = 155
PLOT_TOP = 70
PLOT_RIGHT = 1085
PLOT_BOTTOM = 1000
COLORBAR_LEFT = 1115
COLORBAR_RIGHT = 1168
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = BLACK,
) -> None:
    width, height = _text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _paper_colormap(values: np.ndarray) -> np.ndarray:
    """Parula-like map used to match the paper panel's blue-cyan-yellow scale."""
    anchors = np.asarray(
        [
            [0.2081, 0.1663, 0.5292],
            [0.0780, 0.3300, 0.8260],
            [0.0000, 0.6800, 0.9000],
            [0.0000, 0.7700, 0.5500],
            [0.5000, 0.8000, 0.2500],
            [0.9500, 0.8600, 0.0500],
            [0.9763, 0.9831, 0.0538],
        ],
        dtype=float,
    )
    values = np.clip(values, 0.0, 1.0)
    xp = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(values, xp, anchors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1)


def _matrix_image(transmission: np.ndarray, size: tuple[int, int]) -> Image.Image:
    levels = np.linspace(0.0, 1.0, 101)
    displayed = np.flipud(transmission)
    quantized = levels[np.digitize(np.clip(displayed, 0.0, 1.0), levels, right=True)]
    rgb = (_paper_colormap(quantized) * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def _draw_axes(draw: ImageDraw.ImageDraw) -> None:
    axis_font = _font(48, bold=True)
    tick_font = _font(38)
    panel_font = _font(62, bold=True)

    plot_width = PLOT_RIGHT - PLOT_LEFT
    plot_height = PLOT_BOTTOM - PLOT_TOP
    draw.rectangle((PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM), outline=BLACK, width=3)

    for angle in range(0, 81, 20):
        x = PLOT_LEFT + plot_width * angle / 85
        draw.line((x, PLOT_BOTTOM, x, PLOT_BOTTOM + 14), fill=BLACK, width=3)
        _draw_centered(draw, (x, PLOT_BOTTOM + 44), f"{angle}\N{DEGREE SIGN}", tick_font)

    for wavelength in range(6, 13):
        y = PLOT_BOTTOM - plot_height * (wavelength - 6) / 6
        draw.line((PLOT_LEFT - 14, y, PLOT_LEFT, y), fill=BLACK, width=3)
        _draw_centered(draw, (PLOT_LEFT - 48, y), f"{wavelength}", tick_font)

    draw.text((PLOT_LEFT + 18, PLOT_TOP + 8), "TM", font=panel_font, fill=WHITE)
    _draw_centered(draw, ((PLOT_LEFT + PLOT_RIGHT) / 2, PLOT_BOTTOM + 95), "Angle", axis_font)

    y_label = Image.new("RGBA", (420, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(y_label)
    _draw_centered(label_draw, (210, 35), "Wavelength (\N{MICRO SIGN}m)", axis_font)
    y_label = y_label.rotate(90, expand=True)
    return y_label


def _draw_colorbar(draw: ImageDraw.ImageDraw) -> None:
    tick_font = _font(34)
    label_font = _font(44)
    bar_height = PLOT_BOTTOM - PLOT_TOP
    gradient = np.linspace(1.0, 0.0, bar_height)[:, None]
    rgb = (_paper_colormap(gradient)[:, 0, :] * 255).astype(np.uint8)
    bar = Image.fromarray(np.repeat(rgb[:, None, :], COLORBAR_RIGHT - COLORBAR_LEFT, axis=1), mode="RGB")
    return bar, tick_font, label_font


def render_plot(transmission: np.ndarray, output_base: Path) -> tuple[Path, Path]:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(canvas)

    plot = _matrix_image(transmission, (PLOT_RIGHT - PLOT_LEFT, PLOT_BOTTOM - PLOT_TOP))
    canvas.paste(plot, (PLOT_LEFT, PLOT_TOP))

    y_label = _draw_axes(draw)
    canvas.paste(y_label, (52, 420), y_label)

    bar, tick_font, _label_font = _draw_colorbar(draw)
    canvas.paste(bar, (COLORBAR_LEFT, PLOT_TOP))
    draw.rectangle((COLORBAR_LEFT, PLOT_TOP, COLORBAR_RIGHT, PLOT_BOTTOM), outline=BLACK, width=3)

    for value in np.linspace(0.0, 1.0, 6):
        y = PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * value
        draw.line((COLORBAR_RIGHT, y, COLORBAR_RIGHT + 14, y), fill=BLACK, width=3)
        label = f"{value:.1f}".rstrip("0").rstrip(".")
        width, height = _text_size(draw, label, tick_font)
        draw.text((COLORBAR_RIGHT + 25, y - height / 2), label, font=tick_font, fill=BLACK)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300)
    return png_path, pdf_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    wavelengths = matlab_colon(6e-6, 10e-9, 12e-6)
    theta_deg = matlab_colon(0, 1, 85)

    result = compute_filter(
        base_dir=base_dir,
        wavelength=wavelengths,
        theta_deg=theta_deg,
        low_index="CaF2",
        polarization="p",
        n_materials=2,
        vo2_insulating=True,
    )
    transmission = np.clip(result.transmission_matrix.real, 0.0, 1.0)
    png_path, pdf_path = render_plot(transmission, base_dir / "fig5b_top_left_TM_transmission_matrix")

    print(png_path)
    print(pdf_path)
    print(f"grid shape: {transmission.shape}")
    print(f"transmission min/max: {transmission.min():.6g} / {transmission.max():.6g}")


if __name__ == "__main__":
    main()
