from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plotting_utils import plot_font as _font

from narrowband_directional_filter import _default_thicknesses, compute_filter, matlab_colon


BOUNDS = ((0.0, 90.0), (2.0, 100.0), (2, 100))
THICKNESS_UNIT_M = 100e-9


def _validate_bounds(lthick_2_0: float, hthick_2_0: float, n_pairs_2_0: int) -> int:
    values = (lthick_2_0, hthick_2_0, n_pairs_2_0)
    names = ("lthick_2[0]", "hthick_2[0]", "n_pairs_2[0]")
    for name, value, (lower, upper) in zip(names, values, BOUNDS):
        if not lower <= value <= upper:
            raise ValueError(f"{name}={value} is outside the bound [{lower}, {upper}]")

    rounded_pairs = int(round(n_pairs_2_0))
    if not np.isclose(n_pairs_2_0, rounded_pairs):
        raise ValueError("n_pairs_2[0] must be an integer or integer-valued float")
    return rounded_pairs


def _draw_line_plot(theta_deg: np.ndarray, transmission: np.ndarray, output_path: Path) -> Path:
    width, height = 1200, 850
    left, top, right, bottom = 145, 70, 1080, 710
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    axis_font = _font(40, bold=True)
    tick_font = _font(30)

    draw.rectangle((left, top, right, bottom), outline="black", width=3)

    for angle in range(0, 86, 10):
        x = left + (right - left) * angle / 85
        draw.line((x, bottom, x, bottom + 10), fill="black", width=2)
        label = f"{angle}\N{DEGREE SIGN}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 18), label, font=tick_font, fill="black")

    for value in np.linspace(0, 1, 6):
        y = bottom - (bottom - top) * value
        draw.line((left - 10, y, left, y), fill="black", width=2)
        label = f"{value:.1f}".rstrip("0").rstrip(".")
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 22 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), label, font=tick_font, fill="black")

    clipped = np.clip(transmission, 0.0, 1.0)
    x = left + (right - left) * theta_deg / theta_deg[-1]
    y = bottom - (bottom - top) * clipped
    points = list(zip(x, y))
    if len(points) > 1:
        draw.line(points, fill=(30, 90, 210), width=5, joint="curve")

    title = "Transmission at wavelength = 9 \N{MICRO SIGN}m"
    box = draw.textbbox((0, 0), title, font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 20), title, font=axis_font, fill="black")

    x_label = "Angle"
    box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 780), x_label, font=axis_font, fill="black")

    y_label = "Transmission"
    label_img = Image.new("RGBA", (360, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    box = label_draw.textbbox((0, 0), y_label, font=axis_font)
    label_draw.text(((360 - (box[2] - box[0])) / 2, 8), y_label, font=axis_font, fill="black")
    label_img = label_img.rotate(90, expand=True)
    canvas.paste(label_img, (35, 260), label_img)

    canvas.save(output_path)
    return output_path


def transmission_theta_slice(
    lthick_2_0: float,
    hthick_2_0: float,
    n_pairs_2_0: int,
    *,
    plot: bool = True,
    polarization: str = "p",
    base_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Path]:
    """Return a transmission slice at wavelength = 9 um versus theta (TM by default).

    Parameters follow the requested bounded variables:
    - lthick_2_0 in units of 100 nm, bounded by [0, 90].
    - hthick_2_0 in units of 100 nm, bounded by [2, 100].
    - n_pairs_2_0 as an integer count, bounded by [2, 100].

    The paper values are lthick_2_0=25.5, hthick_2_0=4.5, n_pairs_2_0=5.
    """
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'")

    pair_count = _validate_bounds(lthick_2_0, hthick_2_0, n_pairs_2_0)
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent

    hthicks, lthicks, pair_numbers = _default_thicknesses()
    lthicks[1, 0] = lthick_2_0 * THICKNESS_UNIT_M
    hthicks[1, 0] = hthick_2_0 * THICKNESS_UNIT_M
    pair_numbers[1, 0] = pair_count

    theta_deg = matlab_colon(0, 1, 85)
    result = compute_filter(
        base_dir=root,
        wavelength=np.array([9e-6]),
        theta_deg=theta_deg,
        low_index="CaF2",
        polarization=polarization,
        n_materials=2,
        hthicks=hthicks,
        lthicks=lthicks,
        pair_numbers=pair_numbers,
    )

    transmission = result.transmission_matrix[0].real
    if not plot:
        return theta_deg, transmission

    output_path = root / f"transmission_theta_slice_9um_{polarization}.png"
    _draw_line_plot(theta_deg, transmission, output_path)
    return theta_deg, transmission, output_path


if __name__ == "__main__":
    theta, transmission, path = transmission_theta_slice(25.5, 4.5, 5, plot=True)
    print(path)
    print(f"theta shape: {theta.shape}")
    print(f"transmission min/max: {transmission.min():.6g} / {transmission.max():.6g}")
