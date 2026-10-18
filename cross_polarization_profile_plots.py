"""Evaluate supplied design parameters under both polarizations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plotting_utils import plot_font as _font

from narrowband_directional_filter import _default_thicknesses, compute_filter, load_material_data, matlab_colon


THICKNESS_UNIT_M = 100e-9


@dataclass(frozen=True)
class ParameterSet:
    name: str
    lthick_2: float
    hthick_2: float
    npairs_2: int


def _target(theta_deg: np.ndarray) -> np.ndarray:
    return (theta_deg >= 45.0).astype(float)


def transmission_profile(
    params: ParameterSet,
    polarization: str,
    theta_deg: np.ndarray,
    material_data: dict,
    base_dir: Path,
) -> np.ndarray:
    hthicks, lthicks, pair_numbers = _default_thicknesses()
    lthicks[1, 0] = params.lthick_2 * THICKNESS_UNIT_M
    hthicks[1, 0] = params.hthick_2 * THICKNESS_UNIT_M
    pair_numbers[1, 0] = params.npairs_2

    result = compute_filter(
        base_dir=base_dir,
        wavelength=np.array([9e-6]),
        theta_deg=theta_deg,
        low_index="CaF2",
        polarization=polarization,
        n_materials=2,
        material_data=material_data,
        hthicks=hthicks,
        lthicks=lthicks,
        pair_numbers=pair_numbers,
    )
    return result.transmission_matrix[0].real


def _loss_summary(theta_deg: np.ndarray, transmission: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(transmission, 0.0, 1.0)
    low = clipped[theta_deg <= 43.0]
    high = clipped[theta_deg >= 47.0]
    mean_low = float(np.mean(low))
    mean_high = float(np.mean(high))
    loss = float(np.mean((clipped - _target(theta_deg)) ** 2))
    return loss, mean_low, mean_high


def draw_profile(
    theta_deg: np.ndarray,
    transmission: np.ndarray,
    params: ParameterSet,
    polarization: str,
    output_path: Path,
) -> Path:
    width, height = 1300, 900
    left, top, right, bottom = 150, 90, 1180, 705
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    axis_font = _font(42, bold=True)
    tick_font = _font(30)
    note_font = _font(27)

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

    target_points = [
        (left + (right - left) * theta / 85, bottom - (bottom - top) * value)
        for theta, value in zip(theta_deg, _target(theta_deg))
    ]
    profile = np.clip(transmission, 0.0, 1.0)
    profile_points = [
        (left + (right - left) * theta / 85, bottom - (bottom - top) * value)
        for theta, value in zip(theta_deg, profile)
    ]
    draw.line(target_points, fill=(210, 60, 60), width=4)
    draw.line(profile_points, fill=(30, 90, 220), width=5, joint="curve")
    x_step = left + (right - left) * 45.0 / 85
    draw.line((x_step, top, x_step, bottom), fill=(80, 80, 80), width=2)

    pol_label = "TM" if polarization == "p" else "TE"
    title = f"{params.name} evaluated as {pol_label}"
    box = draw.textbbox((0, 0), title, font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 28), title, font=axis_font, fill="black")

    x_label = "Angle"
    box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 800), x_label, font=axis_font, fill="black")

    y_label = "Transmission"
    label_img = Image.new("RGBA", (360, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    box = label_draw.textbbox((0, 0), y_label, font=axis_font)
    label_draw.text(((360 - (box[2] - box[0])) / 2, 8), y_label, font=axis_font, fill="black")
    label_img = label_img.rotate(90, expand=True)
    canvas.paste(label_img, (35, 275), label_img)

    loss, mean_low, mean_high = _loss_summary(theta_deg, transmission)
    legend = "blue: evaluated profile    red: ideal target"
    summary = f"MSE={loss:.4g} | mean low={mean_low:.3g} | mean high={mean_high:.3g}"
    params_text = f"lthick_2={params.lthick_2:.3g}, hthick_2={params.hthick_2:.3g}, npairs_2={params.npairs_2}"
    draw.rectangle((left + 14, top + 14, left + 765, top + 112), fill="white", outline=(180, 180, 180), width=1)
    draw.text((left + 24, top + 20), legend, font=note_font, fill="black")
    draw.text((left + 24, top + 50), summary, font=note_font, fill="black")
    draw.text((left + 24, top + 80), params_text, font=note_font, fill="black")

    canvas.save(output_path)
    return output_path


def make_contact_sheet(paths: list[Path], output_path: Path) -> Path:
    thumbs = [Image.open(path).convert("RGB").resize((650, 450), Image.Resampling.LANCZOS) for path in paths]
    canvas = Image.new("RGB", (1300, 900), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % 2) * 650
        y = (index // 2) * 450
        canvas.paste(thumb, (x, y))
    canvas.save(output_path)
    return output_path


def generate_cross_polarization_plots(
    tm_params: ParameterSet, te_params: ParameterSet, base_dir: str | Path | None = None
) -> tuple[list[Path], Path]:
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    material_data = load_material_data(root)
    theta_deg = matlab_colon(0, 1, 85)

    outputs: list[Path] = []
    for params, params_slug in [(tm_params, "tm_params"), (te_params, "te_params")]:
        for polarization in ["p", "s"]:
            transmission = transmission_profile(params, polarization, theta_deg, material_data, root)
            output_path = root / f"profile_9um_{params_slug}_eval_{polarization}.png"
            draw_profile(theta_deg, transmission, params, polarization, output_path)
            outputs.append(output_path)
            loss, mean_low, mean_high = _loss_summary(theta_deg, transmission)
            print(f"{output_path.name}: MSE={loss:.6g}, mean_low={mean_low:.6g}, mean_high={mean_high:.6g}")

    contact_sheet = make_contact_sheet(outputs, root / "profile_9um_cross_polarization_2x2.png")
    return outputs, contact_sheet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate two user-supplied designs under TE and TM")
    parser.add_argument("--tm-params", nargs=3, type=float, required=True, metavar=("LOW", "HIGH", "PAIRS"))
    parser.add_argument("--te-params", nargs=3, type=float, required=True, metavar=("LOW", "HIGH", "PAIRS"))
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    from transmission_theta_slice_wrapper import _validate_bounds

    designs = []
    for label, values in (("TM design", args.tm_params), ("TE design", args.te_params)):
        pairs = _validate_bounds(*values)
        designs.append(ParameterSet(label, values[0], values[1], pairs))
    paths, sheet = generate_cross_polarization_plots(*designs, base_dir=args.base_dir)
    for path in paths:
        print(path)
    print(sheet)
