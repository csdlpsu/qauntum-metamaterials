"""Fit shared multilayer geometry for both TE and TM polarizations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plotting_utils import plot_font as _font

from narrowband_directional_filter import _default_thicknesses, compute_filter, load_material_data, matlab_colon


LOW_ANGLE_MAX = 43.0
HIGH_ANGLE_MIN = 47.0
STEP_ANGLE = 45.0
THICKNESS_UNIT_M = 100e-9
BOUNDS = np.asarray([[0.0, 90.0], [2.0, 100.0], [2.0, 100.0]], dtype=float)
FOCUS_ANGLES = np.asarray([46.0, 58.0, 70.0, 85.0], dtype=float)


@dataclass(frozen=True)
class DualCandidate:
    lthick_2: float
    hthick_2: float
    npairs_2: int
    loss: float
    p_step_loss: float
    s_step_loss: float
    agreement_loss: float
    p_mean_low: float
    p_mean_high: float
    s_mean_low: float
    s_mean_high: float


def _target(theta_deg: np.ndarray) -> np.ndarray:
    return (theta_deg >= STEP_ANGLE).astype(float)


def _angle_weights(theta_deg: np.ndarray) -> np.ndarray:
    weights = np.ones_like(theta_deg, dtype=float)
    for angle in FOCUS_ANGLES:
        weights += 3.0 * np.exp(-0.5 * ((theta_deg - angle) / 2.0) ** 2)
    return weights / np.mean(weights)


def _single_step_loss(theta_deg: np.ndarray, transmission: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(transmission.real, 0.0, 1.0)
    low_mask = theta_deg <= LOW_ANGLE_MAX
    high_mask = theta_deg >= HIGH_ANGLE_MIN
    target = _target(theta_deg)
    weights = _angle_weights(theta_deg)
    full_mse = float(np.mean((clipped - target) ** 2))
    focus_mse = float(np.average((clipped - target) ** 2, weights=weights))
    low_leakage = float(np.mean(clipped[low_mask] ** 2))
    total = full_mse + 0.5 * focus_mse + 1.5 * low_leakage
    return total, float(np.mean(clipped[low_mask])), float(np.mean(clipped[high_mask]))


def _profile(
    params: tuple[float, float, int],
    polarization: str,
    theta_deg: np.ndarray,
    material_data: dict,
    base_dir: Path,
) -> np.ndarray:
    lthick_2, hthick_2, npairs_2 = params
    hthicks, lthicks, pair_numbers = _default_thicknesses()
    lthicks[1, 0] = lthick_2 * THICKNESS_UNIT_M
    hthicks[1, 0] = hthick_2 * THICKNESS_UNIT_M
    pair_numbers[1, 0] = int(npairs_2)

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


def _evaluate(
    params: tuple[float, float, int],
    theta_deg: np.ndarray,
    material_data: dict,
    base_dir: Path,
) -> tuple[DualCandidate, np.ndarray, np.ndarray]:
    p_profile = _profile(params, "p", theta_deg, material_data, base_dir)
    s_profile = _profile(params, "s", theta_deg, material_data, base_dir)
    p_loss, p_mean_low, p_mean_high = _single_step_loss(theta_deg, p_profile)
    s_loss, s_mean_low, s_mean_high = _single_step_loss(theta_deg, s_profile)
    weights = _angle_weights(theta_deg)
    agreement_loss = float(np.average((np.clip(p_profile, 0, 1) - np.clip(s_profile, 0, 1)) ** 2, weights=weights))

    # Equal importance to both step fits; stronger agreement weight encourages polarization independence.
    total = p_loss + s_loss + 1.25 * agreement_loss
    candidate = DualCandidate(
        lthick_2=float(params[0]),
        hthick_2=float(params[1]),
        npairs_2=int(params[2]),
        loss=total,
        p_step_loss=p_loss,
        s_step_loss=s_loss,
        agreement_loss=agreement_loss,
        p_mean_low=p_mean_low,
        p_mean_high=p_mean_high,
        s_mean_low=s_mean_low,
        s_mean_high=s_mean_high,
    )
    return candidate, p_profile, s_profile


def _latin_hypercube(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.empty((n_samples, 3), dtype=float)
    for dim, (lower, upper) in enumerate(BOUNDS):
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        samples[:, dim] = lower + cut * (upper - lower)
    samples[:, 2] = np.rint(samples[:, 2])
    samples[:, 2] = np.clip(samples[:, 2], BOUNDS[2, 0], BOUNDS[2, 1])
    return samples


def _local_refine(seed: DualCandidate, theta_deg: np.ndarray, material_data: dict, base_dir: Path) -> DualCandidate:
    best = seed
    l_step = 6.0
    h_step = 6.0
    n_step = 8

    while l_step >= 0.25 or h_step >= 0.25 or n_step >= 1:
        improved = False
        for l_value in [best.lthick_2 - l_step, best.lthick_2, best.lthick_2 + l_step]:
            for h_value in [best.hthick_2 - h_step, best.hthick_2, best.hthick_2 + h_step]:
                for n_value in [best.npairs_2 - n_step, best.npairs_2, best.npairs_2 + n_step]:
                    l_value = float(np.clip(l_value, BOUNDS[0, 0], BOUNDS[0, 1]))
                    h_value = float(np.clip(h_value, BOUNDS[1, 0], BOUNDS[1, 1]))
                    n_value = int(np.clip(round(n_value), BOUNDS[2, 0], BOUNDS[2, 1]))
                    candidate, _, _ = _evaluate((l_value, h_value, n_value), theta_deg, material_data, base_dir)
                    if candidate.loss + 1e-12 < best.loss:
                        best = candidate
                        improved = True
        if not improved:
            l_step *= 0.5
            h_step *= 0.5
            n_step = max(0, n_step // 2)
        elif n_step == 0:
            n_step = 1

    return best


def _draw_profile(theta_deg: np.ndarray, p_profile: np.ndarray, s_profile: np.ndarray, candidate: DualCandidate, output_path: Path) -> Path:
    width, height = 1400, 940
    left, top, right, bottom = 155, 95, 1265, 735
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    axis_font = _font(42, bold=True)
    tick_font = _font(30)
    note_font = _font(26)

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

    target_points = [(left + (right - left) * theta / 85, bottom - (bottom - top) * val) for theta, val in zip(theta_deg, _target(theta_deg))]
    p_points = [(left + (right - left) * theta / 85, bottom - (bottom - top) * val) for theta, val in zip(theta_deg, np.clip(p_profile, 0, 1))]
    s_points = [(left + (right - left) * theta / 85, bottom - (bottom - top) * val) for theta, val in zip(theta_deg, np.clip(s_profile, 0, 1))]
    draw.line(target_points, fill=(210, 60, 60), width=4)
    draw.line(p_points, fill=(30, 90, 220), width=5, joint="curve")
    draw.line(s_points, fill=(0, 150, 90), width=5, joint="curve")
    x_step = left + (right - left) * STEP_ANGLE / 85
    draw.line((x_step, top, x_step, bottom), fill=(80, 80, 80), width=2)

    title = "Shared-Geometry 9 μm Step Fit Across TM and TE"
    box = draw.textbbox((0, 0), title, font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 28), title, font=axis_font, fill="black")
    box = draw.textbbox((0, 0), "Angle", font=axis_font)
    draw.text(((width - (box[2] - box[0])) / 2, 820), "Angle", font=axis_font, fill="black")

    label_img = Image.new("RGBA", (360, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    box = label_draw.textbbox((0, 0), "Transmission", font=axis_font)
    label_draw.text(((360 - (box[2] - box[0])) / 2, 8), "Transmission", font=axis_font, fill="black")
    label_img = label_img.rotate(90, expand=True)
    canvas.paste(label_img, (35, 285), label_img)

    legend = "blue: TM/p    green: TE/s    red: ideal target"
    summary = f"weighted loss={candidate.loss:.4g} | p step={candidate.p_step_loss:.4g} | s step={candidate.s_step_loss:.4g} | agreement={candidate.agreement_loss:.4g}"
    params = f"lthick_2={candidate.lthick_2:.3g}, hthick_2={candidate.hthick_2:.3g}, npairs_2={candidate.npairs_2}"
    means = f"mean high: TM={candidate.p_mean_high:.3g}, TE={candidate.s_mean_high:.3g} | mean low: TM={candidate.p_mean_low:.3g}, TE={candidate.s_mean_low:.3g}"
    draw.rectangle((left + 14, top + 14, left + 940, top + 140), outline=(180, 180, 180), width=1)
    for i, line in enumerate([legend, summary, params, means]):
        draw.text((left + 24, top + 20 + 30 * i), line, font=note_font, fill="black")
    canvas.save(output_path)
    return output_path


def optimize_dual_polarization_step_profile(
    *,
    n_samples: int = 1000,
    n_keep: int = 14,
    seed: int = 11,
    base_dir: str | Path | None = None,
) -> tuple[DualCandidate, np.ndarray, np.ndarray, np.ndarray, Path]:
    if n_samples < 1 or n_keep < 1:
        raise ValueError("n_samples and n_keep must be positive integers")

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    material_data = load_material_data(root)
    theta_deg = matlab_colon(0, 1, 85)
    rng = np.random.default_rng(seed)
    samples = _latin_hypercube(n_samples, rng)

    candidates = []
    for sample in samples:
        candidate, _, _ = _evaluate((sample[0], sample[1], int(sample[2])), theta_deg, material_data, root)
        candidates.append(candidate)
    candidates.sort(key=lambda x: x.loss)

    refined = [_local_refine(candidate, theta_deg, material_data, root) for candidate in candidates[:n_keep]]
    refined.sort(key=lambda x: x.loss)
    best = refined[0]
    _, p_profile, s_profile = _evaluate((best.lthick_2, best.hthick_2, best.npairs_2), theta_deg, material_data, root)
    output_path = _draw_profile(theta_deg, p_profile, s_profile, best, root / "optimized_9um_dual_polarization_weighted_step_profile.png")
    return best, theta_deg, p_profile, s_profile, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--keep", type=int, default=14)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing material workbooks; plots are also written here")
    args = parser.parse_args()
    best, theta, p_profile, s_profile, path = optimize_dual_polarization_step_profile(
        n_samples=args.samples, n_keep=args.keep, seed=args.seed, base_dir=args.base_dir)
    print(path)
    print(best)
    print(f"theta shape: {theta.shape}")
    print(f"TM min/max: {p_profile.min():.6g} / {p_profile.max():.6g}")
    print(f"TE min/max: {s_profile.min():.6g} / {s_profile.max():.6g}")
