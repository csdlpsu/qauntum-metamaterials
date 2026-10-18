"""Fit a 9 micrometer angular step for a single polarization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plotting_utils import plot_font as _font

from narrowband_directional_filter import (
    _default_thicknesses,
    compute_filter,
    load_material_data,
    matlab_colon,
)


LOW_ANGLE_MAX = 43.0
HIGH_ANGLE_MIN = 47.0
STEP_ANGLE = 45.0
THICKNESS_UNIT_M = 100e-9
BOUNDS = np.asarray([[0.0, 90.0], [2.0, 100.0], [2.0, 100.0]], dtype=float)


@dataclass(frozen=True)
class Candidate:
    lthick_2_0: float
    hthick_2_0: float
    n_pairs_2_0: int
    loss: float
    low_loss: float
    high_loss: float
    transition_loss: float
    mean_low: float
    mean_high: float
    contrast: float


def _target(theta_deg: np.ndarray) -> np.ndarray:
    return (theta_deg >= STEP_ANGLE).astype(float)


def _loss(theta_deg: np.ndarray, transmission: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    clipped = np.clip(transmission.real, 0.0, 1.0)
    low_mask = theta_deg <= LOW_ANGLE_MAX
    high_mask = theta_deg >= HIGH_ANGLE_MIN
    transition_mask = (theta_deg > LOW_ANGLE_MAX) & (theta_deg < HIGH_ANGLE_MIN)
    target = _target(theta_deg)

    low_loss = float(np.mean(clipped[low_mask] ** 2))
    high_loss = float(np.mean((1.0 - clipped[high_mask]) ** 2))
    transition_loss = float(np.mean((clipped[transition_mask] - target[transition_mask]) ** 2))
    mean_low = float(np.mean(clipped[low_mask]))
    mean_high = float(np.mean(clipped[high_mask]))
    contrast = mean_high - mean_low

    # The first two terms define the desired step; the transition term is softened
    # because a finite multilayer stack cannot realize an abrupt discontinuity.
    total = low_loss + high_loss + 0.25 * transition_loss
    return total, low_loss, high_loss, transition_loss, mean_low, mean_high, contrast


def _profile(
    params: tuple[float, float, int],
    theta_deg: np.ndarray,
    material_data: dict,
    base_dir: Path,
    polarization: str,
) -> np.ndarray:
    lthick_2_0, hthick_2_0, n_pairs_2_0 = params
    hthicks, lthicks, pair_numbers = _default_thicknesses()
    lthicks[1, 0] = lthick_2_0 * THICKNESS_UNIT_M
    hthicks[1, 0] = hthick_2_0 * THICKNESS_UNIT_M
    pair_numbers[1, 0] = int(n_pairs_2_0)

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
    polarization: str,
) -> Candidate:
    transmission = _profile(params, theta_deg, material_data, base_dir, polarization)
    total, low_loss, high_loss, transition_loss, mean_low, mean_high, contrast = _loss(theta_deg, transmission)
    return Candidate(
        lthick_2_0=float(params[0]),
        hthick_2_0=float(params[1]),
        n_pairs_2_0=int(params[2]),
        loss=total,
        low_loss=low_loss,
        high_loss=high_loss,
        transition_loss=transition_loss,
        mean_low=mean_low,
        mean_high=mean_high,
        contrast=contrast,
    )


def _latin_hypercube(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.empty((n_samples, 3), dtype=float)
    for dim, (lower, upper) in enumerate(BOUNDS):
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        samples[:, dim] = lower + cut * (upper - lower)
    samples[:, 2] = np.rint(samples[:, 2])
    samples[:, 2] = np.clip(samples[:, 2], BOUNDS[2, 0], BOUNDS[2, 1])
    return samples


def _local_refine(
    seed: Candidate,
    theta_deg: np.ndarray,
    material_data: dict,
    base_dir: Path,
    polarization: str,
) -> Candidate:
    best = seed
    l_step = 6.0
    h_step = 6.0
    n_step = 8

    while l_step >= 0.25 or h_step >= 0.25 or n_step >= 1:
        improved = False
        l_values = [best.lthick_2_0 - l_step, best.lthick_2_0, best.lthick_2_0 + l_step]
        h_values = [best.hthick_2_0 - h_step, best.hthick_2_0, best.hthick_2_0 + h_step]
        n_values = [best.n_pairs_2_0 - n_step, best.n_pairs_2_0, best.n_pairs_2_0 + n_step]

        for l_value in l_values:
            for h_value in h_values:
                for n_value in n_values:
                    l_value = float(np.clip(l_value, BOUNDS[0, 0], BOUNDS[0, 1]))
                    h_value = float(np.clip(h_value, BOUNDS[1, 0], BOUNDS[1, 1]))
                    n_value = int(np.clip(round(n_value), BOUNDS[2, 0], BOUNDS[2, 1]))
                    candidate = _evaluate((l_value, h_value, n_value), theta_deg, material_data, base_dir, polarization)
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


def _draw_best_profile(
    theta_deg: np.ndarray,
    transmission: np.ndarray,
    candidate: Candidate,
    output_path: Path,
    polarization: str,
) -> Path:
    width, height = 1300, 900
    left, top, right, bottom = 150, 90, 1180, 705
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    axis_font = _font(42, bold=True)
    tick_font = _font(30)
    note_font = _font(28)

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

    target = _target(theta_deg)
    target_points = [
        (left + (right - left) * theta / 85, bottom - (bottom - top) * value)
        for theta, value in zip(theta_deg, target)
    ]
    profile = np.clip(transmission, 0.0, 1.0)
    profile_points = [
        (left + (right - left) * theta / 85, bottom - (bottom - top) * value)
        for theta, value in zip(theta_deg, profile)
    ]
    draw.line(target_points, fill=(210, 60, 60), width=4)
    draw.line(profile_points, fill=(30, 90, 220), width=5, joint="curve")

    draw.line((left + (right - left) * STEP_ANGLE / 85, top, left + (right - left) * STEP_ANGLE / 85, bottom), fill=(80, 80, 80), width=2)

    pol_label = "TM" if polarization == "p" else "TE"
    title = f"Best 9 \N{MICRO SIGN}m {pol_label} Transmission Step Fit"
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

    summary = (
        f"loss={candidate.loss:.4g} | lthick_2={candidate.lthick_2_0:.3g}, "
        f"hthick_2={candidate.hthick_2_0:.3g}, npairs_2={candidate.n_pairs_2_0}"
    )
    legend = "blue: optimized    red: target"
    draw.rectangle((left + 14, top + 14, left + 650, top + 82), fill="white", outline=(180, 180, 180), width=1)
    draw.text((left + 24, top + 20), legend, font=note_font, fill="black")
    draw.text((left + 24, top + 50), summary, font=note_font, fill="black")
    canvas.save(output_path)
    return output_path


def optimize_step_profile(
    *,
    n_samples: int = 800,
    n_keep: int = 12,
    seed: int = 7,
    polarization: str = "p",
    base_dir: str | Path | None = None,
) -> tuple[Candidate, np.ndarray, np.ndarray, Path]:
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'")

    if n_samples < 1 or n_keep < 1:
        raise ValueError("n_samples and n_keep must be positive integers")

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    material_data = load_material_data(root)
    theta_deg = matlab_colon(0, 1, 85)
    rng = np.random.default_rng(seed)

    samples = _latin_hypercube(n_samples, rng)
    candidates = [
        _evaluate((sample[0], sample[1], int(sample[2])), theta_deg, material_data, root, polarization)
        for sample in samples
    ]
    candidates.sort(key=lambda candidate: candidate.loss)

    refined = [_local_refine(candidate, theta_deg, material_data, root, polarization) for candidate in candidates[:n_keep]]
    refined.sort(key=lambda candidate: candidate.loss)
    best = refined[0]
    transmission = _profile(
        (best.lthick_2_0, best.hthick_2_0, best.n_pairs_2_0),
        theta_deg,
        material_data,
        root,
        polarization,
    )
    output_path = _draw_best_profile(
        theta_deg,
        transmission,
        best,
        root / f"optimized_9um_step_profile_{polarization}.png",
        polarization,
    )
    return best, theta_deg, transmission, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing material workbooks; plots are also written here")
    parser.add_argument("--polarization", choices=("s", "p"), default="p")
    args = parser.parse_args()
    best, theta, transmission, path = optimize_step_profile(
        n_samples=args.samples, n_keep=args.keep, seed=args.seed,
        polarization=args.polarization, base_dir=args.base_dir)
    print(path)
    print(best)
    print(f"theta shape: {theta.shape}")
    print(f"transmission min/max: {transmission.min():.6g} / {transmission.max():.6g}")
