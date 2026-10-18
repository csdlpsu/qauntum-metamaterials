# qauntum metamaterials

Python tools for simulating multilayer optical directional filters and fitting
angular transmission steps at a wavelength of 9 micrometers.

Maintainer: **Ashwin Renganathan** (GitHub: `r-ashwin`), CSDL at Penn State.
This private repository contains source code, tests, and setup documentation.
Material datasets, saved results, figures, papers, slide decks, and MATLAB are
excluded. MATLAB is not needed to install or run this project.

## Install

Use Python **3.10 or newer** (Python 3.11 is used in CI), Git, and access to this
private repository. A virtual environment is recommended. On macOS/Linux:

```bash
git clone https://github.com/csdlpsu/qauntum-metamaterials.git
cd qauntum-metamaterials
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, replace the environment creation/activation commands with:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

The dependencies are NumPy (numerics), openpyxl (material workbooks), and Pillow
(plots). No Excel application, MATLAB, GPU, SciPy, or plotting GUI is required.
Version ranges in `requirements.txt` are compatibility bounds, not a frozen
environment; exact floating-point output can vary across versions/platforms.

## Required material inputs (supplied separately)

Ask the maintainer for the original **RefractiveIndex.xlsx** and
**RefractiveIndex24.xlsx** workbooks. Put both files beside
`narrowband_directional_filter.py`. Do not rename them or change their layouts.
They are deliberately excluded from Git. The repository does not download or
redistribute the datasets, and a fresh clone cannot run workbook-backed
simulations until these inputs are provided. Synthetic-data tests work without
them.

The loader expects a sheet named `Sheet1`. Wavelength cells contain micrometers;
the loader converts these to meters. Refractive indices are dimensionless and
read as `n + 1j*k`. Cells must contain numeric values or cached numeric formula
results; openpyxl does not calculate formulas. Blank cells become NaN and are
not automatically repaired. The supplied rows must be ordered by increasing
wavelength, without duplicates, and cover the range being modeled. The loader
does not enforce these data-quality conditions.

| Workbook | Material | Wavelength cells | n cells | k cells |
| --- | --- | --- | --- | --- |
| RefractiveIndex24.xlsx | TiO2 | A3:A850 | B3:B850 | C3:C850 |
| RefractiveIndex24.xlsx | Si | F3:F142 | G3:G142 | H3:H142 |
| RefractiveIndex24.xlsx | SiO2 | O3:O1912 | P3:P1912 | Q3:Q1912 |
| RefractiveIndex.xlsx | ZnS | AD3:AD314 | AE3:AE314 | AF3:AF314 |
| RefractiveIndex.xlsx | VO2 insulating | S3:S130 | T3:T130 | U3:U130 |
| RefractiveIndex.xlsx | VO2 metallic | V3:V114 | W3:W114 | X3:X114 |
| RefractiveIndex.xlsx | Ta2O5 | AL3:AL214 | AM3:AM214 | assumed zero |
| RefractiveIndex.xlsx | CaF2 | AH3:AH1514 | AI3:AI1514 | AJ3:AJ1514 |
| RefractiveIndex.xlsx | Al2O3 | AP3:AP1514 | AQ3:AQ1514 | AR3:AR1514 |
| RefractiveIndex.xlsx | Weyl | AX3:AX65 | AY3:AY65 | AZ3:AZ65 |

All ten tables are loaded by the workbook loader even when the selected stack
uses only a subset. Ta2O5 is loaded but has no high-index material selector.
For custom material data, supply a dictionary of `MaterialData` objects to
`compute_filter(material_data=...)`, avoiding Excel input entirely.

## Run the solver and plotting tools

Run commands from the repository root with the virtual environment activated.

```bash
# Default TE simulation; prints a summary, creates no output files.
python narrowband_directional_filter.py

# Default TM transmission heatmap; writes PNG and raster PDF.
python plot_fig5b_top_left_transmission.py

# Default TM angular slice at 9 micrometers; writes PNG.
python transmission_theta_slice_wrapper.py
```

The solver defaults to 601 wavelengths from 6 to 12 micrometers in 0.01
micrometer steps, and 86 incidence angles from 0 to 85 degrees in 1-degree
steps. The default stack contains five Si/CaF2 pairs, with high-index layer
thickness 450 nm and low-index layer thickness 2550 nm. `s` means TE and `p`
means TM. Solver and plot commands can take time because the computation uses
nested Python loops.

The plotting scripts write beside the scripts by default. The heatmap produces
`fig5b_top_left_TM_transmission_matrix.png` and `.pdf`; the slice produces
`transmission_theta_slice_9um_p.png`. Filenames refer to the original research
workflow; they do not establish agreement with a published figure. All generated
outputs are ignored by Git.

Programmatic usage, including a custom grid:

```python
import numpy as np
from narrowband_directional_filter import compute_filter
from transmission_theta_slice_wrapper import transmission_theta_slice

result = compute_filter(
    base_dir=".",
    wavelength=np.array([8e-6, 9e-6, 10e-6]),  # meters
    theta_deg=np.array([0., 30., 60.]),         # degrees
    polarization="p",
    low_index="CaF2",
)
print(result.transmission_matrix)  # shape: (wavelengths, angles)

# Slice parameters are in units of 100 nm, followed by the pair count.
theta, transmission = transmission_theta_slice(25.5, 4.5, 5, plot=False)
```

`FilterResult` includes complex `s11`/`s21`, flattened reflectance/transmission/
absorption, and matrices with rows = wavelengths and columns = angles. Flattened
arrays list all wavelengths for one angle before the next angle.
`compute_filter(base_dir=...)` accepts another directory for the workbooks.
For custom stacks, `hthicks`, `lthicks`, and `pair_numbers` have shape `(5, 9)`.
Thicknesses are meters; counts are nonnegative integers. Material rows are
1: TiO2, 2: Si, 3: ZnS, 4: VO2, 5: Weyl. The solver uses the first `n_materials`
rows, with `n_materials` between 1 and 5. Each slot repeats a high-then-low
bilayer the specified number of times; zero counts disable slots. The same
low-index material (`CaF2`, `SiO2`, or `Al2O3`) applies throughout the stack.
`vo2_insulating` chooses one of the two supplied VO2 tables.

## Run the optimizers

```bash
# Single-polarization searches, run separately for TM and TE.
python optimize_9um_step_profile.py --polarization p
python optimize_9um_step_profile.py --polarization s

# One geometry fitted to both polarizations.
python optimize_dual_polarization_step_profile.py
```

Each command prints the best candidate and writes a profile PNG. Single fits
write `optimized_9um_step_profile_p.png` or `_s.png`; the shared fit writes
`optimized_9um_dual_polarization_weighted_step_profile.png`. Single fits default
to 800 Latin-hypercube samples, refinement of the best 12, seed 7. Shared fits
default to 1000 samples, refinement of the best 14, seed 11. These can be long
CPU runs, with no periodic progress display, checkpoint, or resume support.
Use `--help` to inspect options. For a smaller exploratory search:

```bash
python optimize_9um_step_profile.py --polarization p --samples 10 --keep 1 --seed 7
python optimize_dual_polarization_step_profile.py --samples 10 --keep 1 --seed 11
```

Smaller sample counts still invoke local refinement and are not instantaneous.
Both optimizers accept `--base-dir PATH`; this is the directory containing the
workbooks and receiving the output plot. Run in separate directories if you need
to retain plots from multiple runs; repeated runs overwrite matching filenames.

The three design variables are low-layer thickness in `[0, 90]`, high-layer
thickness in `[2, 100]` (both in **100 nm units**), and integer pair count in
`[2, 100]`. Thus the actual thickness bounds are 0–9 micrometers and
0.2–10 micrometers. Fits vary only the first Si/CaF2 slot, at 9 micrometers and
angles 0–85 degrees. They do not optimize the other stack rows or wavelength
bandwidth.

The ideal target is zero below 45 degrees and one at/above 45 degrees. The
single-polarization loss sums low-angle leakage MSE (angles <=43 degrees),
high-angle miss MSE (>=47 degrees), and 0.25 times transition MSE (44–46
degrees). The shared loss sums the TE and TM step losses and 1.25 times a
weighted polarization-disagreement MSE. Each shared step loss equals full-grid
MSE plus 0.5 times angle-weighted MSE plus 1.5 times low-angle leakage MSE. Shared-fit
weights emphasize 46, 58, 70, and 85 degrees with Gaussian width 2 degrees.
These are research objective choices, not universal physical design criteria.

To compare two designs under both polarizations, supply your own candidate
values (LOW HIGH PAIRS). For example, this compares the default geometry with
itself; replace each triple with the values printed by your TM and TE runs:

```bash
python cross_polarization_profile_plots.py --tm-params 25.5 4.5 5 --te-params 25.5 4.5 5
```

This writes four individual profile PNGs and
`profile_9um_cross_polarization_2x2.png`. It accepts `--base-dir PATH`.
No saved optimized candidates or precomputed results are included in this repo.

## Assumptions and limits

### Physical model

- The transfer-matrix model assumes coherent monochromatic plane waves in a
  one-dimensional stack of flat, laterally infinite, homogeneous, isotropic,
  linear, nonmagnetic layers. Each material is represented by scalar relative
  permittivity `epsilon_r = (n + i*k)^2`.
- Incidence and exit media are both air/vacuum with identical reference
  impedances (`eta_0 = 376.7 ohms`). Consequently `R = |S11|^2`, `T = |S21|^2`,
  and `A = 1 - R - T`. There is no explicit substrate, arbitrary ambient/exit
  refractive index, finite beam, surface roughness, thickness disorder,
  fabrication tolerance, or incoherent averaging.
- TE and TM are independent calculations. There is no polarization conversion,
  anisotropic/tensor response, gyrotropy, lateral patterning, diffraction,
  nonlocal response, nonlinear response, or quantum mechanical calculation.
  In particular, selecting the scalar Weyl table does not implement a full
  Weyl-semimetal tensor model. The project title does not imply those features.
- Material dispersion and loss come only from the input tables. VO2 switches
  between two static tables; no temperature dynamics, hysteresis, or intermediate
  phase model is present. No explicit temperature dependence is implemented.

### Data and numerical behavior

- Real and imaginary index parts are interpolated linearly. **Outside either
  end of a material's wavelength range, the final tabulated value is used**,
  including queries below the first wavelength. This preserves the original
  implementation but is not a physically justified extrapolation. Verify all
  material ranges before interpreting a sweep.
- The original `n + i*k` convention, positive-imaginary off-diagonal layer
  matrices, and principal complex square roots are retained. Lossy-media sign
  and branch conventions need physical validation for the intended dataset;
  the code does not enforce passivity or automatically correct `R`, `T`, or `A`.
  Do not assume that an unphysical result is harmless numerical noise.
- Direct products of transfer matrices can become ill-conditioned or overflow
  for thick, strongly absorbing, evanescent, or large-pair-count stacks. There
  is no stabilized scattering-matrix algorithm, adaptive precision, or
  convergence/error estimator. Angles at grazing incidence (±90 degrees) are
  rejected; approaching grazing incidence can still be poorly conditioned.
- Plotting and optimization losses **clip transmission to [0, 1]**. This can
  conceal numerical or physical inconsistencies. Inspect the raw solver output
  and check finiteness, passivity, data coverage, and grid sensitivity before
  trusting a plotted profile or optimized candidate. Heatmap PDF output is a
  raster image embedded in a PDF, not vector artwork.
- Default grids and the 1-degree optimization grid can miss narrow spectral or
  angular resonances. The plotted ideal step is discontinuous; a finite stack
  need not realize it exactly. Reevaluate candidates on finer grids and over
  the wavelength/tolerance ranges relevant to the application.
- Latin-hypercube sampling followed by coordinate refinement is a heuristic.
  A fixed seed makes sampling repeatable within an environment but supplies no
  global-optimality guarantee. Losses differ between single and shared fits,
  so their scalar values are not directly comparable. A good fitted angular
  profile does not demonstrate broadband or fabrication robustness.

### Verification scope

Tests compare small-grid TE/TM calculations with a separate scalar transcription,
exercise a lossless conservation case, interpolation, and invalid inputs. An
optional workbook test checks finite output and matrix dimensions. Because
absorption is defined as `1-R-T`, checking `A+R+T=1` alone is an algebraic identity,
not independent physical validation. These tests do not establish agreement with
experiments, an independent electromagnetic solver, or every published result.

## Tests and collaboration

```bash
python -m unittest discover -v
```

Without the external workbooks, the workbook-dependent test is reported as
skipped; synthetic tests still run. GitHub Actions runs this same suite on Linux
with Python 3.11 and does not receive material files or research results.

The `.gitignore` permits source and repository support files while excluding
datasets, outputs, environments, caches, and editor settings. Before committing,
inspect `git status` and `git diff --cached --name-only`. Do not force-add input
data or results. The legacy presentation-editing helper is excluded because it
depends on an existing results deck and contains fixed result summaries.

Repository access is managed through the private `csdlpsu` organization repo.
No public redistribution license has been assigned in this handoff; coordinate
reuse and sharing with the maintainer.

## Troubleshooting

- **Repository not found on clone:** sign in with the GitHub account granted
  access to `csdlpsu/qauntum-metamaterials`, and complete any required org SSO.
- **Missing NumPy/PIL/openpyxl:** activate the virtual environment, then install
  dependencies with that environment's `python -m pip`.
- **Missing workbook / Sheet1 / invalid cell:** obtain the original inputs,
  check their exact names and worksheet layout, and supply numeric/cached values.
- **Unexpectedly flat or unphysical curves:** inspect the raw arrays, confirm
  meter versus 100 nm units, material ranges, sign convention, and conditioning.
- **Plots look different between computers:** font availability affects rendering;
  the shared helper tries Times, then DejaVu Serif, then Pillow's scalable default.
- **Optimization takes a long time:** reduce samples/retained candidates for
  exploratory runs, remembering that local refinement still has a variable cost.
