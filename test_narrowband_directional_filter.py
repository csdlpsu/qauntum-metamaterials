import unittest
from pathlib import Path

import numpy as np

from narrowband_directional_filter import (
    ETA_0,
    MaterialData,
    compute_filter,
    interp1_with_end_fill,
    matlab_colon,
)


def scalar_reference(material_data, wavelength, theta_deg, low_index="CaF2", polarization="s"):
    """A small, direct MATLAB-style transcription for cross-checking."""
    h_si = 450e-9
    l_caf2 = 2550e-9
    pair_count = 5

    s11 = []
    s21 = []

    for theta in theta_deg * np.pi / 180:
        sin2 = np.sin(theta) ** 2
        for lam in wavelength:
            abcd = np.eye(2, dtype=complex)

            for material_number in (1, 2):
                if material_number == 1:
                    nh_data = material_data["TiO2"]
                else:
                    nh_data = material_data["Si"]
                nl_data = material_data[low_index]

                nh = interp1_with_end_fill(nh_data.wavelengths, nh_data.index, lam)
                nl = interp1_with_end_fill(nl_data.wavelengths, nl_data.index, lam)

                eps_h = nh.real**2 - nh.imag**2 + 1j * 2 * nh.real * nh.imag
                eps_l = nl.real**2 - nl.imag**2 + 1j * 2 * nl.real * nl.imag
                kh = 2 * np.pi / lam * np.sqrt(eps_h - sin2)
                kl = 2 * np.pi / lam * np.sqrt(eps_l - sin2)

                if polarization == "p":
                    zh = ETA_0 * np.sqrt(eps_h - sin2) / eps_h
                    zl = ETA_0 * np.sqrt(eps_l - sin2) / eps_l
                    z0 = ETA_0 * np.cos(theta)
                else:
                    zh = ETA_0 / np.sqrt(eps_h - sin2)
                    zl = ETA_0 / np.sqrt(eps_l - sin2)
                    z0 = ETA_0 / np.cos(theta)

                if material_number == 2:
                    high = np.array(
                        [
                            [np.cos(kh * h_si), 1j * zh * np.sin(kh * h_si)],
                            [1j * np.sin(kh * h_si) / zh, np.cos(kh * h_si)],
                        ],
                        dtype=complex,
                    )
                    low = np.array(
                        [
                            [np.cos(kl * l_caf2), 1j * zl * np.sin(kl * l_caf2)],
                            [1j * np.sin(kl * l_caf2) / zl, np.cos(kl * l_caf2)],
                        ],
                        dtype=complex,
                    )
                    for _ in range(pair_count):
                        abcd = abcd @ high @ low

            a, b = abcd[0]
            c, d = abcd[1]
            s11.append((a * z0 + b - c * z0 * z0 - d * z0) / (a * z0 + b + c * z0 * z0 + d * z0))
            s21.append(2 / (a + b / z0 + c * z0 + d))

    s11 = np.asarray(s11)
    s21 = np.asarray(s21)
    reflectance = np.abs(s11) ** 2
    transmission = np.abs(s21) ** 2
    absorption = 1 - reflectance - transmission
    return s11, s21, reflectance, transmission, absorption


class NarrowbandDirectionalFilterTests(unittest.TestCase):
    def test_matlab_colon_includes_expected_endpoint(self):
        values = matlab_colon(6e-6, 10e-9, 12e-6)
        self.assertEqual(len(values), 601)
        self.assertAlmostEqual(values[0], 6e-6)
        self.assertAlmostEqual(values[-1], 12e-6)

    def test_small_grid_matches_independent_scalar_reference(self):
        wavelengths = np.array([6.0e-6, 7.5e-6, 9.0e-6])
        theta_deg = np.array([0.0, 37.0])
        grid = np.array([5.0e-6, 8.0e-6, 11.0e-6])
        material_data = {
            "TiO2": MaterialData(grid, np.array([2.1 + 0.01j, 2.2 + 0.02j, 2.4 + 0.03j])),
            "Si": MaterialData(grid, np.array([3.35 + 0.02j, 3.42 + 0.03j, 3.5 + 0.04j])),
            "CaF2": MaterialData(grid, np.array([1.38 + 0.001j, 1.4 + 0.002j, 1.43 + 0.003j])),
        }

        result = compute_filter(
            wavelength=wavelengths,
            theta_deg=theta_deg,
            material_data=material_data,
            low_index="CaF2",
            polarization="s",
        )
        s11, s21, reflectance, transmission, absorption = scalar_reference(material_data, wavelengths, theta_deg)

        np.testing.assert_allclose(result.s11, s11, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.s21, s21, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.reflectance, reflectance, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.transmission, transmission, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.absorption, absorption, rtol=1e-13, atol=1e-13)

    def test_workbook_backed_default_subset_shape_and_energy_balance(self):
        root = Path(__file__).resolve().parent
        if not all((root / name).is_file() for name in ("RefractiveIndex.xlsx", "RefractiveIndex24.xlsx")):
            self.skipTest("External material workbooks are not included in the code-only repository")
        result = compute_filter(
            base_dir=root,
            wavelength=np.array([6.0e-6, 8.0e-6]),
            theta_deg=np.array([0.0, 45.0, 85.0]),
        )

        self.assertEqual(result.transmission_matrix.shape, (2, 3))
        np.testing.assert_allclose(
            result.absorption + result.reflectance + result.transmission,
            np.ones_like(result.transmission),
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertTrue(np.all(np.isfinite(result.transmission)))

    def test_tm_matches_scalar_reference(self):
        grid = np.array([5e-6, 12e-6])
        data = {name: MaterialData(grid, np.full(2, index, dtype=complex))
                for name, index in (("TiO2", 2.2), ("Si", 3.4), ("CaF2", 1.4))}
        wavelength = np.array([6e-6, 9e-6])
        angles = np.array([0., 45., 85.])
        result = compute_filter(wavelength=wavelength, theta_deg=angles,
                                material_data=data, polarization="p")
        expected = scalar_reference(data, wavelength, angles, polarization="p")
        np.testing.assert_allclose(result.s11, expected[0], atol=1e-13)
        np.testing.assert_allclose(result.s21, expected[1], atol=1e-13)
        # Independent lossless-stack conservation check (A is defined as 1-R-T).
        np.testing.assert_allclose(result.reflectance + result.transmission, 1., atol=1e-12)

    def test_invalid_solver_inputs_fail_before_loading_workbooks(self):
        for kwargs in (
            {"low_index": "typo"}, {"n_materials": 0}, {"n_materials": 6},
            {"wavelength": np.array([0.])}, {"wavelength": np.array([])},
            {"theta_deg": np.array([90.])}, {"theta_deg": np.array([np.nan])},
            {"hthicks": np.full((5, 9), -1.)},
            {"pair_numbers": np.full((5, 9), 2.5)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_filter(**kwargs)

    def test_interpolation_outside_range_uses_last_value_on_both_sides(self):
        x = np.array([1., 2.])
        y = np.array([2. + 1j, 4. + 3j])
        self.assertEqual(interp1_with_end_fill(x, y, 0.), y[-1])
        self.assertEqual(interp1_with_end_fill(x, y, 3.), y[-1])
        self.assertEqual(interp1_with_end_fill(x, y, 1.5), 3. + 2j)


if __name__ == "__main__":
    unittest.main()
