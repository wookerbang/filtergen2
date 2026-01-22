import unittest

import numpy as np

from src.data.scenarios import sample_scenario_spec


class ScenarioSpecTests(unittest.TestCase):
    def test_bandpass_spec_fixed_recomputes_freq_range(self):
        rng = np.random.default_rng(0)
        spec = sample_scenario_spec(
            rng=rng,
            scenario="bandpass",
            spec_fixed={"fc_hz": 1e9, "bw_frac": 0.2},
        )
        self.assertEqual(spec["filter_type"], "bandpass")
        self.assertAlmostEqual(float(spec["fc_hz"]), 1e9, delta=1e-6)
        self.assertAlmostEqual(float(spec["bw_frac"]), 0.2, delta=1e-9)
        f_min, f_max = spec["freq_range"]
        self.assertAlmostEqual(float(f_min), 0.9e9, delta=1e-6)
        self.assertAlmostEqual(float(f_max), 1.1e9, delta=1e-6)

    def test_bandpass_freq_range_override_updates_fc_and_bw(self):
        rng = np.random.default_rng(0)
        spec = sample_scenario_spec(
            rng=rng,
            scenario="bandpass",
            spec_fixed={"freq_range": [0.9e9, 1.1e9]},
        )
        self.assertEqual(spec["filter_type"], "bandpass")
        self.assertAlmostEqual(float(spec["fc_hz"]), 1e9, delta=1e-6)
        self.assertAlmostEqual(float(spec["bw_frac"]), 0.2, delta=1e-9)
        f_min, f_max = spec["freq_range"]
        self.assertAlmostEqual(float(f_min), 0.9e9, delta=1e-6)
        self.assertAlmostEqual(float(f_max), 1.1e9, delta=1e-6)


if __name__ == "__main__":
    unittest.main()

