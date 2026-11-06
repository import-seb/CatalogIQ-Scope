"""Numerical and procedural tests, distinct from statistical simulation checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from catalogiq.clustered_accuracy import (
    aggregate_groups, analytic_bounds, beta_lower, effective_sample_size,
    group_bootstrap, monte_carlo_interval, validate_group_counts, wilson_lower,
)
from catalogiq.accuracy_simulations import family_sizes, scenarios, simulate_errors
from catalogiq.accuracy_reassessment import evaluate_frozen_audit, run_method_validation, verify_seal
from catalogiq.prediction_contract import Evidence


class CalculationTests(unittest.TestCase):
    def test_wilson_matches_score_test_inversion(self):
        lower = wilson_lower(.99, 1000)
        standardized = (.99-lower)/np.sqrt(lower*(1-lower)/1000)
        self.assertAlmostEqual(standardized, norm.ppf(.95))
        self.assertAlmostEqual(wilson_lower(1., 1000), 1000/(1000+norm.ppf(.95)**2))

    def test_beta_matches_binomial_reference_and_zero_error_formula(self):
        self.assertAlmostEqual(beta_lower(.99, 1000), beta.ppf(.05, 990, 11))
        self.assertAlmostEqual(beta_lower(1, 1000), .05**(1/1000))
        self.assertEqual(beta_lower(0, 1000), 0)
        self.assertEqual(beta_lower(1, 0), 0)

    def test_unequal_sizes_not_substituted_with_number_of_groups(self):
        values = effective_sample_size([1, 1, 8])
        self.assertAlmostEqual(values["n_eff"], 100/66)
        self.assertNotEqual(values["n_eff"], values["groups"])
        self.assertLess(values["n_eff_df"], values["n_eff"])
        self.assertAlmostEqual(effective_sample_size([2]*100)["n_eff"], 100)

    def test_copying_rows_within_each_family_cannot_inflate_information(self):
        first = analytic_bounds([1, 2, 10, 3], [0, 0, 2, 0])
        repeated = analytic_bounds([10, 20, 100, 30], [0, 0, 20, 0])
        for key in ("accuracy", "n_eff", "wilson_size_adjusted", "beta_effective_sample"):
            self.assertAlmostEqual(first[key], repeated[key])

    def test_no_acceptance_and_one_family_do_not_claim_precision(self):
        empty = analytic_bounds([0, 0], [0, 0])
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["wilson_size_adjusted"])
        one = analytic_bounds([100, 0], [0, 0])
        self.assertEqual(one["wilson_size_adjusted"], 0)
        self.assertEqual(one["beta_effective_sample"], 0)
        self.assertIsNone(group_bootstrap([0, 0], [0, 0], repetitions=99)["lower"])

    def test_invalid_counts_rejected(self):
        for sizes, errors in (([1], [2]), ([1.5], [0]), ([1], [-1]), ([np.nan], [0]), ([], []), ([1, 2], [0])):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                validate_group_counts(sizes, errors)
        for alpha in (0, .5, 1):
            with self.assertRaises(ValueError):
                effective_sample_size([1], alpha)

    def test_aggregation_retains_zero_acceptance_groups_and_counts_errors(self):
        counts = aggregate_groups([1, 1, 0, 0], [1, 0, 0, 1], ["g", "g", "h", "i"])
        self.assertEqual(counts.accepted_rows.tolist(), [2, 0, 0])
        self.assertEqual(counts.errors.tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            aggregate_groups([True], [True], [""])

    def test_bootstrap_matches_known_whole_family_distribution(self):
        # Drawing two of two families has outcomes 100%, 20%, 0%, with
        # probabilities .25, .50, .25. Rows must not be sampled independently.
        result = group_bootstrap([1, 4], [0, 4], repetitions=19999, seed=1)
        self.assertEqual(result["lower"], 0)
        self.assertAlmostEqual(result["zero_error_resample_fraction"], .25, delta=.015)
        self.assertEqual(result, group_bootstrap([1, 4], [0, 4], repetitions=19999, seed=1))

    def test_bootstrap_zero_error_degeneracy_is_exposed(self):
        bootstrap = group_bootstrap([2]*1000, [0]*1000, repetitions=999)
        self.assertEqual(bootstrap["lower"], 1)
        self.assertTrue(bootstrap["zero_observed_errors"])
        analytic = analytic_bounds([2]*1000, [0]*1000)
        self.assertLess(analytic["wilson_size_adjusted"], 1)
        self.assertLess(analytic["beta_effective_sample"], 1)

    def test_error_concentration_changes_bootstrap_at_same_total_accuracy(self):
        concentrated = group_bootstrap([10]*100, [10]+[0]*99, repetitions=9999)
        spread = group_bootstrap([10]*100, [1]*10+[0]*90, repetitions=9999)
        self.assertLess(concentrated["lower"], spread["lower"])

    def test_zero_denominator_resamples_are_explicit(self):
        result = group_bootstrap([0, 5], [0, 1], repetitions=9999)
        self.assertLess(result["valid_repetitions"], result["repetitions"])
        self.assertAlmostEqual(result["lower"], .8)

    def test_monte_carlo_interval_is_for_independent_trials(self):
        lo, hi = monte_carlo_interval(50, 1000)
        self.assertLess(lo, .05)
        self.assertGreater(hi, .05)
        self.assertEqual(monte_carlo_interval(0, 1000)[0], 0)
        self.assertEqual(monte_carlo_interval(1000, 1000)[1], 1)


class SimulationAndProtectionTests(unittest.TestCase):
    def test_scenarios_span_requested_conditions_without_catalog_data(self):
        cases = scenarios()
        self.assertTrue(any(c["true_accuracy"] < .99 for c in cases))
        self.assertTrue(any(c["true_accuracy"] > .99 for c in cases))
        self.assertTrue(any(c["true_accuracy"] > .999 for c in cases))
        self.assertEqual(len({c["case"] for c in cases}), len(cases))
        for case in cases:
            sizes = family_sizes(case["size_pattern"])
            errors = simulate_errors(sizes, case, np.random.default_rng(2))
            self.assertEqual(len(sizes), 2000)
            self.assertTrue(((errors >= 0) & (errors <= sizes)).all())
            if case["error_process"] in ("whole_family", "large_only"):
                self.assertTrue(((errors == 0) | (errors == sizes)).all())
            if case["error_process"] == "large_only":
                self.assertEqual(errors[sizes < 30].sum(), 0)
                chance = case["true_error_rate"]*sizes.sum()/sizes[sizes >= 30].sum()
                self.assertAlmostEqual(chance*sizes[sizes >= 30].sum()/sizes.sum(), case["true_error_rate"])

    def test_simulation_publication_is_sealed_and_never_loads_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"validation"
            with patch("catalogiq.accuracy_simulations.scenarios", return_value=scenarios()[:1]), \
                    patch("catalogiq.accuracy_reassessment.scenarios", return_value=scenarios()[:1]), \
                    patch("catalogiq.accuracy_reassessment.load_run", side_effect=AssertionError("No catalog")), \
                    patch("catalogiq.accuracy_reassessment.frozen_identity_context", side_effect=AssertionError("No catalog")):
                run_method_validation(output, trials=100, bootstrap_repetitions=99)
            verify_seal(output, Evidence())
            decision = json.loads((output/"method_decision.json").read_text())
            self.assertFalse(decision["general_99_percent_certification_validated"])
            self.assertFalse(json.loads((output/"protocol.json").read_text())["catalog_data_read"])
            with self.assertRaises(FileExistsError):
                run_method_validation(output, trials=100, bootstrap_repetitions=99)

    def test_no_audit_read_without_sealed_simulation_decision(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("catalogiq.accuracy_reassessment.frozen_identity_context") as loader:
            base = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                evaluate_frozen_audit(base, base/"old", base/"missing-simulations", base/"new")
            loader.assert_not_called()
            self.assertFalse((base/"new").exists())
            self.assertFalse(list(base.glob("accuracy-staging-*")))

    def test_outputs_cannot_modify_existing_experiment_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaisesRegex(ValueError, "separate"):
                evaluate_frozen_audit(base, base/"old", base/"validation", base/"old/child")
            self.assertFalse((base/"old").exists())

    def test_old_evidence_guard_still_blocks_protected_test(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("catalogiq.prediction_contract.sha256", side_effect=AssertionError("No hash")):
            with self.assertRaises(PermissionError):
                Evidence().capture(Path(tmp)/"test.csv")


if __name__ == "__main__":
    unittest.main()
