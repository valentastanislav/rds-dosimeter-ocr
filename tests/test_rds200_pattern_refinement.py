#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_flow as flow


class Rds200PatternRefinementTest(unittest.TestCase):
    patterns = core.STANDARD_DIGIT_PATTERNS

    def test_matching_binary_seven_only_raises_confidence(self) -> None:
        levels = np.asarray((26, 47, 49, 3, 2, 3, 0), dtype=float)

        baseline = core.decode_pattern_digit(levels, self.patterns)
        refined = core.decode_pattern_digit(
            levels,
            self.patterns,
            rds200_pattern_refinement=True,
        )

        self.assertEqual(baseline[0], 7)
        self.assertAlmostEqual(baseline[1], 0.19920052919285902)
        self.assertEqual(refined[0], 7)
        self.assertAlmostEqual(refined[1], 0.74)

    def test_uniform_all_active_glare_can_override_to_eight(self) -> None:
        levels = np.asarray(
            (86.4, 104, 103, 97, 87, 108.7, 103),
            dtype=float,
        )

        baseline = core.decode_pattern_digit(levels, self.patterns)
        refined = core.decode_pattern_digit(
            levels,
            self.patterns,
            rds200_pattern_refinement=True,
        )

        self.assertEqual(baseline[0], 4)
        self.assertEqual(refined[0], 8)

    def test_nonuniform_all_active_levels_do_not_override_to_eight(self) -> None:
        levels = np.asarray((16, 101, 101, 98, 16, 100, 90), dtype=float)
        self.assertLess(float(np.min(levels) / np.max(levels)), 0.70)

        baseline = core.decode_pattern_digit(levels, self.patterns)
        refined = core.decode_pattern_digit(
            levels,
            self.patterns,
            rds200_pattern_refinement=True,
        )

        self.assertNotEqual(refined[0], 8)
        self.assertEqual(refined, baseline)

    def test_existing_non_eight_binary_override_is_unchanged(self) -> None:
        levels = np.asarray((8, 50, 50, 0, 0, 0, 0), dtype=float)

        baseline = core.decode_pattern_digit(levels, self.patterns)
        refined = core.decode_pattern_digit(
            levels,
            self.patterns,
            rds200_pattern_refinement=True,
        )

        self.assertEqual(baseline[0], 7)
        self.assertEqual(refined, baseline)

    def test_disabled_refinement_is_exactly_the_existing_behavior(self) -> None:
        vectors = (
            (26, 47, 49, 3, 2, 3, 0),
            (86.4, 104, 103, 97, 87, 108.7, 103),
            (16, 101, 101, 98, 16, 100, 90),
        )
        for vector in vectors:
            levels = np.asarray(vector, dtype=float)
            self.assertEqual(
                core.decode_pattern_digit(levels, self.patterns),
                core.decode_pattern_digit(
                    levels,
                    self.patterns,
                    rds200_pattern_refinement=False,
                ),
            )

    def test_cli_pattern_refinement_override_is_tristate(self) -> None:
        default, remaining = flow.parse_wrapper_args(("--track-time", "1.0"))
        self.assertIsNone(default.rds200_pattern_refinement)
        self.assertEqual(remaining, [])

        enabled, remaining = flow.parse_wrapper_args(
            ("--rds200-pattern-refinement", "--track-time", "1.0")
        )
        self.assertTrue(enabled.rds200_pattern_refinement)
        self.assertEqual(remaining, [])

        disabled, remaining = flow.parse_wrapper_args(
            ("--no-rds200-pattern-refinement", "--track-time", "1.0")
        )
        self.assertFalse(disabled.rds200_pattern_refinement)
        self.assertEqual(remaining, [])

if __name__ == "__main__":
    unittest.main()
