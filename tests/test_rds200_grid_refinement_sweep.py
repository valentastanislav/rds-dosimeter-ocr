#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_fixedgrid as fixed
import dosimeter_get_values_flow as flow
from tests.rds200_grid_refinement_sweep import (
    CandidateMetrics,
    GridCandidate,
    generate_candidates,
    make_frozen_rds200_profile,
    offset_grid_by_pixels,
    rank_candidates,
)


def metrics(
    candidate: GridCandidate,
    *,
    mean_confidence: float = 0.5,
    median_confidence: float = 0.5,
    recognized_samples: int = 10,
    median_margin: float = 7.0,
    recognized_digits: int = 30,
    p10: float = 4.0,
) -> CandidateMetrics:
    return CandidateMetrics(
        candidate=candidate,
        total_samples=10,
        display_samples=10,
        recognized_samples=recognized_samples,
        recognized_fraction=recognized_samples / 10,
        recognized_digits=recognized_digits,
        total_digits=30,
        digit_recognition_fraction=recognized_digits / 30,
        ambiguous_digits=30 - recognized_digits,
        ambiguous_digit_fraction=(30 - recognized_digits) / 30,
        p10_min_margin=p10,
        median_min_margin=median_margin,
        mean_min_margin=median_margin,
        mean_digit_confidence=mean_confidence,
        median_digit_confidence=median_confidence,
        recognized_digit1=10,
        recognized_digit2=10,
        recognized_digit3=recognized_digits - 20,
        p10_margin_digit1=p10,
        p10_margin_digit2=p10,
        p10_margin_digit3=p10,
        median_confidence_digit1=median_confidence,
        median_confidence_digit2=median_confidence,
        median_confidence_digit3=median_confidence,
        evaluation_seconds=0.0,
    )


class Rds200GridRefinementSweepTest(unittest.TestCase):
    def test_pixel_offsets_convert_to_normalized_grid(self) -> None:
        actual = offset_grid_by_pixels(
            (0.2, 0.3, 0.8, 0.9),
            100,
            200,
            1,
            -2,
            -3,
            4,
        )
        self.assertEqual(actual, (0.21, 0.29, 0.77, 0.92))

    def test_radius_one_generates_every_four_edge_candidate(self) -> None:
        candidates = generate_candidates((0.2, 0.3, 0.8, 0.9), 100, 100, 1)
        self.assertEqual(len(candidates), 3**4)
        self.assertEqual(
            len(
                {
                    (item.dx1, item.dy1, item.dx2, item.dy2)
                    for item in candidates
                }
            ),
            3**4,
        )

    def test_exact_base_grid_is_always_included(self) -> None:
        base = (0.2, 0.3, 0.8, 0.9)
        candidates = generate_candidates(base, 100, 100, 2)
        exact = [
            item
            for item in candidates
            if (item.dx1, item.dy1, item.dx2, item.dy2) == (0, 0, 0, 0)
        ]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].grid, base)

    def test_invalid_grids_are_rejected(self) -> None:
        self.assertIsNone(
            offset_grid_by_pixels(
                (0.0, 0.2, 0.01, 0.8),
                100,
                100,
                2,
                0,
                -2,
                0,
            )
        )
        candidates = generate_candidates((0.0, 0.2, 0.01, 0.8), 100, 100, 2)
        self.assertLess(len(candidates), 5**4)
        for item in candidates:
            x1, y1, x2, y2 = item.grid
            self.assertTrue(0.0 <= x1 < x2 <= 1.0)
            self.assertTrue(0.0 <= y1 < y2 <= 1.0)

    def test_ranking_is_confidence_first_and_deterministic(self) -> None:
        base_grid = (0.2, 0.3, 0.8, 0.9)
        mean_winner = GridCandidate(2, 2, 2, 2, base_grid)
        median_winner = GridCandidate(2, 2, 2, 1, base_grid)
        recognized_winner = GridCandidate(2, 2, 1, 1, base_grid)
        margin_winner = GridCandidate(2, 1, 1, 1, base_grid)
        perturbation_winner = GridCandidate(0, 0, 0, 0, base_grid)
        lower_recognized = GridCandidate(1, 1, 1, 1, base_grid)

        values = [
            metrics(
                mean_winner,
                mean_confidence=0.91,
                median_confidence=0.80,
                recognized_samples=8,
                median_margin=1.0,
            ),
            metrics(
                median_winner,
                mean_confidence=0.90,
                median_confidence=0.96,
                recognized_samples=8,
                median_margin=1.0,
            ),
            metrics(
                recognized_winner,
                mean_confidence=0.90,
                median_confidence=0.95,
                recognized_samples=9,
                median_margin=1.0,
            ),
            metrics(
                margin_winner,
                mean_confidence=0.90,
                median_confidence=0.95,
                recognized_samples=9,
                median_margin=2.0,
            ),
            metrics(
                perturbation_winner,
                mean_confidence=0.90,
                median_confidence=0.95,
                recognized_samples=9,
                median_margin=2.0,
            ),
            metrics(
                lower_recognized,
                mean_confidence=0.90,
                median_confidence=0.95,
                recognized_samples=8,
                median_margin=100.0,
            ),
        ]

        first = rank_candidates(values)
        second = rank_candidates(list(reversed(values)))
        expected = [
            mean_winner,
            median_winner,
            perturbation_winner,
            margin_winner,
            recognized_winner,
            lower_recognized,
        ]

        self.assertEqual([item.candidate for item in first], expected)
        self.assertEqual([item.candidate for item in second], expected)

    def test_profile_and_original_polygons_are_not_mutated(self) -> None:
        original = core.PROFILES["rds200"]
        boxes_before = original.digit_boxes
        polygons_before = {
            name: polygon.copy()
            for name, polygon in original.segment_polygons.items()
        }
        digit_polygons_before = (
            None
            if original.digit_segment_polygons is None
            else tuple(
                {
                    name: polygon.copy()
                    for name, polygon in digit.items()
                }
                for digit in original.digit_segment_polygons
            )
        )

        frozen = make_frozen_rds200_profile(original)
        candidate = fixed.make_fixed_profile(
            frozen,
            (0.239351, 0.455056, 0.679513, 0.792135),
        )

        self.assertIsNot(frozen, original)
        self.assertIsNot(candidate, frozen)
        self.assertEqual(original.digit_boxes, boxes_before)
        if digit_polygons_before is None:
            self.assertIsNone(original.digit_segment_polygons)
        else:
            self.assertIsNotNone(original.digit_segment_polygons)
            for current_digit, before_digit in zip(
                original.digit_segment_polygons,
                digit_polygons_before,
            ):
                for name, polygon in before_digit.items():
                    np.testing.assert_array_equal(
                        current_digit[name],
                        polygon,
                    )
        for name, polygon in polygons_before.items():
            np.testing.assert_array_equal(original.segment_polygons[name], polygon)

    def test_production_cache_observer_is_opt_in(self) -> None:
        parameter = inspect.signature(flow.main).parameters[
            "display_cache_observer"
        ]
        self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
