#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from dosimeter_get_values_flow import parse_wrapper_args
from dosimeter_rds30_joint_spatial import (
    CoupledGeometryState,
    JointSpatialConfig,
    SegmentEvidence,
    SegmentEvidenceModel,
    build_geometry_emissions,
    combine_segment_evidence,
    confidence_accepts,
    geometry_alignment_score_series,
    geometry_offsets,
    glyph_log_likelihood,
    select_preview_frame_indices,
    temporal_geometry_penalty,
    weak_nine_margin_rejects,
)


def make_preview_diagnostics(count: int = 20) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            frame_index=index,
            confidence=float(index),
            digit_margins=(float(index),) * 4,
            digits=(-1, 0, 5, 1),
            accepted=True,
        )
        for index in range(count)
    ]


class JointSpatialDecoderTest(unittest.TestCase):
    def test_coupled_geometry_field(self) -> None:
        state = CoupledGeometryState(tx=2, ty=-1, kx=6, ky=3)
        self.assertEqual(
            geometry_offsets(state, 4),
            ((-4, -4), (0, -2), (4, 0), (8, 2)),
        )

    def test_temporal_geometry_penalty(self) -> None:
        config = JointSpatialConfig(
            temporal_translation_weight=0.5,
            temporal_differential_weight=0.25,
        )
        first = CoupledGeometryState(0, 0, 0, 0)
        second = CoupledGeometryState(2, -1, 4, 0)
        self.assertAlmostEqual(
            temporal_geometry_penalty(first, second, config),
            2.5,
        )
        self.assertEqual(temporal_geometry_penalty(first, first, config), 0.0)

    def test_segment_evidence_combination(self) -> None:
        model = SegmentEvidenceModel(
            intercept=-1.0,
            global_weight=0.1,
            local_weight=0.1,
            oriented_edge_weight=0.05,
            cross_edge_weight=-0.05,
            continuity_weight=0.05,
            disagreement_weight=-0.1,
        )
        coherent = SegmentEvidence(30.0, 28.0, 20.0, 2.0, 18.0)
        conflicting = SegmentEvidence(5.0, 35.0, 5.0, 5.0, 3.0)
        self.assertGreater(
            combine_segment_evidence(coherent, model),
            combine_segment_evidence(conflicting, model),
        )

    def test_glyph_likelihood_is_deterministic(self) -> None:
        probabilities = np.asarray((0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.4))
        pattern = (1, 1, 1, 0, 0, 0, 0)
        expected = sum(
            math.log(probability if active else 1.0 - probability)
            for probability, active in zip(probabilities, pattern)
        )
        self.assertAlmostEqual(
            glyph_log_likelihood(probabilities, pattern),
            expected,
        )

    def test_confidence_rejection_semantics(self) -> None:
        self.assertTrue(confidence_accepts(0.40, 0.40))
        self.assertFalse(confidence_accepts(0.399, 0.40))
        self.assertFalse(confidence_accepts(float("nan"), 0.40))

    def test_weak_nine_below_margin_is_rejected(self) -> None:
        accepted = confidence_accepts(0.358, 0.358) and not weak_nine_margin_rejects(
            (0, 5, 9),
            (2.0, 1.0, 0.399),
            0.400,
        )
        self.assertFalse(accepted)

    def test_nine_at_or_above_margin_is_not_rejected(self) -> None:
        accepted_at_threshold = (
            confidence_accepts(0.358, 0.358)
            and not weak_nine_margin_rejects(
                (0, 5, 9),
                (2.0, 1.0, 0.400),
                0.400,
            )
        )
        accepted_above_threshold = (
            confidence_accepts(0.358, 0.358)
            and not weak_nine_margin_rejects((9,), (0.401,), 0.400)
        )
        self.assertTrue(accepted_at_threshold)
        self.assertTrue(accepted_above_threshold)

    def test_non_nine_below_margin_is_unaffected(self) -> None:
        accepted = confidence_accepts(0.358, 0.358) and not weak_nine_margin_rejects(
            (0, 5, 8),
            (0.01, 0.02, 0.03),
            0.400,
        )
        self.assertTrue(accepted)

    def test_omitted_nine_margin_preserves_acceptance(self) -> None:
        accepted = confidence_accepts(0.358, 0.358) and not weak_nine_margin_rejects(
            (9,),
            (-100.0,),
            None,
        )
        self.assertTrue(accepted)

        default_args, _ = parse_wrapper_args(("--track-time", "1.0"))
        self.assertIsNone(default_args.joint_spatial_min_nine_margin)

        enabled_args, _ = parse_wrapper_args(
            (
                "--track-time",
                "1.0",
                "--joint-spatial-min-nine-margin",
                "0.400",
            )
        )
        self.assertEqual(enabled_args.joint_spatial_min_nine_margin, 0.400)

    def test_glyph_independent_emission_ignores_legal_glyph_scores(self) -> None:
        glyph_scores = [
            np.asarray(((position + 1.0, position + 2.0),), dtype=np.float32)
            for position in range(4)
        ]
        changed_glyph_scores = [
            np.asarray(((1000.0 - position, -1000.0 + position),), dtype=np.float32)
            for position in range(4)
        ]
        physical_scores = [
            np.asarray(((10.0 * position, 10.0 * position + 1.0),), dtype=np.float32)
            for position in range(4)
        ]
        state_offsets = np.asarray(
            (
                (0, 0, 0, 0),
                (1, 1, 1, 1),
            ),
            dtype=np.int16,
        )
        penalties = np.asarray((0.0, 0.2), dtype=np.float32)

        original = build_geometry_emissions(
            "glyph-independent",
            glyph_scores,
            physical_scores,
            state_offsets,
            penalties,
        )
        changed = build_geometry_emissions(
            "glyph-independent",
            changed_glyph_scores,
            physical_scores,
            state_offsets,
            penalties,
        )
        np.testing.assert_array_equal(original, changed)

        changed_leading_blank = list(physical_scores)
        changed_leading_blank[0] = np.asarray(
            ((9999.0, -9999.0),),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            original,
            build_geometry_emissions(
                "glyph-independent",
                glyph_scores,
                changed_leading_blank,
                state_offsets,
                penalties,
            ),
        )

    def test_default_glyph_best_emission_matches_previous_sum(self) -> None:
        glyph_scores = [
            np.asarray(
                (
                    (position + 0.25, position + 0.5),
                    (position + 0.75, position + 1.0),
                ),
                dtype=np.float32,
            )
            for position in range(4)
        ]
        state_offsets = np.asarray(
            (
                (0, 0, 0, 0),
                (1, 1, 1, 1),
            ),
            dtype=np.int16,
        )
        penalties = np.asarray((0.0, 0.2), dtype=np.float32)
        expected = np.zeros((2, 2), dtype=np.float32)
        for position in range(4):
            expected += glyph_scores[position][:, state_offsets[:, position]]
        expected -= penalties[None, :]

        actual = build_geometry_emissions(
            "glyph-best",
            glyph_scores,
            None,
            state_offsets,
            penalties,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_geometry_alignment_requires_directional_continuous_support(self) -> None:
        aligned = np.zeros((1, 7, 5), dtype=float)
        aligned[:, :3] = (35.0, 32.0, 28.0, 2.0, 25.0)
        cross_oriented = aligned.copy()
        cross_oriented[:, :3, 2] = 12.0
        cross_oriented[:, :3, 3] = 20.0
        discontinuous = aligned.copy()
        discontinuous[:, :3, 4] = -2.0

        aligned_score = geometry_alignment_score_series(aligned)[0]
        self.assertGreater(
            aligned_score,
            geometry_alignment_score_series(cross_oriented)[0],
        )
        self.assertGreater(
            aligned_score,
            geometry_alignment_score_series(discontinuous)[0],
        )

    def test_forced_preview_frame_is_added(self) -> None:
        frames = make_preview_diagnostics()
        automatic, _ = select_preview_frame_indices(frames)
        self.assertNotIn(12, automatic)

        selected, invalid = select_preview_frame_indices(frames, (12,))
        self.assertIn(12, selected)
        self.assertTrue(automatic.issubset(selected))
        self.assertEqual(invalid, [])

    def test_automatic_preview_selection_still_works(self) -> None:
        selected, invalid = select_preview_frame_indices(
            make_preview_diagnostics()
        )
        self.assertTrue({0, 7, 10, 17, 18, 19}.issubset(selected))
        self.assertEqual(invalid, [])

    def test_preview_selection_default_is_unchanged(self) -> None:
        frames = make_preview_diagnostics()
        implicit, implicit_invalid = select_preview_frame_indices(frames)
        explicit, explicit_invalid = select_preview_frame_indices(frames, ())
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit_invalid, explicit_invalid)

        args, remaining = parse_wrapper_args(("--track-time", "1.0"))
        self.assertIsNone(args.joint_spatial_preview_frame)
        self.assertEqual(remaining, [])

    def test_out_of_range_forced_preview_frames_are_rejected(self) -> None:
        selected, invalid = select_preview_frame_indices(
            make_preview_diagnostics(),
            (-1, 12, 20),
        )
        self.assertIn(12, selected)
        self.assertNotIn(-1, selected)
        self.assertNotIn(20, selected)
        self.assertEqual(invalid, [-1, 20])

    def test_preview_frame_cli_option_is_repeatable(self) -> None:
        args, remaining = parse_wrapper_args(
            (
                "--track-time",
                "1.0",
                "--joint-spatial-preview-frame",
                "1482",
                "--joint-spatial-preview-frame",
                "1490",
            )
        )
        self.assertEqual(args.joint_spatial_preview_frame, [1482, 1490])
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
