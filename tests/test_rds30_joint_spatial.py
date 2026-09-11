#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

import numpy as np

from dosimeter_rds30_joint_spatial import (
    CoupledGeometryState,
    JointSpatialConfig,
    SegmentEvidence,
    SegmentEvidenceModel,
    combine_segment_evidence,
    confidence_accepts,
    geometry_offsets,
    glyph_log_likelihood,
    temporal_geometry_penalty,
)


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


if __name__ == "__main__":
    unittest.main()
