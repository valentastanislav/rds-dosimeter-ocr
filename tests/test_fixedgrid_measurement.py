#!/usr/bin/env python3

import unittest

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_fixedgrid as fixed


class LocalMeasurementTest(unittest.TestCase):
    def test_local_p70_minus_segment_p10(self) -> None:
        profile = core.PROFILES["rds30"]
        segment_masks, ring_masks = fixed.make_local_masks(profile)
        segment_mask = segment_masks[0]
        ring_mask = ring_masks[0]

        patch = np.full((130, 65), 180, dtype=np.uint8)
        patch[segment_mask] = (
            np.arange(np.count_nonzero(segment_mask), dtype=np.uint16) % 91
        ).astype(np.uint8)
        patch[ring_mask] = (
            100
            + np.arange(np.count_nonzero(ring_mask), dtype=np.uint16) % 101
        ).astype(np.uint8)

        config = core.FixedGridMeasurementConfig(
            mode="local",
            segment_percentile=10.0,
            background_percentile=70.0,
        )
        measured = fixed.local_darkness([patch], profile, config)[0, 0]
        expected = (
            np.percentile(patch[ring_mask], 70)
            - np.percentile(patch[segment_mask], 10)
        )

        self.assertEqual(measured, expected)


if __name__ == "__main__":
    unittest.main()
