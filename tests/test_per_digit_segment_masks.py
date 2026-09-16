import unittest
from dataclasses import replace

import numpy as np

import dosimeter_get_values as core


class PerDigitSegmentMaskTests(unittest.TestCase):
    def test_digit_specific_segment_polygons_are_used(self):
        profile = core.PROFILES["rds200"]

        shifted = tuple(
            {
                name: polygon
                + np.asarray((offset, 0), dtype=polygon.dtype)
                for name, polygon in profile.segment_polygons.items()
            }
            for offset in (0, 2, 4)
        )

        test_profile = replace(
            profile,
            digit_segment_polygons=shifted,
        )

        masks0 = core.make_segment_masks(test_profile, digit_index=0)
        masks1 = core.make_segment_masks(test_profile, digit_index=1)

        self.assertFalse(
            np.array_equal(masks0[0], masks1[0])
        )

        legacy = core.make_segment_masks(profile)
        default = core.make_segment_masks(test_profile)

        for a, b in zip(legacy, default):
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
