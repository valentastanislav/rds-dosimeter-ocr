#!/usr/bin/env python3

import unittest

import dosimeter_get_values as core
import dosimeter_get_values_flow as flow


class Rds30ProfileDefaultsTest(unittest.TestCase):
    @staticmethod
    def parsed(*extra):
        args, remaining = flow.parse_wrapper_args(
            ["--track-time", "1.0", *extra]
        )
        assert remaining == []
        return args

    def test_rds30_gets_validated_defaults(self):
        args = self.parsed()
        flow.apply_profile_decoder_defaults(args, core.PROFILES["rds30"])

        self.assertEqual(args.decoder_strategy, "rds30-joint-spatial")
        self.assertEqual(
            args.joint_spatial_geometry_emission,
            "glyph-independent",
        )
        self.assertEqual(args.joint_spatial_confidence, 0.358)
        self.assertEqual(args.joint_spatial_min_nine_margin, 0.400)

    def test_rds200_keeps_historical_defaults(self):
        args = self.parsed()
        flow.apply_profile_decoder_defaults(args, core.PROFILES["rds200"])

        self.assertEqual(args.decoder_strategy, "default")
        self.assertEqual(args.joint_spatial_geometry_emission, "glyph-best")
        self.assertIsNone(args.joint_spatial_confidence)
        self.assertIsNone(args.joint_spatial_min_nine_margin)

    def test_explicit_legacy_rds30_decoder_is_preserved(self):
        args = self.parsed("--decoder-strategy", "default")
        flow.apply_profile_decoder_defaults(args, core.PROFILES["rds30"])

        self.assertEqual(args.decoder_strategy, "default")
        self.assertIsNone(args.joint_spatial_confidence)
        self.assertIsNone(args.joint_spatial_min_nine_margin)

    def test_explicit_rds30_overrides_are_preserved(self):
        args = self.parsed(
            "--decoder-strategy", "rds30-joint-spatial",
            "--joint-spatial-geometry-emission", "glyph-best",
            "--joint-spatial-confidence", "0.51",
            "--joint-spatial-min-nine-margin", "0.72",
        )
        flow.apply_profile_decoder_defaults(args, core.PROFILES["rds30"])

        self.assertEqual(args.decoder_strategy, "rds30-joint-spatial")
        self.assertEqual(args.joint_spatial_geometry_emission, "glyph-best")
        self.assertEqual(args.joint_spatial_confidence, 0.51)
        self.assertEqual(args.joint_spatial_min_nine_margin, 0.72)


if __name__ == "__main__":
    unittest.main()
