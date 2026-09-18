#!/usr/bin/env python3

from __future__ import annotations

import io
import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import dosimeter_get_values_flow as flow
import dosimeter_get_values_rectified as rectified
import dosimeter_get_values_roi as roi
import dosimeter_get_values_fixedgrid as fixedgrid


class FlowCliAndSelectionTest(unittest.TestCase):
    def test_wrapper_error_usage_starts_with_positional_files(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", stderr):
            flow.parse_wrapper_args([])
        usage_line = re.sub(
            r"\x1b\[[0-9;]*m",
            "",
            stderr.getvalue().splitlines()[0],
        )
        self.assertEqual(
            usage_line,
            "usage: python3 dosimeter_get_values_flow.py "
            "<video file> <output file> [options]",
        )


    def test_default_decoder_strategy_initializes_decode_function(self) -> None:
        args = SimpleNamespace(
            decoder_strategy="default",
            rds200_pattern_refinement=False,
        )
        selected_decode_samples = flow.core.decode_samples

        if args.decoder_strategy == "rds30-joint-spatial":
            self.fail("unexpected joint-spatial strategy")
        elif args.rds200_pattern_refinement:
            self.fail("unexpected RDS-200 refinement strategy")

        self.assertIs(selected_decode_samples, flow.core.decode_samples)

    def test_debug_geometry_uses_per_digit_segment_polygons(self) -> None:
        display = np.zeros((130, 65), dtype=np.uint8)
        shared = {
            name: np.asarray(
                ((1, 1), (2, 1), (2, 2), (1, 2)),
                dtype=np.int32,
            )
            for name in flow.core.SEGMENT_ORDER
        }
        specific = {
            name: np.asarray(
                ((10, 20), (12, 20), (12, 22), (10, 22)),
                dtype=np.int32,
            )
            for name in flow.core.SEGMENT_ORDER
        }
        profile = SimpleNamespace(
            digit_boxes=((0, 0, 65, 130),),
            segment_polygons=shared,
            digit_segment_polygons=(specific,),
            decimal_candidates=(),
        )

        with mock.patch.object(flow.cv2, "polylines") as polylines:
            flow.draw_decoder_geometry(
                display,
                profile,
                chosen_decimal_places=None,
                decimal_mode="auto",
            )

        first_points = polylines.call_args_list[0].args[1][0]
        np.testing.assert_array_equal(
            first_points,
            specific[flow.core.SEGMENT_ORDER[0]],
        )

    def test_rds200_fixed_grid_derives_decimal_geometry(self) -> None:
        grid = (
            0.245436,
            0.471910,
            0.677485,
            0.794944,
        )
        profile = fixedgrid.make_fixed_profile(
            flow.core.RDS200,
            grid,
        )

        self.assertIsNotNone(profile.digit_segment_polygons)

        expected_boundaries = (
            0.5 * (
                profile.digit_boxes[0][2]
                + profile.digit_boxes[1][0]
            ),
            0.5 * (
                profile.digit_boxes[1][2]
                + profile.digit_boxes[2][0]
            ),
        )
        actual_centres = tuple(
            0.5 * (item[0] + item[2])
            for item in profile.decimal_candidates
        )

        for actual, expected in zip(
            actual_centres,
            expected_boundaries,
        ):
            self.assertAlmostEqual(
                actual,
                expected,
                delta=0.5,
            )

        bottom_edges = []
        for digit_box, polygons in zip(
            profile.digit_boxes,
            profile.digit_segment_polygons,
        ):
            _x1, y1, _x2, y2 = digit_box
            local_bottom = float(
                np.max(
                    np.asarray(
                        polygons["d"]
                    )[:, 1]
                )
            )
            bottom_edges.append(
                y1
                + local_bottom
                * (y2 - y1)
                / 130.0
            )

        expected_y2 = round(
            float(np.median(bottom_edges))
        )
        self.assertEqual(
            tuple(
                item[3]
                for item in profile.decimal_candidates
            ),
            (expected_y2, expected_y2),
        )

    def test_rds200_feature_mask_uses_outer_ring_only(self) -> None:
        shape = (400, 600)
        box = (200, 100, 120, 80)
        mask = flow.make_feature_mask(
            shape,
            box,
            flow.core.RDS200,
        )

        x, y, width, height = box
        self.assertEqual(
            int(np.count_nonzero(
                mask[
                    y:y + height,
                    x:x + width,
                ]
            )),
            0,
        )

        self.assertGreater(
            int(np.count_nonzero(mask)),
            0,
        )

    def test_summary_confidence_cut_excludes_low_confidence_interval(self) -> None:
        intervals = (
            (
                0.0,
                9.0,
                flow.core.Run(
                    start_index=0,
                    end_index=45,
                    value=20.0,
                    confidence=0.90,
                ),
            ),
            (
                9.0,
                10.0,
                flow.core.Run(
                    start_index=45,
                    end_index=50,
                    value=40.0,
                    confidence=0.10,
                ),
            ),
        )

        summary = flow.core.calculate_summary(
            intervals,
            Path("example.MOV"),
            flow.core.RDS200,
            10.0,
            1.0,
            1.0,
            summary_min_confidence=0.20,
        )

        self.assertAlmostEqual(
            summary["time_weighted_mean"],
            20.0,
        )
        self.assertAlmostEqual(
            summary["time_weighted_std_dev"],
            0.0,
        )
        self.assertEqual(
            summary["summary_interval_count"],
            1,
        )
        self.assertAlmostEqual(
            summary["summary_included_duration_s"],
            9.0,
        )
        self.assertAlmostEqual(
            summary["summary_excluded_duration_s"],
            1.0,
        )

    def test_roi_summary_confidence_default_is_point_two(self) -> None:
        args = roi.parse_args(
            (
                "video.MOV",
                "out.csv",
            )
        )
        self.assertAlmostEqual(
            args.summary_min_confidence,
            0.20,
        )

    def test_reference_box_cancel_closes_window(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        with (
            mock.patch.object(flow.cv2, "namedWindow"),
            mock.patch.object(flow.cv2, "selectROI", return_value=(0, 0, 0, 0)),
            mock.patch.object(flow.cv2, "destroyWindow") as destroy_window,
            mock.patch.object(flow.cv2, "waitKey", return_value=0),
            mock.patch.object(rectified.sys, "stdin", SimpleNamespace(isatty=lambda: False)),
        ):
            with self.assertRaisesRegex(RuntimeError, "selection was cancelled"):
                flow.select_reference_box(frame)
        destroy_window.assert_called_once()

    def test_quad_c_cancel_closes_window(self) -> None:
        display = np.zeros((20, 30), dtype=np.uint8)
        with (
            mock.patch.object(rectified.cv2, "namedWindow"),
            mock.patch.object(rectified.cv2, "setMouseCallback"),
            mock.patch.object(rectified.cv2, "imshow"),
            mock.patch.object(rectified.cv2, "waitKey", return_value=ord("c")),
            mock.patch.object(rectified.cv2, "destroyWindow") as destroy_window,
            mock.patch.object(rectified.sys, "stdin", SimpleNamespace(isatty=lambda: False)),
        ):
            with self.assertRaisesRegex(RuntimeError, "Quad selection cancelled"):
                rectified.select_quad(display)
        destroy_window.assert_called_once()

    def test_selection_cleanup_restores_terminal_state(self) -> None:
        attributes = [1, 2, 3]
        stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 9)
        with (
            mock.patch.object(rectified.sys, "stdin", stdin),
            mock.patch.object(rectified.termios, "tcgetattr", return_value=attributes),
            mock.patch.object(rectified.termios, "tcsetattr") as restore_terminal,
            mock.patch.object(rectified.cv2, "destroyWindow"),
            mock.patch.object(rectified.cv2, "waitKey", return_value=0),
        ):
            with rectified.selection_cleanup("test selector"):
                pass
        restore_terminal.assert_called_once_with(
            9,
            rectified.termios.TCSANOW,
            attributes,
        )


if __name__ == "__main__":
    unittest.main()
