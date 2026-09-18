#!/usr/bin/env python3

from __future__ import annotations

import io
import re
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np

import dosimeter_get_values_flow as flow
import dosimeter_get_values_rectified as rectified
import dosimeter_get_values_flow_digitshape as digitshape


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

    def test_decimal_candidates_align_with_transformed_bottom_segment(self) -> None:
        bottom = np.asarray(
            ((10, 120), (40, 120), (40, 129), (10, 129)),
            dtype=np.int32,
        )
        per_digit = tuple(
            {"d": bottom.copy()}
            for _ in range(3)
        )
        profile = replace(
            flow.core.RDS200,
            canonical_height=356,
            digit_boxes=(
                (122, 166, 194, 279),
                (192, 166, 264, 279),
                (262, 166, 334, 279),
            ),
            digit_segment_polygons=per_digit,
            decimal_candidates=(
                (180, 246, 191, 259, 2),
                (251, 246, 261, 259, 1),
            ),
        )

        aligned = digitshape.align_decimal_candidates_to_bottom_segments(
            profile
        )

        expected_y2 = round(
            np.median(
                [
                    y1 + 129 * (y2 - y1) / 130.0
                    for _x1, y1, _x2, y2 in profile.digit_boxes
                ]
            )
        )
        self.assertEqual(
            tuple(item[3] for item in aligned.decimal_candidates),
            (expected_y2, expected_y2),
        )
        self.assertEqual(
            tuple(item[3] - item[1] for item in aligned.decimal_candidates),
            (13, 13),
        )
        widths = tuple(
            item[2] - item[0]
            for item in aligned.decimal_candidates
        )
        self.assertEqual(widths, (11, 10))

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
            for item in aligned.decimal_candidates
        )
        for actual, expected in zip(actual_centres, expected_boundaries):
            self.assertAlmostEqual(actual, expected, delta=0.5)

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
