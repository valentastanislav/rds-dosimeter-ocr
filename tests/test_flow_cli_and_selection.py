#!/usr/bin/env python3

from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import dosimeter_get_values_flow as flow
import dosimeter_get_values_rectified as rectified


class FlowCliAndSelectionTest(unittest.TestCase):
    def test_wrapper_error_usage_starts_with_positional_files(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", stderr):
            flow.parse_wrapper_args([])
        self.assertEqual(
            stderr.getvalue().splitlines()[0],
            "usage: python3 dosimeter_get_values_flow.py "
            "<video file> <output file> [options]",
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
