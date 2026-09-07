#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

import cv2

import dosimeter_get_values as core
import dosimeter_get_values_roi as roi_app
import dosimeter_get_values_rectified as rect_app


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video"
    )

    parser.add_argument(
        "--profile",
        default="rds200",
        choices=sorted(core.PROFILES),
    )

    parser.add_argument(
        "--roi",
        required=True,
        type=roi_app.parse_roi,
    )

    parser.add_argument(
        "--quad",
        required=True,
        type=rect_app.parse_quad,
    )

    parser.add_argument(
        "--time",
        type=float,
        default=6.8,
        help="reference time in seconds (default: 6.8)",
    )

    parser.add_argument(
        "--processing-width",
        type=int,
        default=540,
    )

    args = parser.parse_args()

    try:

        video = __import__("pathlib").Path(
            args.video
        )

        profile = core.PROFILES[
            args.profile
        ]

        info = core.probe_video(
            video
        )

        sample_fps = (
            profile.default_sample_fps
        )

        print(
            f"Finding display near t={args.time:.3f} s...",
            file=sys.stderr,
        )

        display, actual_time = (
            rect_app.find_reference_display(
                video,
                info,
                profile,
                args.roi,
                args.processing_width,
                sample_fps,
                args.time,
            )
        )

        print(
            f"Display found at t={actual_time:.3f} s.",
            file=sys.stderr,
        )

        rectified = (
            rect_app.rectify_display(
                display,
                profile,
                args.quad,
            )
        )

        # ------------------------------------------------------
        # IMPORTANT:
        # absolutely NO old digit boxes are drawn here.
        # ------------------------------------------------------

        shown = cv2.cvtColor(
            rectified,
            cv2.COLOR_GRAY2BGR,
        )

        cv2.putText(
            shown,
            "Select ONLY the complete 3 large digits",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            shown,
            "Include full top/bottom/left/right segments",
            (10, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        window = (
            "Select complete 3-digit LCD area"
        )

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )

        x, y, width, height = (
            cv2.selectROI(
                window,
                shown,
                showCrosshair=True,
                fromCenter=False,
            )
        )

        cv2.destroyAllWindows()

        if (
            width <= 0
            or height <= 0
        ):

            raise RuntimeError(
                "Selection cancelled."
            )

        h, w = (
            rectified.shape[:2]
        )

        grid = (
            x / w,
            y / h,
            (x + width) / w,
            (y + height) / h,
        )

        print()
        print("Selected grid:")
        print(
            "  --grid "
            + ",".join(
                f"{value:.6f}"
                for value in grid
            )
        )

        print()
        print("Pixel coordinates:")
        print(
            f"  x = {x} .. {x + width}"
        )
        print(
            f"  y = {y} .. {y + height}"
        )
        print(
            f"  width  = {width}"
        )
        print(
            f"  height = {height}"
        )

        # Save exactly what was selected for later inspection.
        preview = shown.copy()

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + width,
                y + height,
            ),
            (0, 255, 0),
            2,
        )

        filename = (
            "selected_grid_reference.jpg"
        )

        cv2.imwrite(
            filename,
            preview,
        )

        print()
        print(
            f"Preview written to: {filename}"
        )

        return 0

    except (
        RuntimeError,
        ValueError,
        OSError,
        cv2.error,
    ) as exc:

        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )