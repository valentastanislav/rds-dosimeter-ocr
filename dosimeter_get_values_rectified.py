#!/usr/bin/env python3
"""
Perspective-rectified front-end for dosimeter_get_values_roi.py.

Workflow:

    video
      -> ROI-restricted display tracking
      -> perspective rectification
      -> original RDS decoder

For --select-quad the display detector is warmed up sequentially from
the beginning of the video.  This is important because the ROI detector
uses temporal tracking and may fail when asked to acquire an isolated
frame in the middle of the video.

Corner order:

    1 = top-left
    2 = top-right
    3 = bottom-right
    4 = bottom-left

Example:

    python3 dosimeter_get_values_rectified.py IMG_1151edited.mov out.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --select-quad \
        --contrast auto \
        --raw-output raw.csv \
        --debug-dir debug

Then reuse the printed quad:

    python3 dosimeter_get_values_rectified.py IMG_1151edited.mov out.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad x1,y1,x2,y2,x3,y3,x4,y4 \
        --contrast auto \
        --debug-dir debug
"""

from __future__ import annotations

import argparse
import sys
import termios
from contextlib import contextmanager

import cv2
import numpy as np


try:
    import dosimeter_get_values as core
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values.py.\n"
        "Put this script in the same directory."
    ) from exc


try:
    import dosimeter_get_values_roi as roi_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_roi.py.\n"
        "Put this script in the same directory."
    ) from exc


# Original ROI-aware detector from dosimeter_get_values_roi.py.
ORIGINAL_ROI_FINDER = roi_app.find_display_crop_roi


# ======================================================================
# Quad parsing / formatting
# ======================================================================


def parse_quad(text: str) -> np.ndarray:
    try:
        values = [
            float(value.strip())
            for value in text.split(",")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--quad must contain eight numbers"
        ) from exc

    if len(values) != 8:
        raise argparse.ArgumentTypeError(
            "--quad must contain exactly eight numbers: "
            "x1,y1,x2,y2,x3,y3,x4,y4"
        )

    points = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(4, 2)

    if (
        np.any(points < 0.0)
        or np.any(points > 1.0)
    ):
        raise argparse.ArgumentTypeError(
            "--quad coordinates must be between 0 and 1"
        )

    return points


def format_quad(
    quad: np.ndarray,
) -> str:
    return ",".join(
        f"{value:.6f}"
        for value in quad.ravel()
    )


# ======================================================================
# Sequential acquisition of reference display
# ======================================================================


def find_reference_display(
    video,
    info: core.VideoInfo,
    profile: core.Profile,
    roi,
    processing_width: int,
    sample_fps: float,
    target_time: float,
    search_after_s: float = 2.0,
) -> tuple[np.ndarray, float]:
    """
    Run the normal ROI display detector sequentially from t=0.

    This reproduces the real processing path, including previous_box
    tracking.  The valid display crop closest to target_time is returned.
    """

    previous_box = None

    best_display: np.ndarray | None = None
    best_time: float | None = None
    best_distance = float("inf")

    stop_time = min(
        info.duration,
        target_time + search_after_s,
    )

    for frame_index, frame in enumerate(
        core.iter_ffmpeg_frames(
            video,
            info,
            sample_fps,
            processing_width,
        )
    ):
        time_s = (
            frame_index
            / sample_fps
        )

        display, returned_box = (
            ORIGINAL_ROI_FINDER(
                frame,
                profile,
                previous_box,
                roi,
            )
        )

        if display is not None:
            previous_box = returned_box

            distance = abs(
                time_s
                - target_time
            )

            if distance < best_distance:
                best_distance = distance
                best_display = display.copy()
                best_time = time_s

        # We have already gone sufficiently past the requested time.
        if time_s >= stop_time:
            break

    if (
        best_display is None
        or best_time is None
    ):
        raise RuntimeError(
            "No valid display crop was found while scanning toward "
            f"t={target_time:.3f} s."
        )

    return (
        best_display,
        best_time,
    )


# ======================================================================
# Interactive four-corner selection
# ======================================================================


@contextmanager
def selection_cleanup(window_name: str):
    """Always close a selector window and restore terminal input state."""

    terminal_state = None
    try:
        if sys.stdin.isatty():
            file_descriptor = sys.stdin.fileno()
            terminal_state = (
                file_descriptor,
                termios.tcgetattr(file_descriptor),
            )
    except (OSError, ValueError, termios.error):
        terminal_state = None

    try:
        yield
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
        if terminal_state is not None:
            file_descriptor, attributes = terminal_state
            try:
                termios.tcsetattr(
                    file_descriptor,
                    termios.TCSANOW,
                    attributes,
                )
            except (OSError, ValueError, termios.error):
                pass


def select_quad(
    display: np.ndarray,
) -> np.ndarray:
    """
    Click:

        1. top-left
        2. top-right
        3. bottom-right
        4. bottom-left

    Then SPACE or ENTER.

    R resets the selection.
    C or ESC cancels.
    """

    if display.ndim == 2:
        original = cv2.cvtColor(
            display,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        original = display.copy()

    height, width = (
        original.shape[:2]
    )

    points: list[
        tuple[int, int]
    ] = []

    labels = (
        "TL",
        "TR",
        "BR",
        "BL",
    )

    window_name = (
        "Select corners: TL -> TR -> BR -> BL | "
        "SPACE/ENTER=accept  R=reset  C/ESC=cancel"
    )

    def redraw() -> None:
        canvas = original.copy()

        cv2.putText(
            canvas,
            (
                "Click: TL -> TR -> BR -> BL"
            ),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        for index, point in enumerate(
            points
        ):
            cv2.circle(
                canvas,
                point,
                6,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                labels[index],
                (
                    point[0] + 8,
                    point[1] - 7,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) >= 2:
            for index in range(
                len(points) - 1
            ):
                cv2.line(
                    canvas,
                    points[index],
                    points[index + 1],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if len(points) == 4:
            cv2.line(
                canvas,
                points[3],
                points[0],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            window_name,
            canvas,
        )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        userdata,
    ) -> None:
        del flags
        del userdata

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):
            points.append(
                (
                    int(x),
                    int(y),
                )
            )

            redraw()

    try:
        with selection_cleanup(window_name):
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

            cv2.setMouseCallback(
                window_name,
                mouse_callback,
            )

            redraw()

            while True:
                key = (
                    cv2.waitKey(30)
                    & 0xFF
                )

                if key in (
                    27,
                    ord("c"),
                    ord("C"),
                ):
                    raise RuntimeError(
                        "Quad selection cancelled."
                    )

                if key in (
                    ord("r"),
                    ord("R"),
                ):
                    points.clear()
                    redraw()
                    continue

                if (
                    len(points) == 4
                    and key in (
                        13,
                        10,
                        32,
                    )
                ):
                    break
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "Quad selection cancelled."
        ) from exc

    array = np.asarray(
        points,
        dtype=np.float32,
    )

    array[:, 0] /= float(
        width - 1
    )

    array[:, 1] /= float(
        height - 1
    )

    return array


# ======================================================================
# Perspective transformation
# ======================================================================


def rectify_display(
    display: np.ndarray,
    profile: core.Profile,
    normalized_quad: np.ndarray,
) -> np.ndarray:
    """
    Warp the selected physical display rectangle into the exact canonical
    geometry expected by the original decoder.
    """

    source_height, source_width = (
        display.shape[:2]
    )

    source = np.asarray(
        normalized_quad,
        dtype=np.float32,
    ).copy()

    source[:, 0] *= float(
        source_width - 1
    )

    source[:, 1] *= float(
        source_height - 1
    )

    destination = np.asarray(
        [
            [
                0.0,
                0.0,
            ],
            [
                profile.canonical_width - 1.0,
                0.0,
            ],
            [
                profile.canonical_width - 1.0,
                profile.canonical_height - 1.0,
            ],
            [
                0.0,
                profile.canonical_height - 1.0,
            ],
        ],
        dtype=np.float32,
    )

    matrix = (
        cv2.getPerspectiveTransform(
            source,
            destination,
        )
    )

    rectified = cv2.warpPerspective(
        display,
        matrix,
        (
            profile.canonical_width,
            profile.canonical_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rectified


# ======================================================================
# Extra command-line arguments
# ======================================================================


def parse_extra_arguments() -> tuple[
    argparse.Namespace,
    list[str],
]:
    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--select-quad",
        action="store_true",
    )

    parser.add_argument(
        "--quad",
        type=parse_quad,
        default=None,
    )

    parser.add_argument(
        "--quad-time",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--quad-search-after",
        type=float,
        default=2.0,
    )

    return parser.parse_known_args()


def parse_roi_arguments(
    remaining: list[str],
) -> argparse.Namespace:
    saved_argv = sys.argv

    try:
        sys.argv = [
            saved_argv[0],
            *remaining,
        ]

        return (
            roi_app.parse_args()
        )

    finally:
        sys.argv = saved_argv


# ======================================================================
# Main
# ======================================================================


def main() -> int:
    extra, remaining = (
        parse_extra_arguments()
    )

    args = parse_roi_arguments(
        remaining
    )

    if (
        extra.select_quad
        and extra.quad is not None
    ):
        print(
            "Error: use either --select-quad or --quad, not both.",
            file=sys.stderr,
        )
        return 1

    if (
        not extra.select_quad
        and extra.quad is None
    ):
        print(
            "Error: use --select-quad for the first run "
            "or supply an existing --quad.",
            file=sys.stderr,
        )
        return 1

    if args.roi is None:
        print(
            "Error: an explicit --roi is required.",
            file=sys.stderr,
        )
        return 1

    if extra.quad_search_after < 0:
        print(
            "Error: --quad-search-after cannot be negative.",
            file=sys.stderr,
        )
        return 1

    try:
        profile = (
            core.PROFILES[
                args.profile
            ]
        )

        info = core.probe_video(
            args.video
        )

        sample_fps = (
            args.sample_fps
            if args.sample_fps is not None
            else profile.default_sample_fps
        )

        normalized_quad = (
            extra.quad
        )

        # ==============================================================
        # Interactive quad selection
        # ==============================================================

        if extra.select_quad:
            requested_time = (
                extra.quad_time
                if extra.quad_time is not None
                else 0.5 * info.duration
            )

            requested_time = max(
                0.0,
                min(
                    requested_time,
                    info.duration,
                ),
            )

            print(
                (
                    "Scanning sequentially toward "
                    f"t={requested_time:.3f} s "
                    "to acquire the display..."
                ),
                file=sys.stderr,
            )

            reference_display, actual_time = (
                find_reference_display(
                    args.video,
                    info,
                    profile,
                    args.roi,
                    args.processing_width,
                    sample_fps,
                    requested_time,
                    search_after_s=(
                        extra.quad_search_after
                    ),
                )
            )

            print(
                (
                    "Reference display acquired at "
                    f"t={actual_time:.3f} s."
                ),
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

            print(
                "Click the four corners of the DISPLAY BEZEL:",
                file=sys.stderr,
            )

            print(
                "  1 = top-left",
                file=sys.stderr,
            )

            print(
                "  2 = top-right",
                file=sys.stderr,
            )

            print(
                "  3 = bottom-right",
                file=sys.stderr,
            )

            print(
                "  4 = bottom-left",
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

            normalized_quad = (
                select_quad(
                    reference_display
                )
            )

            print(
                "",
                file=sys.stderr,
            )

            print(
                "Selected perspective quad:",
                file=sys.stderr,
            )

            print(
                (
                    "  --quad "
                    + format_quad(
                        normalized_quad
                    )
                ),
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

        if normalized_quad is None:
            raise RuntimeError(
                "No quadrilateral available."
            )

        # ==============================================================
        # Replace the ROI display finder.
        #
        # IMPORTANT:
        # Everything downstream now sees the rectified image:
        #
        #   - segment extraction
        #   - decimal extraction
        #   - adaptive y_shift
        #   - debug green overlay
        #
        # Thus the debug grid corresponds to the image actually decoded.
        # ==============================================================

        def rectified_finder(
            frame,
            in_profile,
            previous_box,
            roi,
        ):
            display, box = (
                ORIGINAL_ROI_FINDER(
                    frame,
                    in_profile,
                    previous_box,
                    roi,
                )
            )

            if display is None:
                return (
                    None,
                    box,
                )

            return (
                rectify_display(
                    display,
                    in_profile,
                    normalized_quad,
                ),
                box,
            )

        roi_app.find_display_crop_roi = (
            rectified_finder
        )

        # ==============================================================
        # Run standard ROI pipeline
        # ==============================================================

        saved_argv = sys.argv

        try:
            sys.argv = [
                saved_argv[0],
                *remaining,
            ]

            result = (
                roi_app.main()
            )

        finally:
            sys.argv = (
                saved_argv
            )

        print()

        print(
            "Perspective quad used:"
        )

        print(
            (
                "  --quad "
                + format_quad(
                    normalized_quad
                )
            )
        )

        return result

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
