#!/usr/bin/env python3
"""
ROI-enabled front-end for dosimeter_get_values.py.

Adds:
    --select-roi
        Interactively select the region in which the dosimeter display
        is allowed to occur.

    --roi x1,y1,x2,y2
        Reuse a normalized ROI without opening a GUI.
        Coordinates are fractions of the oriented video frame:
        0 <= x1 < x2 <= 1
        0 <= y1 < y2 <= 1

The original dosimeter_get_values.py remains unchanged and supplies
the seven-segment decoding, temporal processing, interval creation,
summary generation, and debug overlays.

Recommended first run:

    python3 dosimeter_get_values_roi.py IMG_1151.MOV out.csv \
        --profile rds200 \
        --select-roi \
        --raw-output raw1151.csv \
        --debug-dir debug1151

The selected ROI is printed in a reusable form, for example:

    --roi 0.3125,0.2210,0.7012,0.6845

Then the same geometry can be reused without GUI:

    python3 dosimeter_get_values_roi.py IMG_1152.MOV out.csv \
        --profile rds200 \
        --roi 0.3125,0.2210,0.7012,0.6845
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

try:
    import dosimeter_get_values as core
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values.py.\n"
        "Put dosimeter_get_values_roi.py in the same directory as "
        "dosimeter_get_values.py."
    ) from exc


ROI = tuple[float, float, float, float]
Box = tuple[int, int, int, int]
DarknessExtractor = Callable[
    [
        np.ndarray,
        core.Profile,
        Sequence[np.ndarray],
    ],
    np.ndarray,
]
DecodeSamples = Callable[
    ...,
    list[core.DecodedSample],
]


# Keep the original detector.  With no ROI, the new script can therefore
# reproduce the old behaviour exactly.
ORIGINAL_FIND_DISPLAY_CROP = core.find_display_crop


# ======================================================================
# ROI parsing / conversion
# ======================================================================


def parse_roi(text: str) -> ROI:
    try:
        values = tuple(float(x.strip()) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--roi must contain four numbers: x1,y1,x2,y2"
        ) from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "--roi must contain four numbers: x1,y1,x2,y2"
        )

    x1, y1, x2, y2 = values

    if not (
        0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "--roi coordinates must satisfy "
            "0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )

    return x1, y1, x2, y2


def format_roi(roi: ROI) -> str:
    return ",".join(f"{value:.6f}" for value in roi)


def expand_roi(
    roi: ROI,
    fraction: float,
) -> ROI:
    x1, y1, x2, y2 = roi

    width = x2 - x1
    height = y2 - y1

    x_padding = fraction * width
    y_padding = fraction * height

    return (
        max(0.0, x1 - x_padding),
        max(0.0, y1 - y_padding),
        min(1.0, x2 + x_padding),
        min(1.0, y2 + y_padding),
    )


def roi_to_pixels(
    roi: ROI,
    frame_width: int,
    frame_height: int,
) -> Box:
    x1, y1, x2, y2 = roi

    px1 = int(round(x1 * frame_width))
    py1 = int(round(y1 * frame_height))
    px2 = int(round(x2 * frame_width))
    py2 = int(round(y2 * frame_height))

    px1 = max(0, min(px1, frame_width - 1))
    py1 = max(0, min(py1, frame_height - 1))

    px2 = max(px1 + 1, min(px2, frame_width))
    py2 = max(py1 + 1, min(py2, frame_height))

    return (
        px1,
        py1,
        px2 - px1,
        py2 - py1,
    )


# ======================================================================
# Read one frame with the same ffmpeg orientation/scaling as the
# normal video processing.
# ======================================================================


def read_frame_at_time(
    video: Path,
    info: core.VideoInfo,
    time_s: float,
    processing_width: int,
) -> np.ndarray:
    oriented_width, oriented_height = info.oriented_size

    processing_height = core.even_number(
        processing_width
        * oriented_height
        / oriented_width
    )

    time_s = max(
        0.0,
        min(float(time_s), max(0.0, info.duration - 0.01)),
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{time_s:.6f}",
        "-i",
        str(video),
        "-an",
        "-frames:v",
        "1",
        "-vf",
        f"scale={processing_width}:{processing_height}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH."
        ) from exc

    if result.returncode != 0:
        message = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        raise RuntimeError(
            f"Could not extract ROI selection frame:\n{message}"
        )

    expected_size = (
        processing_width
        * processing_height
        * 3
    )

    if len(result.stdout) < expected_size:
        raise RuntimeError(
            "ffmpeg returned an incomplete ROI selection frame."
        )

    return np.frombuffer(
        result.stdout[:expected_size],
        dtype=np.uint8,
    ).reshape(
        processing_height,
        processing_width,
        3,
    )


# ======================================================================
# Interactive ROI selection
# ======================================================================


def select_roi_interactively(
    video: Path,
    info: core.VideoInfo,
    processing_width: int,
    time_s: float,
    padding: float,
) -> ROI:
    frame = read_frame_at_time(
        video,
        info,
        time_s,
        processing_width,
    )

    height, width = frame.shape[:2]

    window_name = (
        "Select dosimeter DISPLAY area - "
        "ENTER/SPACE = accept, ESC = cancel"
    )

    try:
        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        x, y, w, h = cv2.selectROI(
            window_name,
            frame,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow(window_name)

    except cv2.error as exc:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        raise RuntimeError(
            "OpenCV could not open the ROI selection window. "
            "Use --roi x1,y1,x2,y2 instead."
        ) from exc

    if w <= 0 or h <= 0:
        raise RuntimeError(
            "ROI selection was cancelled."
        )

    roi = (
        x / width,
        y / height,
        (x + w) / width,
        (y + h) / height,
    )

    # A little margin is useful because the contour detector needs
    # to see the complete black bezel.
    roi = expand_roi(
        roi,
        padding,
    )

    return roi


# ======================================================================
# Safe normalized crop
# ======================================================================


def normalized_display_from_box(
    gray: np.ndarray,
    box: Box,
    profile: core.Profile,
) -> np.ndarray | None:
    frame_height, frame_width = gray.shape

    x, y, width, height = box

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(
        frame_width,
        x + width,
    )

    y2 = min(
        frame_height,
        y + height,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = gray[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (
            profile.canonical_width,
            profile.canonical_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


# ======================================================================
# Candidate validation
#
# This is deliberately conservative.
#
# A random dark rectangle in the room can satisfy the geometric cuts
# of the old detector.  A real RADOS display should additionally have:
#
#   - a dark bezel surrounding a lighter LCD region,
#   - significant image structure exactly where the digit boxes live.
# ======================================================================


def display_candidate_metrics(
    display: np.ndarray,
    profile: core.Profile,
) -> tuple[bool, float, float, float]:
    height, width = display.shape

    if height < 20 or width < 20:
        return False, 0.0, 0.0, float("-inf")

    border_x = max(
        3,
        int(round(0.07 * width)),
    )

    border_y = max(
        3,
        int(round(0.07 * height)),
    )

    border = np.concatenate(
        (
            display[:border_y, :].ravel(),
            display[-border_y:, :].ravel(),
            display[:, :border_x].ravel(),
            display[:, -border_x:].ravel(),
        )
    )

    # Envelope containing all configured digit fields.
    digit_x1 = min(
        box[0]
        for box in profile.digit_boxes
    )

    digit_y1 = min(
        box[1]
        for box in profile.digit_boxes
    )

    digit_x2 = max(
        box[2]
        for box in profile.digit_boxes
    )

    digit_y2 = max(
        box[3]
        for box in profile.digit_boxes
    )

    digit_width = digit_x2 - digit_x1
    digit_height = digit_y2 - digit_y1

    # Include some LCD around the digits, not just the segments.
    lcd_x1 = max(
        0,
        int(round(digit_x1 - 0.25 * digit_width)),
    )

    lcd_x2 = min(
        width,
        int(round(digit_x2 + 0.25 * digit_width)),
    )

    lcd_y1 = max(
        0,
        int(round(digit_y1 - 0.30 * digit_height)),
    )

    lcd_y2 = min(
        height,
        int(round(digit_y2 + 0.30 * digit_height)),
    )

    lcd_region = display[
        lcd_y1:lcd_y2,
        lcd_x1:lcd_x2,
    ]

    digit_region = display[
        digit_y1:digit_y2,
        digit_x1:digit_x2,
    ]

    if (
        border.size == 0
        or lcd_region.size == 0
        or digit_region.size == 0
    ):
        return False, 0.0, 0.0, float("-inf")

    # The LCD need not be uniformly bright.  A high percentile is
    # more reliable under reflections than the mean.
    lcd_level = float(
        np.percentile(
            lcd_region,
            70,
        )
    )

    bezel_level = float(
        np.percentile(
            border,
            50,
        )
    )

    bezel_contrast = (
        lcd_level
        - bezel_level
    )

    digit_p10 = float(
        np.percentile(
            digit_region,
            10,
        )
    )

    digit_p90 = float(
        np.percentile(
            digit_region,
            90,
        )
    )

    digit_range = (
        digit_p90
        - digit_p10
    )

    patch_ranges: list[float] = []

    for patch in core.digit_patches(
        display,
        profile,
    ):
        patch_ranges.append(
            float(
                np.percentile(
                    patch,
                    90,
                )
                - np.percentile(
                    patch,
                    10,
                )
            )
        )

    median_patch_range = (
        float(np.median(patch_ranges))
        if patch_ranges
        else 0.0
    )

    maximum_patch_range = (
        max(patch_ranges)
        if patch_ranges
        else 0.0
    )

    # Conservative validity test:
    #
    # - real LCD should be lighter than bezel,
    # - digit area should not be essentially uniform,
    # - at least one individual digit location needs visible structure.
    valid = (
        bezel_contrast >= 5.0
        and digit_range >= 7.0
        and maximum_patch_range >= 8.0
    )

    quality = (
        1.2 * bezel_contrast
        + 0.6 * digit_range
        + 0.6 * median_patch_range
    )

    return (
        valid,
        bezel_contrast,
        digit_range,
        float(quality),
    )


# ======================================================================
# Local tracker around an already known display box
# ======================================================================


def local_tracking_candidates(
    gray: np.ndarray,
    profile: core.Profile,
    previous_box: Box,
    roi_box: Box,
) -> list[
    tuple[
        float,
        Box,
        np.ndarray,
    ]
]:
    px, py, pw, ph = previous_box

    rx, ry, rw, rh = roi_box

    roi_x2 = rx + rw
    roi_y2 = ry + rh

    previous_center_x = (
        px + pw / 2.0
    )

    previous_center_y = (
        py + ph / 2.0
    )

    candidates: list[
        tuple[
            float,
            Box,
            np.ndarray,
        ]
    ] = []

    scales = (
        0.96,
        1.00,
        1.04,
    )

    offsets = (
        -0.04,
        0.0,
        0.04,
    )

    for scale in scales:
        width = max(
            2,
            int(round(pw * scale)),
        )

        height = max(
            2,
            int(round(ph * scale)),
        )

        for dx in offsets:
            for dy in offsets:
                center_x = (
                    previous_center_x
                    + dx * pw
                )

                center_y = (
                    previous_center_y
                    + dy * ph
                )

                x = int(
                    round(
                        center_x
                        - width / 2.0
                    )
                )

                y = int(
                    round(
                        center_y
                        - height / 2.0
                    )
                )

                # Stay inside the user-approved ROI.
                if (
                    x < rx
                    or y < ry
                    or x + width > roi_x2
                    or y + height > roi_y2
                ):
                    continue

                box = (
                    x,
                    y,
                    width,
                    height,
                )

                display = normalized_display_from_box(
                    gray,
                    box,
                    profile,
                )

                if display is None:
                    continue

                (
                    valid,
                    _bezel_contrast,
                    _digit_range,
                    quality,
                ) = display_candidate_metrics(
                    display,
                    profile,
                )

                if not valid:
                    continue

                movement_penalty = (
                    10.0 * abs(dx)
                    + 10.0 * abs(dy)
                    + 12.0 * abs(scale - 1.0)
                )

                score = (
                    quality
                    - movement_penalty
                )

                candidates.append(
                    (
                        score,
                        box,
                        display,
                    )
                )

    return candidates


# ======================================================================
# ROI-aware display detector
# ======================================================================


def find_display_crop_roi(
    frame: np.ndarray,
    profile: core.Profile,
    previous_box: Box | None,
    roi: ROI | None,
) -> tuple[
    np.ndarray | None,
    Box | None,
]:
    # No ROI: preserve original behaviour.
    if roi is None:
        return ORIGINAL_FIND_DISPLAY_CROP(
            frame,
            profile,
            previous_box,
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    frame_height, frame_width = gray.shape

    roi_box = roi_to_pixels(
        roi,
        frame_width,
        frame_height,
    )

    rx, ry, rw, rh = roi_box

    roi_gray = gray[
        ry:ry + rh,
        rx:rx + rw,
    ]

    if roi_gray.size == 0:
        return None, previous_box

    candidates: list[
        tuple[
            float,
            Box,
            np.ndarray,
        ]
    ] = []

    # ----------------------------------------------------------
    # First try to stay close to the previously verified display.
    # ----------------------------------------------------------

    if previous_box is not None:
        candidates.extend(
            local_tracking_candidates(
                gray,
                profile,
                previous_box,
                roi_box,
            )
        )

    # ----------------------------------------------------------
    # Global reacquisition, but ONLY inside the selected ROI.
    # ----------------------------------------------------------

    blurred = cv2.GaussianBlur(
        roi_gray,
        (5, 5),
        0,
    )

    otsu_threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    threshold_candidates = {
        int(profile.display_threshold),
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        int(round(float(otsu_threshold))),
    }

    threshold_candidates = sorted(
        value
        for value in threshold_candidates
        if 30 <= value <= 150
    )

    roi_area = float(
        rw * rh
    )

    preferred_aspect = (
        0.5
        * (
            profile.display_aspect_min
            + profile.display_aspect_max
        )
    )

    for threshold in threshold_candidates:
        binary = (
            blurred < threshold
        ).astype(
            np.uint8
        ) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones(
                (5, 5),
                dtype=np.uint8,
            ),
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            local_x, local_y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            if (
                width <= 0
                or height <= 0
            ):
                continue

            bounding_area = float(
                width * height
            )

            contour_area = float(
                cv2.contourArea(
                    contour
                )
            )

            if bounding_area <= 0:
                continue

            area_fraction = (
                bounding_area
                / max(roi_area, 1.0)
            )

            aspect = (
                width / height
            )

            fill = (
                contour_area
                / bounding_area
            )

            # ROI already removes most background, so these can be
            # looser than the original full-frame cuts.
            if not (
                0.95 <= aspect <= 1.95
            ):
                continue

            if area_fraction < 0.015:
                continue

            # Thresholds that turn essentially the entire ROI black
            # are not useful.
            if area_fraction > 0.92:
                continue

            if fill < 0.25:
                continue

            x = (
                rx + local_x
            )

            y = (
                ry + local_y
            )

            box = (
                x,
                y,
                width,
                height,
            )

            display = normalized_display_from_box(
                gray,
                box,
                profile,
            )

            if display is None:
                continue

            (
                valid,
                _bezel_contrast,
                _digit_range,
                quality,
            ) = display_candidate_metrics(
                display,
                profile,
            )

            # This is the important protection against a random dark
            # piece of background becoming a fake dosimeter display.
            if not valid:
                continue

            aspect_error = abs(
                math.log(
                    max(
                        aspect,
                        1e-6,
                    )
                    / preferred_aspect
                )
            )

            score = quality

            score += (
                8.0
                * math.exp(
                    -2.0
                    * aspect_error
                )
            )

            score += (
                4.0
                * min(
                    fill,
                    1.0,
                )
            )

            if previous_box is not None:
                px, py, pw, ph = (
                    previous_box
                )

                previous_center_x = (
                    px + pw / 2.0
                )

                previous_center_y = (
                    py + ph / 2.0
                )

                center_x = (
                    x + width / 2.0
                )

                center_y = (
                    y + height / 2.0
                )

                distance = math.hypot(
                    center_x
                    - previous_center_x,
                    center_y
                    - previous_center_y,
                )

                reference_size = max(
                    pw,
                    ph,
                    1,
                )

                normalized_distance = (
                    distance
                    / reference_size
                )

                scale_difference = abs(
                    math.log(
                        max(width, 1)
                        / max(pw, 1)
                    )
                )

                score += (
                    12.0
                    * math.exp(
                        -1.5
                        * normalized_distance
                    )
                )

                score += (
                    6.0
                    * math.exp(
                        -2.0
                        * scale_difference
                    )
                )

            candidates.append(
                (
                    float(score),
                    box,
                    display,
                )
            )

    if not candidates:
        # Important: do NOT invent a display from an invalid object.
        return None, previous_box

    score, box, display = max(
        candidates,
        key=lambda item: item[0],
    )

    del score

    return (
        display,
        box,
    )


# ======================================================================
# Debug output using the same ROI-aware finder
# ======================================================================


def save_debug_screenshots_roi(
    video: Path,
    info: core.VideoInfo,
    profile: core.Profile,
    sample_fps: float,
    processing_width: int,
    intervals: Sequence[
        tuple[
            float,
            float,
            core.Run,
        ]
    ],
    output_dir: Path,
    roi: ROI | None,
    contrast_mode: str,
) -> None:
    old_finder = core.find_display_crop

    def roi_finder(
        frame: np.ndarray,
        in_profile: core.Profile,
        previous_box: Box | None,
    ) -> tuple[
        np.ndarray | None,
        Box | None,
    ]:
        return find_display_crop_roi(
            frame,
            in_profile,
            previous_box,
            roi,
        )

    core.find_display_crop = (
        roi_finder
    )

    try:
        core.save_debug_screenshots(
            video,
            info,
            profile,
            sample_fps,
            processing_width,
            intervals,
            output_dir,
            contrast_mode=contrast_mode,
        )
    finally:
        core.find_display_crop = (
            old_finder
        )


# ======================================================================
# CLI
# ======================================================================


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract changing seven-segment values from a "
            "dosimeter video, with optional ROI restriction."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="input video, for example video.MOV",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="interval CSV output",
    )

    parser.add_argument(
        "--profile",
        choices=sorted(
            core.PROFILES
        ),
        default="rds200",
        help=(
            "dosimeter profile "
            "(default: rds200)"
        ),
    )

    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help=(
            "restrict display search to normalized "
            "x1,y1,x2,y2 coordinates, for example "
            "0.25,0.20,0.75,0.75"
        ),
    )

    parser.add_argument(
        "--select-roi",
        action="store_true",
        help=(
            "show one video frame and interactively select "
            "the region containing the dosimeter display"
        ),
    )

    parser.add_argument(
        "--roi-time",
        type=float,
        default=None,
        help=(
            "time in seconds used for --select-roi; "
            "default is the middle of the video"
        ),
    )

    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.08,
        help=(
            "relative padding automatically added around an "
            "interactively selected ROI (default: 0.08)"
        ),
    )

    parser.add_argument(
        "--decimal-places",
        choices=(
            "auto",
            "0",
            "1",
            "2",
            "3",
        ),
        default="auto",
        help=(
            "decimal places: auto detects the moving "
            "RDS-200 decimal point; a number forces a "
            "fixed position (default: auto)"
        ),
    )

    parser.add_argument(
        "--contrast",
        choices=(
            "auto",
            "none",
            "clahe",
        ),
        default="auto",
        help=(
            "RDS-200 digit contrast handling "
            "(default: auto)"
        ),
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "reject samples below this digit confidence; "
            "profile default if omitted"
        ),
    )

    parser.add_argument(
        "--decimal-switch-penalty",
        type=float,
        default=4.0,
        help=(
            "penalty for moving the RDS-200 decimal point "
            "between adjacent samples (default: 4.0)"
        ),
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help=(
            "samples per second; "
            "profile default if omitted"
        ),
    )

    parser.add_argument(
        "--filter-window",
        "--median-window",
        dest="filter_window",
        type=int,
        default=None,
        help=(
            "odd temporal segment-filter window; "
            "profile default if omitted"
        ),
    )

    parser.add_argument(
        "--mode-window",
        type=int,
        default=None,
        help=(
            "odd temporal value-mode window; "
            "profile default if omitted"
        ),
    )

    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.6,
        help=(
            "merge recognized intervals shorter than "
            "this many seconds (default: 0.6)"
        ),
    )

    parser.add_argument(
        "--processing-width",
        type=int,
        default=540,
        help=(
            "working video width used by ffmpeg "
            "(default: 540)"
        ),
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "summary JSON path; "
            "default is OUTPUT.summary.json"
        ),
    )

    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help=(
            "optional CSV containing every decoded sample "
            "before interval merging"
        ),
    )

    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for normalized display "
            "debug screenshots"
        ),
    )

    return parser.parse_args(argv)


# ======================================================================
# Validation
# ======================================================================


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.video.is_file():
        raise RuntimeError(
            f"Input video does not exist: {args.video}"
        )

    if (
        args.roi is not None
        and args.select_roi
    ):
        raise RuntimeError(
            "Use either --roi or --select-roi, not both."
        )

    if (
        args.roi_time is not None
        and args.roi_time < 0
    ):
        raise RuntimeError(
            "--roi-time cannot be negative."
        )

    if not (
        0.0 <= args.roi_padding <= 1.0
    ):
        raise RuntimeError(
            "--roi-padding must be between 0 and 1."
        )

    if (
        args.sample_fps is not None
        and args.sample_fps <= 0
    ):
        raise RuntimeError(
            "--sample-fps must be positive."
        )

    if args.processing_width < 200:
        raise RuntimeError(
            "--processing-width must be at least 200 pixels."
        )

    for name, value in (
        (
            "--filter-window",
            args.filter_window,
        ),
        (
            "--mode-window",
            args.mode_window,
        ),
    ):
        if (
            value is not None
            and (
                value < 1
                or value % 2 == 0
            )
        ):
            raise RuntimeError(
                f"{name} must be a positive odd integer."
            )

    if args.min_interval < 0:
        raise RuntimeError(
            "--min-interval cannot be negative."
        )

    if (
        args.min_confidence is not None
        and not (
            0.0
            <= args.min_confidence
            <= 1.0
        )
    ):
        raise RuntimeError(
            "--min-confidence must be between 0 and 1."
        )

    if (
        args.decimal_switch_penalty < 0
    ):
        raise RuntimeError(
            "--decimal-switch-penalty cannot be negative."
        )


# ======================================================================
# Main
# ======================================================================


def main(
    argv: Sequence[str] | None = None,
    darkness_extractor: DarknessExtractor | None = None,
    profile_override: core.Profile | None = None,
    decode_samples: DecodeSamples | None = None,
) -> int:
    args = parse_args(argv)

    selected_darkness_extractor = (
        core.extract_darkness
        if darkness_extractor is None
        else darkness_extractor
    )

    selected_decode_samples = (
        core.decode_samples
        if decode_samples is None
        else decode_samples
    )

    try:
        validate_args(
            args
        )

        profile = (
            core.PROFILES[
                args.profile
            ]
            if profile_override is None
            else profile_override
        )

        sample_fps = (
            args.sample_fps
            if args.sample_fps is not None
            else profile.default_sample_fps
        )

        filter_window = (
            args.filter_window
            if args.filter_window is not None
            else profile.default_filter_window
        )

        mode_window = (
            args.mode_window
            if args.mode_window is not None
            else profile.default_mode_window
        )

        minimum_confidence = (
            args.min_confidence
            if args.min_confidence is not None
            else profile.default_min_confidence
        )

        info = core.probe_video(
            args.video
        )

        roi: ROI | None = (
            args.roi
        )

        if args.select_roi:
            roi_time = (
                args.roi_time
                if args.roi_time is not None
                else 0.5 * info.duration
            )

            print(
                f"Selecting ROI from t={roi_time:.3f} s...",
                file=sys.stderr,
            )

            roi = select_roi_interactively(
                args.video,
                info,
                args.processing_width,
                roi_time,
                args.roi_padding,
            )

            print(
                "",
                file=sys.stderr,
            )

            print(
                "Selected ROI:",
                file=sys.stderr,
            )

            print(
                f"  --roi {format_roi(roi)}",
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

        if roi is not None:
            print(
                f"ROI restriction: {format_roi(roi)}",
                file=sys.stderr,
            )

        segment_masks = (
            core.make_segment_masks(
                profile
            )
        )

        shift_mask_bank = (
            core.make_y_shift_mask_bank(
                profile
            )
        )

        samples: list[
            core.Sample
        ] = []

        previous_box: (
            Box | None
        ) = None

        previous_shift: (
            int | None
        ) = None

        print(
            f"Processing {args.video} "
            f"with profile {profile.name} "
            f"at {sample_fps:g} samples/s "
            f"(contrast={args.contrast}, "
            f"min_confidence={minimum_confidence:g})...",
            file=sys.stderr,
        )

        for (
            frame_index,
            frame,
        ) in enumerate(
            core.iter_ffmpeg_frames(
                args.video,
                info,
                sample_fps,
                args.processing_width,
            )
        ):
            (
                display,
                new_box,
            ) = find_display_crop_roi(
                frame,
                profile,
                previous_box,
                roi,
            )

            # Update tracking only after a VALID display crop.
            if display is not None:
                previous_box = (
                    new_box
                )

            if display is None:
                samples.append(
                    core.Sample(
                        frame_index
                        / sample_fps,
                        None,
                        display_found=False,
                        aux_darkness=None,
                        decimal_scores=None,
                        alignment_shift=(
                            previous_shift
                            or 0
                        ),
                    )
                )

                continue

            if profile.adaptive_y_shift:
                effective_contrast = (
                    args.contrast
                    if profile.name
                    == "rds200"
                    else "none"
                )

                (
                    measured_darkness,
                    previous_shift,
                ) = core.extract_darkness_adaptive(
                    display,
                    profile,
                    shift_mask_bank,
                    previous_shift,
                    effective_contrast,
                )

            else:
                measured_darkness = (
                    selected_darkness_extractor(
                        display,
                        profile,
                        segment_masks,
                    )
                )

                previous_shift = 0

            auxiliary_darkness = None

            if (
                profile.aux_segment_percentile
                is not None
            ):
                active_masks = (
                    shift_mask_bank.get(
                        previous_shift
                        or 0,
                        segment_masks,
                    )
                )

                auxiliary_darkness = (
                    core.extract_darkness_from_patches(
                        core.digit_patches(
                            display,
                            profile,
                        ),
                        active_masks,
                        segment_percentile=(
                            profile.aux_segment_percentile
                        ),
                    )
                )

            samples.append(
                core.Sample(
                    frame_index
                    / sample_fps,
                    measured_darkness,
                    display_found=True,
                    aux_darkness=(
                        auxiliary_darkness
                    ),
                    decimal_scores=(
                        core.extract_decimal_scores(
                            display,
                            profile,
                            y_shift=(
                                previous_shift
                                or 0
                            ),
                        )
                    ),
                    alignment_shift=(
                        previous_shift
                        or 0
                    ),
                )
            )

        if not samples:
            raise RuntimeError(
                "No frames were decoded from the video."
            )

        decimal_places_override = (
            None
            if args.decimal_places
            == "auto"
            else int(
                args.decimal_places
            )
        )

        decoded_filtered = (
            selected_decode_samples(
                samples,
                profile,
                filter_window,
                decimal_places_override=(
                    decimal_places_override
                ),
                minimum_confidence=(
                    minimum_confidence
                ),
                decimal_switch_penalty=(
                    args.decimal_switch_penalty
                ),
            )
        )

        # --------------------------------------------------
        # Important:
        #
        # The temporal segment filter is allowed to use
        # neighbouring frames, but a frame in which the display
        # itself was NOT found must not count as a directly
        # recognized sample.
        #
        # Gap filling happens later and is explicitly marked with
        # confidence=0.
        # --------------------------------------------------

        decoded_raw: list[
            core.DecodedSample
        ] = []

        for (
            sample,
            decoded,
        ) in zip(
            samples,
            decoded_filtered,
        ):
            if sample.display_found:
                decoded_raw.append(
                    decoded
                )
            else:
                decoded_raw.append(
                    core.DecodedSample(
                        decoded.time_s,
                        None,
                        0.0,
                    )
                )

        display_found_count = sum(
            sample.display_found
            for sample in samples
        )

        recognized_count = sum(
            sample.value is not None
            for sample in decoded_raw
        )

        display_found_fraction = (
            display_found_count
            / len(samples)
        )

        recognized_fraction = (
            recognized_count
            / len(decoded_raw)
        )

        if display_found_count:
            recognized_given_display = (
                recognized_count
                / display_found_count
            )
        else:
            recognized_given_display = (
                0.0
            )

        print(
            f"Display detection: "
            f"{100.0 * display_found_fraction:.1f}% "
            f"({display_found_count}/{len(samples)} frames)",
            file=sys.stderr,
        )

        print(
            f"Digit recognition: "
            f"{100.0 * recognized_fraction:.1f}% "
            f"({recognized_count}/{len(decoded_raw)} frames)",
            file=sys.stderr,
        )

        print(
            f"Recognition when display found: "
            f"{100.0 * recognized_given_display:.1f}%",
            file=sys.stderr,
        )

        if args.raw_output is not None:
            core.write_raw_csv(
                args.raw_output,
                decoded_raw,
            )

        if recognized_count == 0:
            raise RuntimeError(
                "The video contains no recognized values."
            )

        filled = (
            core.fill_unrecognized(
                decoded_raw
            )
        )

        smoothed = (
            core.centered_mode(
                filled,
                mode_window,
            )
        )

        runs = (
            core.make_runs(
                smoothed
            )
        )

        minimum_samples = max(
            1,
            int(
                math.ceil(
                    args.min_interval
                    * sample_fps
                )
            ),
        )

        runs = (
            core.merge_short_runs(
                runs,
                minimum_samples,
                preserve_edges=(
                    profile.preserve_edge_runs
                ),
            )
        )

        if (
            profile.choose_run_value_at_center
        ):
            runs = (
                core.choose_run_values_from_centers(
                    runs,
                    filled,
                )
            )

        intervals = (
            core.run_boundaries(
                runs,
                sample_fps,
                info.duration,
            )
        )

        core.write_interval_csv(
            args.output,
            intervals,
        )

        if args.debug_dir is not None:
            save_debug_screenshots_roi(
                args.video,
                info,
                profile,
                sample_fps,
                args.processing_width,
                intervals,
                args.debug_dir,
                roi,
                args.contrast,
            )

        summary_path = (
            args.summary
        )

        if summary_path is None:
            summary_path = (
                args.output.with_suffix(
                    args.output.suffix
                    + ".summary.json"
                )
            )

        summary = (
            core.calculate_summary(
                intervals,
                args.video,
                profile,
                info.duration,
                display_found_fraction,
                recognized_fraction,
            )
        )

        summary[
            "recognized_given_display_fraction"
        ] = recognized_given_display

        summary[
            "roi"
        ] = (
            None
            if roi is None
            else list(roi)
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.write_text(
            __import__(
                "json"
            ).dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        core.print_summary(
            summary
        )

        print()

        print(
            f"Intervals written to:    {args.output}"
        )

        print(
            f"Summary written to:      {summary_path}"
        )

        if (
            args.raw_output
            is not None
        ):
            print(
                f"Raw samples written to:  {args.raw_output}"
            )

        if (
            args.debug_dir
            is not None
        ):
            print(
                f"Debug screenshots:       {args.debug_dir}"
            )

        if roi is not None:
            print()
            print(
                "Reusable ROI:"
            )
            print(
                f"  --roi {format_roi(roi)}"
            )

        return 0

    except (
        RuntimeError,
        ValueError,
        OSError,
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
