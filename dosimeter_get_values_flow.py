#!/usr/bin/env python3
"""
Optical-flow stabilization for hand-held RADOS dosimeter video.

Required files in the same directory:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

Processing:

    video
      -> establish one automatic or manual reference display box
      -> sequential Lucas-Kanade optical flow
      -> RANSAC similarity transform
      -> stabilize complete processing frame
      -> use ONE fixed display crop
      -> fixed perspective correction (--quad)
      -> fixed digit grid (--grid)
      -> decoder

IMPORTANT DEBUG BEHAVIOUR

The stabilized + rectified display image used during the MAIN decoding
pass is cached for every sampled frame.

Debug JPGs are generated directly from that cache.

Therefore:

    pixels visible in debug JPG
        ==
    pixels actually passed to the digit decoder

There is no second geometrical/tracking pass for debug output.

The selected decimal-point position is also captured from the actual
decoder.  If --decimal-places is fixed, only that decimal point is
highlighted.  If decimal-place auto detection is used, the Viterbi-selected
state for the corresponding raw frame is highlighted.

Example:

    python3 dosimeter_get_values_flow.py IMG_1151edited.mov out_flow.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad 0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493 \
        --grid 0.302231,0.438202,0.726166,0.786517 \
        --track-time 6.8 \
        --decimal-places 2 \
        --contrast auto \
        --raw-output raw_flow.csv \
        --debug-dir debug_flow
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


# ======================================================================
# Imports from existing scripts
# ======================================================================


try:
    import dosimeter_get_values as core
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values.py."
    ) from exc


try:
    import dosimeter_get_values_roi as roi_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_roi.py."
    ) from exc


try:
    import dosimeter_get_values_rectified as rect_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_rectified.py."
    ) from exc


try:
    import dosimeter_get_values_fixedgrid as fixed_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_fixedgrid.py."
    ) from exc


# Save the genuine ROI finder before anything replaces it.
BASE_ROI_FINDER = (
    roi_app.find_display_crop_roi
)

ReferenceBox = tuple[
    float,
    float,
    float,
    float,
]
ReferenceBoxSelector = Callable[
    [np.ndarray],
    ReferenceBox,
]
FLOW_CLI_USAGE = (
    "python3 dosimeter_get_values_flow.py "
    "<video file> <output file> [options]"
)

FLOW_CLI_HELP = """\
usage: python3 dosimeter_get_values_flow.py <video file> <output file> [options]

Primary optical-flow OCR pipeline for RADOS RDS-30 and RDS-200 videos.

Quick start — RDS-30 beta
  python3 dosimeter_get_values_flow.py VIDEO.MOV intervals.csv \\
    --profile rds30 \\
    --track-time 20.0 \\
    --select-reference-box \\
    --select-quad \\
    --raw-output raw.csv

Quick start — RDS-200 first run
  python3 dosimeter_get_values_flow.py VIDEO.MOV intervals.csv \\
    --profile rds200 \\
    --track-time 20.0 \\
    --select-reference-box \\
    --select-quad \\
    --select-grid \\
    --raw-output raw.csv

Manual geometry
  --track-time SECONDS
      Reference frame used to define geometry.

  --select-reference-box
      Interactively select a loose box around the complete physical display.

  --reference-box x1,y1,x2,y2
      Reuse a previously selected reference box.

  --select-quad
      Interactively select the four physical LCD corners.

  --quad x1,y1,x2,y2,x3,y3,x4,y4
      Reuse a previously selected perspective quad.

  --select-grid
      Interactively select the RDS-200 digit grid.

  --grid x1,y1,x2,y2
      Reuse a previously selected RDS-200 digit grid.

  Geometry is video-specific. Do not reuse reference-box, quad, or RDS-200
  grid values for a different video without verifying them.
  RDS-30 uses profile-defined digit geometry and does not use --grid.

RDS-30 beta decoder
  With --profile rds30, the validated beta configuration is automatic:
    decoder strategy:        rds30-joint-spatial
    geometry emission:       glyph-independent
    confidence threshold:    0.358
    minimum nine margin:     0.400

  Explicit command-line values override these profile defaults.
  Use --decoder-strategy default to request the historical decoder.

  --decoder-strategy {default,rds30-joint-spatial}
  --joint-spatial-geometry-emission {glyph-best,glyph-independent}
  --joint-spatial-confidence VALUE
  --joint-spatial-min-nine-margin VALUE
  --joint-spatial-diagnostics-dir DIR
  --joint-spatial-preview-frame FRAME_INDEX

RDS-200 decoder
  --decoder-strategy default
      Historical RDS-200 decoder.

  --rds200-pattern-refinement
      Enable the opt-in RDS-200 binary-pattern refinement. It may strengthen
      confidence for exact pattern matches and recover uniformly active digit 8.
      Disabled by default.

General decoding
  --profile {rds200,rds30}
  --decimal-places {auto,0,1,2,3}
  --contrast {auto,none,clahe}
  --min-confidence VALUE
  --filter-window N
  --mode-window N
  --sample-fps VALUE
  --processing-width PIXELS

Output
  --raw-output FILE
      Write every decoded sample before interval merging.

  --summary FILE
      Write summary JSON.

  --debug-dir DIR
      Write interval debug images.

Tracking
  --flow-max-translation PIXELS
  --flow-max-rotation DEGREES
  --flow-min-scale VALUE
  --flow-max-scale VALUE
  --flow-min-inliers N
  --flow-redetect-every N

  -h, --help
      Show this help and exit.

See README.md for setup, detailed geometry instructions, and beta-test guidance.
"""


# ======================================================================
# Manual reference-display box
# ======================================================================


def parse_reference_box(
    text: str,
) -> ReferenceBox:
    try:
        values = tuple(
            float(value.strip())
            for value in text.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--reference-box must contain four numbers: x1,y1,x2,y2"
        ) from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "--reference-box must contain four numbers: x1,y1,x2,y2"
        )

    x1, y1, x2, y2 = values

    if not (
        0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "--reference-box coordinates must satisfy "
            "0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )

    return values


def format_reference_box(
    box: ReferenceBox,
) -> str:
    return ",".join(
        f"{value:.6f}"
        for value in box
    )


def select_reference_box(
    frame: np.ndarray,
) -> ReferenceBox:
    height, width = frame.shape[:2]

    window_name = (
        "Select COMPLETE physical LCD/display region - "
        "ENTER/SPACE accept, C/ESC cancel"
    )

    try:
        with rect_app.selection_cleanup(window_name):
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

            x, y, box_width, box_height = cv2.selectROI(
                window_name,
                frame,
                showCrosshair=True,
                fromCenter=False,
            )

    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV could not open the reference-box selection window. "
            "Use --reference-box x1,y1,x2,y2 instead."
        ) from exc
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "Reference-box selection cancelled."
        ) from exc

    if box_width <= 0 or box_height <= 0:
        raise RuntimeError(
            "Reference-box selection was cancelled."
        )

    return (
        x / width,
        y / height,
        (x + box_width) / width,
        (y + box_height) / height,
    )


# ======================================================================
# Extra command-line options
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        usage=FLOW_CLI_USAGE,
    )

    parser.add_argument(
        "--track-time",
        type=float,
        required=True,
        help=(
            "reference time; normally use the time at which "
            "the fixed digit grid was selected"
        ),
    )

    parser.add_argument(
        "--reference-box",
        type=parse_reference_box,
        default=None,
        help=(
            "complete physical display region at --track-time, "
            "as normalized oriented-frame x1,y1,x2,y2"
        ),
    )

    parser.add_argument(
        "--select-reference-box",
        action="store_true",
        help=(
            "interactively select the complete display region "
            "on the --track-time frame"
        ),
    )

    parser.add_argument(
        "--select-quad",
        action="store_true",
        help=(
            "select the perspective quad inside the manual "
            "reference display crop"
        ),
    )

    parser.add_argument(
        "--flow-max-translation",
        type=float,
        default=18.0,
        help=(
            "maximum frame-to-frame translation at processing "
            "resolution (default: 18 px)"
        ),
    )

    parser.add_argument(
        "--flow-max-rotation",
        type=float,
        default=2.5,
        help=(
            "maximum frame-to-frame rotation in degrees "
            "(default: 2.5)"
        ),
    )

    parser.add_argument(
        "--flow-min-scale",
        type=float,
        default=0.975,
        help=(
            "minimum allowed frame-to-frame scale "
            "(default: 0.975)"
        ),
    )

    parser.add_argument(
        "--flow-max-scale",
        type=float,
        default=1.025,
        help=(
            "maximum allowed frame-to-frame scale "
            "(default: 1.025)"
        ),
    )

    parser.add_argument(
        "--flow-min-inliers",
        type=int,
        default=8,
        help=(
            "minimum RANSAC inliers for accepting motion "
            "(default: 8)"
        ),
    )

    parser.add_argument(
        "--flow-redetect-every",
        type=int,
        default=10,
        help=(
            "redetect optical-flow features after this many "
            "frames (default: 10)"
        ),
    )

    parser.add_argument(
        "--decoder-strategy",
        choices=(
            "default",
            "rds30-joint-spatial",
        ),
        default=None,
        help=(
            "decoder strategy applied to cached main-pass displays "
            "(default: profile-dependent)"
        ),
    )

    parser.add_argument(
        "--rds200-pattern-refinement",
        action="store_true",
        help=(
            "enable the experimental RDS-200 pattern-decoder "
            "confidence/8 refinement"
        ),
    )

    parser.add_argument(
        "--joint-spatial-confidence",
        type=float,
        default=None,
        help=(
            "accepted-frame confidence threshold for the opt-in "
            "RDS-30 joint-spatial decoder"
        ),
    )

    parser.add_argument(
        "--joint-spatial-min-nine-margin",
        type=float,
        default=None,
        help=(
            "reject a joint-spatial sample when any decoded 9 has a "
            "smaller glyph margin; disabled by default"
        ),
    )

    parser.add_argument(
        "--joint-spatial-geometry-emission",
        choices=(
            "glyph-best",
            "glyph-independent",
        ),
        default=None,
        help=(
            "geometry Viterbi emission for the RDS-30 joint-spatial "
            "decoder (default: profile-dependent)"
        ),
    )

    parser.add_argument(
        "--joint-spatial-diagnostics-dir",
        type=Path,
        default=None,
        help=(
            "optional geometry and visual diagnostics directory for "
            "the opt-in RDS-30 joint-spatial decoder"
        ),
    )

    parser.add_argument(
        "--joint-spatial-preview-frame",
        type=int,
        action="append",
        default=None,
        metavar="FRAME_INDEX",
        help=(
            "add a sampled-frame index to the joint-spatial diagnostic "
            "preview set; may be repeated"
        ),
    )

    return parser.parse_known_args(
        argv
    )


RDS30_DEFAULT_DECODER_STRATEGY = "rds30-joint-spatial"
RDS30_DEFAULT_GEOMETRY_EMISSION = "glyph-independent"
RDS30_DEFAULT_CONFIDENCE = 0.358
RDS30_DEFAULT_MIN_NINE_MARGIN = 0.400


def apply_profile_decoder_defaults(
    wrapper_args: argparse.Namespace,
    profile: core.Profile,
) -> None:
    """Resolve profile-specific decoder defaults without overriding CLI values."""

    if wrapper_args.decoder_strategy is None:
        wrapper_args.decoder_strategy = (
            RDS30_DEFAULT_DECODER_STRATEGY
            if profile.name == "rds30"
            else "default"
        )

    if wrapper_args.joint_spatial_geometry_emission is None:
        wrapper_args.joint_spatial_geometry_emission = (
            RDS30_DEFAULT_GEOMETRY_EMISSION
            if profile.name == "rds30"
            else "glyph-best"
        )

    if (
        profile.name == "rds30"
        and wrapper_args.decoder_strategy == RDS30_DEFAULT_DECODER_STRATEGY
    ):
        if wrapper_args.joint_spatial_confidence is None:
            wrapper_args.joint_spatial_confidence = RDS30_DEFAULT_CONFIDENCE

        if wrapper_args.joint_spatial_min_nine_margin is None:
            wrapper_args.joint_spatial_min_nine_margin = (
                RDS30_DEFAULT_MIN_NINE_MARGIN
            )


# ======================================================================
# Transformation helpers
# ======================================================================


def identity_matrix() -> np.ndarray:

    return np.eye(
        3,
        dtype=np.float64,
    )


def affine_to_homogeneous(
    affine: np.ndarray,
) -> np.ndarray:

    result = identity_matrix()

    result[
        0:2,
        :
    ] = affine

    return result


def decompose_similarity(
    matrix: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    a = float(
        matrix[0, 0]
    )

    c = float(
        matrix[1, 0]
    )

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )

    scale = math.hypot(
        a,
        c,
    )

    angle = math.degrees(
        math.atan2(
            c,
            a,
        )
    )

    return (
        tx,
        ty,
        angle,
        scale,
    )


# ======================================================================
# Feature mask
# ======================================================================


def make_feature_mask(
    frame_shape: tuple[int, ...],
    box: tuple[int, int, int, int],
    profile: core.Profile,
) -> np.ndarray:
    """
    Track static features on and around the display.

    The changing large digits are excluded.
    """

    frame_height = (
        frame_shape[0]
    )

    frame_width = (
        frame_shape[1]
    )

    mask = np.zeros(
        (
            frame_height,
            frame_width,
        ),
        dtype=np.uint8,
    )

    x, y, width, height = (
        box
    )

    pad_x = int(
        round(
            0.08 * width
        )
    )

    pad_y = int(
        round(
            0.08 * height
        )
    )

    outer_x1 = max(
        0,
        x - pad_x,
    )

    outer_y1 = max(
        0,
        y - pad_y,
    )

    outer_x2 = min(
        frame_width,
        x + width + pad_x,
    )

    outer_y2 = min(
        frame_height,
        y + height + pad_y,
    )

    cv2.rectangle(
        mask,
        (
            outer_x1,
            outer_y1,
        ),
        (
            outer_x2 - 1,
            outer_y2 - 1,
        ),
        255,
        -1,
    )

    # Exclude the profile-specific changing numeric field.
    (
        exclusion_x1,
        exclusion_y1,
        exclusion_x2,
        exclusion_y2,
    ) = profile.flow_feature_exclusion_box

    inner_x1 = int(
        round(
            x + exclusion_x1 * width
        )
    )

    inner_x2 = int(
        round(
            x + exclusion_x2 * width
        )
    )

    inner_y1 = int(
        round(
            y + exclusion_y1 * height
        )
    )

    inner_y2 = int(
        round(
            y + exclusion_y2 * height
        )
    )

    cv2.rectangle(
        mask,
        (
            inner_x1,
            inner_y1,
        ),
        (
            inner_x2,
            inner_y2,
        ),
        0,
        -1,
    )

    return mask


# ======================================================================
# Feature detection
# ======================================================================


def detect_features(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=140,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )

    if points is None:

        return None

    return points.astype(
        np.float32
    )


# ======================================================================
# Warp reference mask into current-frame coordinates
# ======================================================================


def warp_mask(
    reference_mask: np.ndarray,
    cumulative: np.ndarray,
) -> np.ndarray:

    height, width = (
        reference_mask.shape
    )

    return cv2.warpAffine(
        reference_mask,
        cumulative[
            0:2
        ].astype(
            np.float32
        ),
        (
            width,
            height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ======================================================================
# One optical-flow step
# ======================================================================


def estimate_step(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    maximum_translation: float,
    maximum_rotation: float,
    minimum_scale: float,
    maximum_scale: float,
    minimum_inliers: int,
) -> tuple[
    bool,
    np.ndarray,
    np.ndarray | None,
    int,
    int,
]:

    current_points, status, error = (
        cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )
    )

    if (
        current_points is None
        or status is None
    ):

        return (
            False,
            identity_matrix(),
            None,
            0,
            0,
        )

    status = (
        status.reshape(-1)
        == 1
    )

    previous_flat = (
        previous_points.reshape(
            -1,
            2,
        )
    )

    current_flat = (
        current_points.reshape(
            -1,
            2,
        )
    )

    good_previous = (
        previous_flat[
            status
        ]
    )

    good_current = (
        current_flat[
            status
        ]
    )

    # Reject obviously poor LK matches.
    if error is not None:

        good_error = (
            error.reshape(-1)[
                status
            ]
        )

        keep = (
            good_error < 35.0
        )

        good_previous = (
            good_previous[
                keep
            ]
        )

        good_current = (
            good_current[
                keep
            ]
        )

    number_tracked = len(
        good_previous
    )

    if (
        number_tracked
        < minimum_inliers
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            0,
        )

    affine, inlier_mask = (
        cv2.estimateAffinePartial2D(
            good_previous,
            good_current,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
    )

    if (
        affine is None
        or inlier_mask is None
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            0,
        )

    inliers = (
        inlier_mask.reshape(-1)
        == 1
    )

    number_inliers = int(
        np.sum(
            inliers
        )
    )

    if (
        number_inliers
        < minimum_inliers
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            number_inliers,
        )

    inlier_fraction = (
        number_inliers
        / max(
            number_tracked,
            1,
        )
    )

    if (
        inlier_fraction < 0.45
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            number_inliers,
        )

    tx = float(
        affine[0, 2]
    )

    ty = float(
        affine[1, 2]
    )

    translation = math.hypot(
        tx,
        ty,
    )

    a = float(
        affine[0, 0]
    )

    c = float(
        affine[1, 0]
    )

    scale = math.hypot(
        a,
        c,
    )

    angle = math.degrees(
        math.atan2(
            c,
            a,
        )
    )

    if (
        translation
        > maximum_translation
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            number_inliers,
        )

    if (
        abs(angle)
        > maximum_rotation
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            number_inliers,
        )

    if not (
        minimum_scale
        <= scale
        <= maximum_scale
    ):

        return (
            False,
            identity_matrix(),
            None,
            number_tracked,
            number_inliers,
        )

    new_points = (
        good_current[
            inliers
        ]
        .reshape(
            -1,
            1,
            2,
        )
        .astype(
            np.float32
        )
    )

    return (
        True,
        affine_to_homogeneous(
            affine
        ),
        new_points,
        number_tracked,
        number_inliers,
    )


# ======================================================================
# Precompute motion for the whole sampled video
# ======================================================================


def precompute_motion(
    video: Path,
    info: core.VideoInfo,
    profile: core.Profile,
    roi,
    processing_width: int,
    sample_fps: float,
    reference_time: float,
    maximum_translation: float,
    maximum_rotation: float,
    minimum_scale: float,
    maximum_scale: float,
    minimum_inliers: int,
    redetect_every: int,
    manual_reference_box: ReferenceBox | None = None,
    reference_box_selector: ReferenceBoxSelector | None = None,
) -> tuple[
    list[np.ndarray],
    tuple[int, int, int, int],
    tuple[int, int],
    dict[str, float | int],
]:

    requested_reference_index = int(
        round(
            reference_time
            * sample_fps
        )
    )

    gray_frames: list[
        np.ndarray
    ] = []

    reference_box = None
    actual_reference_index = None

    previous_box = None

    manual_reference = (
        manual_reference_box is not None
        or reference_box_selector is not None
    )

    print(
        "Pre-reading sampled frames and locating reference display...",
        file=sys.stderr,
    )

    for index, frame in enumerate(
        core.iter_ffmpeg_frames(
            video,
            info,
            sample_fps,
            processing_width,
        )
    ):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray_frames.append(
            gray
        )

        if (
            manual_reference
            and index == requested_reference_index
        ):
            selected_reference_box = (
                reference_box_selector(
                    frame
                )
                if reference_box_selector is not None
                else manual_reference_box
            )

            if selected_reference_box is None:
                raise RuntimeError(
                    "No manual reference display box was supplied."
                )

            reference_box = (
                roi_app.roi_to_pixels(
                    selected_reference_box,
                    gray.shape[1],
                    gray.shape[0],
                )
            )

            actual_reference_index = index

        # Preserve automatic acquisition when no manual box is used.
        elif (
            not manual_reference
            and actual_reference_index is None
            and index
            <= requested_reference_index + 10
        ):

            display, box = (
                BASE_ROI_FINDER(
                    frame,
                    profile,
                    previous_box,
                    roi,
                )
            )

            if display is not None:

                previous_box = (
                    box
                )

                if (
                    index
                    >= requested_reference_index
                ):

                    reference_box = (
                        box
                    )

                    actual_reference_index = (
                        index
                    )

    if not gray_frames:

        raise RuntimeError(
            "No frames were read."
        )

    if (
        reference_box is None
        or actual_reference_index is None
    ):

        raise RuntimeError(
            "Could not establish display box near reference time."
        )

    frame_count = len(
        gray_frames
    )

    reference_gray = (
        gray_frames[
            actual_reference_index
        ]
    )

    frame_height, frame_width = (
        reference_gray.shape
    )

    reference_mask = (
        make_feature_mask(
            reference_gray.shape,
            reference_box,
            profile,
        )
    )

    reference_points = (
        detect_features(
            reference_gray,
            reference_mask,
        )
    )

    if (
        reference_points is None
        or len(
            reference_points
        ) < minimum_inliers
    ):

        raise RuntimeError(
            "Not enough trackable features around the reference display."
        )

    print(
        (
            f"Reference frame: "
            f"{actual_reference_index} "
            f"(t={actual_reference_index / sample_fps:.3f} s)"
        ),
        file=sys.stderr,
    )

    print(
        (
            f"Reference display box: "
            f"x={reference_box[0]}, "
            f"y={reference_box[1]}, "
            f"w={reference_box[2]}, "
            f"h={reference_box[3]}"
        ),
        file=sys.stderr,
    )

    print(
        (
            f"Initial tracking features: "
            f"{len(reference_points)}"
        ),
        file=sys.stderr,
    )

    transforms = [
        identity_matrix()
        for _ in range(
            frame_count
        )
    ]

    stats: dict[
        str,
        float | int,
    ] = {
        "accepted": 0,
        "rejected": 0,
        "tracked_points_sum": 0,
        "inliers_sum": 0,
        "steps": 0,
        "redetections": 0,
    }

    # ----------------------------------------------------------
    # Track one temporal direction from the reference frame.
    # ----------------------------------------------------------

    def track_sequence(
        indices,
    ) -> None:

        previous_gray = (
            reference_gray
        )

        previous_points = (
            reference_points.copy()
        )

        cumulative = (
            identity_matrix()
        )

        frames_since_redetect = 0

        for current_index in indices:

            current_gray = (
                gray_frames[
                    current_index
                ]
            )

            (
                accepted,
                delta,
                tracked_points,
                number_tracked,
                number_inliers,
            ) = estimate_step(
                previous_gray,
                current_gray,
                previous_points,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            )

            stats[
                "steps"
            ] += 1

            stats[
                "tracked_points_sum"
            ] += number_tracked

            stats[
                "inliers_sum"
            ] += number_inliers

            if accepted:

                # delta maps previous -> current.
                # cumulative maps reference -> previous.
                cumulative = (
                    delta
                    @ cumulative
                )

                transforms[
                    current_index
                ] = (
                    cumulative.copy()
                )

                previous_points = (
                    tracked_points
                )

                stats[
                    "accepted"
                ] += 1

            else:

                # Conservative fallback:
                # assume zero additional motion for this one step.
                transforms[
                    current_index
                ] = (
                    cumulative.copy()
                )

                stats[
                    "rejected"
                ] += 1

            frames_since_redetect += 1

            need_redetect = (
                previous_points is None
                or len(
                    previous_points
                ) < 25
                or frames_since_redetect
                >= redetect_every
            )

            if need_redetect:

                current_mask = (
                    warp_mask(
                        reference_mask,
                        cumulative,
                    )
                )

                new_points = (
                    detect_features(
                        current_gray,
                        current_mask,
                    )
                )

                if (
                    new_points is not None
                    and len(
                        new_points
                    ) >= minimum_inliers
                ):

                    previous_points = (
                        new_points
                    )

                    frames_since_redetect = 0

                    stats[
                        "redetections"
                    ] += 1

            previous_gray = (
                current_gray
            )

    # Forward.
    track_sequence(
        range(
            actual_reference_index + 1,
            frame_count,
        )
    )

    # Backward.
    track_sequence(
        range(
            actual_reference_index - 1,
            -1,
            -1,
        )
    )

    return (
        transforms,
        reference_box,
        (
            frame_width,
            frame_height,
        ),
        stats,
    )


# ======================================================================
# Crop the reference display from a stabilized frame
# ======================================================================


def crop_reference_display(
    stabilized_frame: np.ndarray,
    reference_box: tuple[int, int, int, int],
    profile: core.Profile,
) -> np.ndarray | None:

    x, y, width, height = (
        reference_box
    )

    frame_height, frame_width = (
        stabilized_frame.shape[:2]
    )

    x1 = max(
        0,
        x,
    )

    y1 = max(
        0,
        y,
    )

    x2 = min(
        frame_width,
        x + width,
    )

    y2 = min(
        frame_height,
        y + height,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    crop = (
        stabilized_frame[
            y1:y2,
            x1:x2,
        ]
    )

    if crop.size == 0:

        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.resize(
        gray,
        (
            profile.canonical_width,
            profile.canonical_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


# ======================================================================
# Debug overlay
# ======================================================================


def draw_decoder_geometry(
    display: np.ndarray,
    profile: core.Profile,
    chosen_decimal_places: int | None,
    decimal_mode: str,
) -> np.ndarray:
    """
    Draw exactly the fixed digit geometry used by the decoder.

    Because fixedgrid disables adaptive y-shift, shift is exactly zero.
    """

    overlay = cv2.cvtColor(
        display,
        cv2.COLOR_GRAY2BGR,
    )

    shift = 0

    # ----------------------------------------------------------
    # Digit boxes + seven-segment polygons
    # ----------------------------------------------------------

    for (
        x1,
        y1,
        x2,
        y2,
    ) in profile.digit_boxes:

        cv2.rectangle(
            overlay,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                180,
                0,
            ),
            1,
        )

        box_width = max(
            1,
            x2 - x1,
        )

        box_height = max(
            1,
            y2 - y1,
        )

        scale_x = (
            box_width
            / 65.0
        )

        scale_y = (
            box_height
            / 130.0
        )

        for name in (
            core.SEGMENT_ORDER
        ):

            local = (
                profile.segment_polygons[
                    name
                ].astype(
                    float
                )
            )

            points = np.empty_like(
                local,
                dtype=np.int32,
            )

            points[
                :,
                0
            ] = np.rint(
                x1
                + local[
                    :,
                    0
                ]
                * scale_x
            ).astype(
                np.int32
            )

            points[
                :,
                1
            ] = np.rint(
                y1
                + (
                    local[
                        :,
                        1
                    ]
                    + shift
                )
                * scale_y
            ).astype(
                np.int32
            )

            cv2.polylines(
                overlay,
                [
                    points
                ],
                True,
                (
                    0,
                    180,
                    0,
                ),
                1,
            )

    # ----------------------------------------------------------
    # Decimal candidates
    #
    # Selected:
    #   red, thick
    #
    # Unselected:
    #   grey, thin
    # ----------------------------------------------------------

    for (
        x1,
        y1,
        x2,
        y2,
        decimal_places,
    ) in profile.decimal_candidates:

        selected = (
            chosen_decimal_places
            is not None
            and decimal_places
            == chosen_decimal_places
        )

        if selected:

            color = (
                0,
                0,
                230,
            )

            thickness = 2

        else:

            color = (
                150,
                150,
                150,
            )

            thickness = 1

        cv2.rectangle(
            overlay,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            thickness,
        )

        if selected:

            if decimal_mode == "fixed":

                label = (
                    f".{decimal_places} FIXED"
                )

            else:

                label = (
                    f".{decimal_places} AUTO"
                )

        else:

            label = (
                f".{decimal_places}"
            )

        cv2.putText(
            overlay,
            label,
            (
                x1 - 5,
                max(
                    15,
                    y1 - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay


# ======================================================================
# Cached debug writer
# ======================================================================


def make_cached_debug_writer(
    display_cache: dict[int, np.ndarray],
    transforms: Sequence[np.ndarray],
    decimal_sequence: list[int],
    decimal_override: int | None,
):
    """
    Build a cached writer compatible with core.save_debug_screenshots().

    It NEVER reopens or reprocesses video frames.

    The display image comes directly from display_cache, which was populated
    during the actual decoding pass.
    """

    def save_cached_debug_screenshots(
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
        contrast_mode: str = "auto",
        display_finder: core.DisplayFinder | None = None,
    ) -> None:

        # These are intentionally unused.
        del video
        del info
        del processing_width
        del contrast_mode
        del display_finder

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            output_dir
            / "index.csv"
        )

        with index_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.writer(
                handle
            )

            writer.writerow(
                [
                    "interval",
                    "start_s",
                    "end_s",
                    "center_s",
                    "frame_index",
                    "value",
                    "confidence",
                    "decimal_places",
                    "debug_source",
                    "image",
                ]
            )

            for (
                interval_number,
                (
                    start,
                    end,
                    run,
                ),
            ) in enumerate(
                intervals,
                start=1,
            ):

                requested_center = (
                    0.5
                    * (
                        start
                        + end
                    )
                )

                frame_index = int(
                    round(
                        requested_center
                        * sample_fps
                    )
                )

                frame_index = max(
                    0,
                    min(
                        frame_index,
                        len(
                            transforms
                        )
                        - 1,
                    ),
                )

                center_time = (
                    frame_index
                    / sample_fps
                )

                value_text = (
                    f"{run.value:.6g}"
                )

                safe_value = (
                    value_text
                    .replace(
                        "-",
                        "m",
                    )
                    .replace(
                        ".",
                        "p",
                    )
                )

                filename = (
                    f"interval_{interval_number:03d}_"
                    f"t{center_time:08.3f}_"
                    f"value_{safe_value}.jpg"
                )

                display = (
                    display_cache.get(
                        frame_index
                    )
                )

                # --------------------------------------------------
                # Decimal state actually used by decoder
                # --------------------------------------------------

                if (
                    decimal_override
                    is not None
                ):

                    chosen_decimal_places = (
                        decimal_override
                    )

                    decimal_mode = (
                        "fixed"
                    )

                elif (
                    0
                    <= frame_index
                    < len(
                        decimal_sequence
                    )
                ):

                    chosen_decimal_places = int(
                        decimal_sequence[
                            frame_index
                        ]
                    )

                    decimal_mode = (
                        "auto"
                    )

                else:

                    chosen_decimal_places = (
                        None
                    )

                    decimal_mode = (
                        "unknown"
                    )

                # --------------------------------------------------
                # Missing cache entry should be explicit.
                #
                # NEVER silently reconstruct geometry here.
                # --------------------------------------------------

                if display is None:

                    canvas = np.full(
                        (
                            150,
                            760,
                            3,
                        ),
                        255,
                        dtype=np.uint8,
                    )

                    cv2.putText(
                        canvas,
                        (
                            f"interval {interval_number}: "
                            f"t={center_time:.3f} s "
                            f"value={value_text}"
                        ),
                        (
                            15,
                            45,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (
                            0,
                            0,
                            0,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        canvas,
                        (
                            "CACHED MAIN-PASS DISPLAY MISSING"
                        ),
                        (
                            15,
                            95,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (
                            0,
                            0,
                            220,
                        ),
                        2,
                        cv2.LINE_AA,
                    )

                else:

                    overlay = (
                        draw_decoder_geometry(
                            display,
                            profile,
                            chosen_decimal_places,
                            decimal_mode,
                        )
                    )

                    matrix = (
                        transforms[
                            frame_index
                        ]
                    )

                    (
                        tx,
                        ty,
                        angle,
                        scale,
                    ) = decompose_similarity(
                        matrix
                    )

                    header_height = (
                        72
                    )

                    canvas = np.full(
                        (
                            overlay.shape[0]
                            + header_height,
                            overlay.shape[1],
                            3,
                        ),
                        255,
                        dtype=np.uint8,
                    )

                    canvas[
                        header_height:
                    ] = (
                        overlay
                    )

                    # First header line.
                    cv2.putText(
                        canvas,
                        (
                            f"interval {interval_number}: "
                            f"t={center_time:.3f}s, "
                            f"value={value_text}, "
                            f"confidence={run.confidence:.3f}, "
                            f"decimal={chosen_decimal_places}"
                        ),
                        (
                            8,
                            27,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (
                            0,
                            0,
                            0,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                    # Second header line proves which transform/cache
                    # entry was actually used.
                    cv2.putText(
                        canvas,
                        (
                            f"CACHED MAIN PASS frame={frame_index} | "
                            f"flow dx={tx:+.2f}px "
                            f"dy={ty:+.2f}px "
                            f"rot={angle:+.3f}deg "
                            f"scale={scale:.5f}"
                        ),
                        (
                            8,
                            54,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (
                            0,
                            80,
                            0,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                image_path = (
                    output_dir
                    / filename
                )

                if not cv2.imwrite(
                    str(
                        image_path
                    ),
                    canvas,
                ):

                    raise RuntimeError(
                        f"Could not write debug image: {image_path}"
                    )

                writer.writerow(
                    [
                        interval_number,
                        f"{start:.3f}",
                        f"{end:.3f}",
                        f"{center_time:.3f}",
                        frame_index,
                        value_text,
                        f"{run.confidence:.4f}",
                        (
                            ""
                            if chosen_decimal_places
                            is None
                            else chosen_decimal_places
                        ),
                        "cached_main_pass",
                        filename,
                    ]
                )

    return (
        save_cached_debug_screenshots
    )


# ======================================================================
# Main
# ======================================================================


def main(
    argv: Sequence[str] | None = None,
    decode_samples: roi_app.DecodeSamples | None = None,
    profile_override: core.Profile | None = None,
) -> int:

    selected_argv = (
        sys.argv[1:]
        if argv is None
        else list(argv)
    )

    if "-h" in selected_argv or "--help" in selected_argv:
        print(FLOW_CLI_HELP)
        return 0

    wrapper_args, remaining = (
        parse_wrapper_args(
            selected_argv
        )
    )

    # ----------------------------------------------------------
    # Parse normal fixed-grid / ROI arguments
    # ----------------------------------------------------------

    if (
        wrapper_args.select_reference_box
        and wrapper_args.reference_box is not None
    ):
        print(
            "Error: use either --select-reference-box or --reference-box.",
            file=sys.stderr,
        )

        return 1

    manual_reference_requested = (
        wrapper_args.select_reference_box
        or wrapper_args.reference_box is not None
    )

    if (
        wrapper_args.select_quad
        and not manual_reference_requested
    ):
        print(
            "Error: --select-quad requires --select-reference-box "
            "or --reference-box.",
            file=sys.stderr,
        )

        return 1

    (
        fixed_extra,
        roi_remaining,
    ) = (
        fixed_app.parse_extra_args(
            remaining,
            require_quad=(
                not wrapper_args.select_quad
            ),
            usage=FLOW_CLI_USAGE,
        )
    )

    if (
        wrapper_args.select_quad
        and fixed_extra.quad is not None
    ):
        print(
            "Error: use either --select-quad or --quad.",
            file=sys.stderr,
        )

        return 1

    args = (
        fixed_app.parse_roi_args(
            roi_remaining,
            usage=FLOW_CLI_USAGE,
        )
    )

    if (
        args.roi is None
        and not manual_reference_requested
    ):

        print(
            "Error: --roi is required.",
            file=sys.stderr,
        )

        return 1

    try:

        profile = (
            core.PROFILES[
                args.profile
            ]
            if profile_override is None
            else profile_override
        )

        apply_profile_decoder_defaults(
            wrapper_args,
            profile,
        )

        if (
            wrapper_args.joint_spatial_confidence is not None
            and not 0.0 <= wrapper_args.joint_spatial_confidence <= 1.0
        ):
            raise RuntimeError(
                "--joint-spatial-confidence must be between 0 and 1."
            )

        if (
            wrapper_args.joint_spatial_min_nine_margin is not None
            and (
                not math.isfinite(wrapper_args.joint_spatial_min_nine_margin)
                or wrapper_args.joint_spatial_min_nine_margin < 0.0
            )
        ):
            raise RuntimeError(
                "--joint-spatial-min-nine-margin must be a finite, "
                "non-negative number."
            )

        if (
            wrapper_args.joint_spatial_min_nine_margin is not None
            and wrapper_args.decoder_strategy != "rds30-joint-spatial"
        ):
            raise RuntimeError(
                "--joint-spatial-min-nine-margin requires "
                "--decoder-strategy rds30-joint-spatial."
            )

        if (
            wrapper_args.decoder_strategy == "rds30-joint-spatial"
            and decode_samples is not None
        ):
            raise RuntimeError(
                "The joint-spatial CLI strategy cannot be combined with "
                "an injected decode_samples callable."
            )

        if (
            wrapper_args.rds200_pattern_refinement
            and profile.name != "rds200"
        ):
            raise RuntimeError(
                "--rds200-pattern-refinement requires --profile rds200."
            )

        if (
            wrapper_args.rds200_pattern_refinement
            and wrapper_args.decoder_strategy != "default"
        ):
            raise RuntimeError(
                "--rds200-pattern-refinement requires "
                "--decoder-strategy default."
            )

        if (
            wrapper_args.rds200_pattern_refinement
            and decode_samples is not None
        ):
            raise RuntimeError(
                "--rds200-pattern-refinement cannot be combined with "
                "an injected decode_samples callable."
            )

        fixed_app.validate_geometry_options(
            profile,
            fixed_extra.select_grid,
            fixed_extra.grid,
        )

        info = (
            core.probe_video(
                args.video
            )
        )

        sample_fps = (
            args.sample_fps
            if args.sample_fps
            is not None
            else profile.default_sample_fps
        )

        track_time = max(
            0.0,
            min(
                wrapper_args.track_time,
                info.duration,
            ),
        )

        resolved_reference_box = (
            wrapper_args.reference_box
        )

        resolved_quad = (
            fixed_extra.quad
        )

        resolved_grid = (
            fixed_extra.grid
        )

        manual_geometry_selection = (
            wrapper_args.select_reference_box
            or wrapper_args.select_quad
            or (
                manual_reference_requested
                and fixed_extra.select_grid
            )
        )

        def initialize_manual_reference(
            frame: np.ndarray,
        ) -> ReferenceBox:
            nonlocal resolved_reference_box
            nonlocal resolved_quad
            nonlocal resolved_grid

            if wrapper_args.select_reference_box:
                resolved_reference_box = (
                    select_reference_box(
                        frame
                    )
                )

            if resolved_reference_box is None:
                raise RuntimeError(
                    "No manual reference display box is available."
                )

            print(
                "Selected reference display box:",
                file=sys.stderr,
            )

            print(
                (
                    "  --reference-box "
                    + format_reference_box(
                        resolved_reference_box
                    )
                ),
                file=sys.stderr,
            )

            if (
                not wrapper_args.select_quad
                and not fixed_extra.select_grid
            ):
                return resolved_reference_box

            reference_box_pixels = (
                roi_app.roi_to_pixels(
                    resolved_reference_box,
                    frame.shape[1],
                    frame.shape[0],
                )
            )

            reference_display = (
                crop_reference_display(
                    frame,
                    reference_box_pixels,
                    profile,
                )
            )

            if reference_display is None:
                raise RuntimeError(
                    "The manual reference display box produced an empty crop."
                )

            if wrapper_args.select_quad:
                resolved_quad = (
                    rect_app.select_quad(
                        reference_display
                    )
                )

                print(
                    "Selected perspective quad:",
                    file=sys.stderr,
                )

                print(
                    (
                        "  --quad "
                        + rect_app.format_quad(
                            resolved_quad
                        )
                    ),
                    file=sys.stderr,
                )

            if resolved_quad is None:
                raise RuntimeError(
                    "No perspective quad is available for grid selection."
                )

            if fixed_extra.select_grid:
                reference_rectified = (
                    rect_app.rectify_display(
                        reference_display,
                        profile,
                        resolved_quad,
                    )
                )

                resolved_grid = (
                    fixed_app.select_digit_grid(
                        reference_rectified,
                        profile,
                    )
                )

                print(
                    "Selected fixed digit grid:",
                    file=sys.stderr,
                )

                print(
                    (
                        "  --grid "
                        + fixed_app.format_grid(
                            resolved_grid
                        )
                    ),
                    file=sys.stderr,
                )

            return resolved_reference_box

        # ------------------------------------------------------
        # Precompute optical-flow geometry
        # ----------------------------------------------------------

        (
            transforms,
            reference_box,
            processing_size,
            tracking_stats,
        ) = precompute_motion(
            args.video,
            info,
            profile,
            args.roi,
            args.processing_width,
            sample_fps,
            track_time,
            wrapper_args.flow_max_translation,
            wrapper_args.flow_max_rotation,
            wrapper_args.flow_min_scale,
            wrapper_args.flow_max_scale,
            wrapper_args.flow_min_inliers,
            wrapper_args.flow_redetect_every,
            manual_reference_box=(
                resolved_reference_box
                if manual_reference_requested
                and not manual_geometry_selection
                else None
            ),
            reference_box_selector=(
                initialize_manual_reference
                if manual_geometry_selection
                else None
            ),
        )

        fixedgrid_argv = remaining

        if manual_geometry_selection:
            if resolved_quad is None:
                raise RuntimeError(
                    "No perspective quad is available."
                )

            if (
                profile.fixedgrid_geometry_mode
                == "manual_grid"
                and resolved_grid is None
            ):
                raise RuntimeError(
                    "No fixed digit grid is available."
                )

            fixedgrid_argv = [
                *roi_remaining,
                "--quad",
                rect_app.format_quad(
                    resolved_quad
                ),
            ]

            if resolved_grid is not None:
                fixedgrid_argv.extend(
                    [
                        "--grid",
                        fixed_app.format_grid(
                            resolved_grid
                        ),
                    ]
                )

            if fixed_extra.decoder_measurement != "profile":
                fixedgrid_argv.extend(
                    [
                        "--decoder-measurement",
                        fixed_extra.decoder_measurement,
                    ]
                )

        (
            frame_width,
            frame_height,
        ) = (
            processing_size
        )

        # ------------------------------------------------------
        # Main-pass display cache.
        #
        # This is the key debug fix.
        # ----------------------------------------------------------

        display_cache: dict[
            int,
            np.ndarray,
        ] = {}

        # ------------------------------------------------------
        # Capture actual automatic decimal-position sequence.
        # ----------------------------------------------------------

        captured_decimal_sequence: list[
            int
        ] = []

        def capture_decimal_sequence(
            decimal_places_sequence: Sequence[int],
        ) -> None:
            captured_decimal_sequence.clear()

            captured_decimal_sequence.extend(
                int(value)
                for value in decimal_places_sequence
            )

        if wrapper_args.decoder_strategy == "rds30-joint-spatial":
            from dosimeter_rds30_joint_spatial import (
                make_joint_spatial_decoder,
            )

            selected_decode_samples = make_joint_spatial_decoder(
                display_cache,
                rejection_threshold=(
                    wrapper_args.joint_spatial_confidence
                ),
                diagnostics_dir=(
                    wrapper_args.joint_spatial_diagnostics_dir
                ),
                geometry_emission_strategy=(
                    wrapper_args.joint_spatial_geometry_emission
                ),
                preview_frames=(
                    wrapper_args.joint_spatial_preview_frame
                ),
                min_nine_margin=(
                    wrapper_args.joint_spatial_min_nine_margin
                ),
            )
        elif wrapper_args.rds200_pattern_refinement:
            def selected_decode_samples(
                samples,
                in_profile,
                filter_window,
                decimal_places_override=None,
                minimum_confidence=0.0,
                decimal_switch_penalty=4.0,
                decimal_sequence_observer=None,
            ):
                return core.decode_samples(
                    samples,
                    in_profile,
                    filter_window,
                    decimal_places_override=decimal_places_override,
                    minimum_confidence=minimum_confidence,
                    decimal_switch_penalty=decimal_switch_penalty,
                    decimal_sequence_observer=decimal_sequence_observer,
                    rds200_pattern_refinement=True,
                )

        def decode_samples_with_decimal_observer(
            samples,
            in_profile,
            filter_window,
            decimal_places_override=None,
            minimum_confidence=0.0,
            decimal_switch_penalty=4.0,
        ):

            return selected_decode_samples(
                samples,
                in_profile,
                filter_window,
                decimal_places_override=(
                    decimal_places_override
                ),
                minimum_confidence=(
                    minimum_confidence
                ),
                decimal_switch_penalty=(
                    decimal_switch_penalty
                ),
                decimal_sequence_observer=(
                    capture_decimal_sequence
                ),
            )

        decimal_override = (
            None
            if args.decimal_places
            == "auto"
            else int(
                args.decimal_places
            )
        )

        # ------------------------------------------------------
        # Build the flow-aware display finder factory
        # ----------------------------------------------------------

        frame_index = -1

        def make_flow_finder(
            quad: np.ndarray,
        ):

            def finder(
                frame,
                in_profile,
                previous_box,
                roi,
            ):

                nonlocal frame_index

                # roi is no longer used here:
                # the physical reference box is already known.
                del roi

                if previous_box is None:

                    frame_index = 0

                else:

                    frame_index += 1

                if (
                    frame_index < 0
                    or frame_index
                    >= len(
                        transforms
                    )
                ):

                    return (
                        None,
                        reference_box,
                    )

                cumulative = (
                    transforms[
                        frame_index
                    ]
                )

                # cumulative maps:
                #
                # reference coordinates -> current frame coordinates.
                #
                # With WARP_INVERSE_MAP, the output is expressed in
                # reference coordinates.
                stabilized = (
                    cv2.warpAffine(
                        frame,
                        cumulative[
                            0:2
                        ].astype(
                            np.float32
                        ),
                        (
                            frame_width,
                            frame_height,
                        ),
                        flags=(
                            cv2.INTER_CUBIC
                            | cv2.WARP_INVERSE_MAP
                        ),
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                )

                display = (
                    crop_reference_display(
                        stabilized,
                        reference_box,
                        in_profile,
                    )
                )

                if display is None:

                    return (
                        None,
                        reference_box,
                    )

                rectified = (
                    rect_app.rectify_display(
                        display,
                        in_profile,
                        quad,
                    )
                )

                # --------------------------------------------------
                # EXACT image passed downstream to digit extraction.
                # --------------------------------------------------

                display_cache[
                    frame_index
                ] = (
                    rectified.copy()
                )

                return (
                    rectified,
                    reference_box,
                )

            return finder

        # ------------------------------------------------------
        # Construct cached debug writer.
        #
        # It will read ONLY display_cache.
        # ----------------------------------------------------------

        cached_debug_writer = (
            make_cached_debug_writer(
                display_cache,
                transforms,
                captured_decimal_sequence,
                decimal_override,
            )
        )

        # ------------------------------------------------------
        # Run existing fixed-grid decoder
        # ----------------------------------------------------------

        result = (
            fixed_app.main(
                fixedgrid_argv,
                decode_samples=(
                    decode_samples_with_decimal_observer
                ),
                base_profile_override=(
                    profile
                ),
                debug_writer=(
                    cached_debug_writer
                ),
                rectified_finder_factory=(
                    make_flow_finder
                ),
            )
        )

        # ------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        print()

        print(
            "Optical-flow tracking diagnostics:"
        )

        print(
            (
                f"  steps               : "
                f"{tracking_stats['steps']}"
            )
        )

        print(
            (
                f"  accepted motion     : "
                f"{tracking_stats['accepted']}"
            )
        )

        print(
            (
                f"  rejected motion     : "
                f"{tracking_stats['rejected']}"
            )
        )

        print(
            (
                f"  point redetections  : "
                f"{tracking_stats['redetections']}"
            )
        )

        if (
            tracking_stats[
                "steps"
            ] > 0
        ):

            print(
                (
                    "  mean tracked pts    : "
                    f"{tracking_stats['tracked_points_sum'] / tracking_stats['steps']:.1f}"
                )
            )

            print(
                (
                    "  mean RANSAC inliers : "
                    f"{tracking_stats['inliers_sum'] / tracking_stats['steps']:.1f}"
                )
            )

        print()

        print(
            "Debug cache diagnostics:"
        )

        print(
            (
                f"  cached displays     : "
                f"{len(display_cache)}"
            )
        )

        if (
            decimal_override
            is not None
        ):

            print(
                (
                    f"  decimal mode        : fixed "
                    f"({decimal_override} places)"
                )
            )

        else:

            print(
                "  decimal mode        : auto"
            )

            print(
                (
                    f"  captured states     : "
                    f"{len(captured_decimal_sequence)}"
                )
            )

        print(
            "  debug source        : cached main-pass displays"
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
