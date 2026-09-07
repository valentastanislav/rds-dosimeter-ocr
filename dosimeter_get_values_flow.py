#!/usr/bin/env python3
"""
Optical-flow stabilization for hand-held RDS-200 video.

Required files in the same directory:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

Processing:

    video
      -> establish one reference display box
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
from typing import Sequence

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


# ======================================================================
# Extra command-line options
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
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

    return parser.parse_known_args(
        argv
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

    # Exclude most of the changing numeric field.
    inner_x1 = int(
        round(
            x + 0.23 * width
        )
    )

    inner_x2 = int(
        round(
            x + 0.78 * width
        )
    )

    inner_y1 = int(
        round(
            y + 0.40 * height
        )
    )

    inner_y2 = int(
        round(
            y + 0.86 * height
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

        # Use the normal detector only while establishing
        # the physical reference display box.
        if (
            actual_reference_index is None
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
    Build a replacement for core.save_debug_screenshots().

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
    ) -> None:

        # These are intentionally unused.
        del video
        del info
        del processing_width
        del contrast_mode

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
) -> int:

    selected_argv = (
        sys.argv[1:]
        if argv is None
        else list(argv)
    )

    wrapper_args, remaining = (
        parse_wrapper_args(
            selected_argv
        )
    )

    # ----------------------------------------------------------
    # Parse normal fixed-grid / ROI arguments
    # ----------------------------------------------------------

    (
        fixed_extra,
        roi_remaining,
    ) = (
        fixed_app.parse_extra_args(
            remaining
        )
    )

    args = (
        fixed_app.parse_roi_args(
            roi_remaining
        )
    )

    if args.roi is None:

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

        original_decimal_decoder = (
            core.decode_decimal_places_sequence
        )

        def capturing_decimal_decoder(
            decimal_scores,
            integer_values,
            in_profile,
            switch_penalty,
        ):

            result = (
                original_decimal_decoder(
                    decimal_scores,
                    integer_values,
                    in_profile,
                    switch_penalty,
                )
            )

            captured_decimal_sequence.clear()

            captured_decimal_sequence.extend(
                int(value)
                for value in result
            )

            return result

        core.decode_decimal_places_sequence = (
            capturing_decimal_decoder
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
        # Replace fixedgrid's display finder
        # ----------------------------------------------------------

        original_factory = (
            fixed_app.make_rectified_finder
        )

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

        fixed_app.make_rectified_finder = (
            make_flow_finder
        )

        # ------------------------------------------------------
        # Replace debug writer.
        #
        # It will read ONLY display_cache.
        # ----------------------------------------------------------

        original_debug_writer = (
            core.save_debug_screenshots
        )

        cached_debug_writer = (
            make_cached_debug_writer(
                display_cache,
                transforms,
                captured_decimal_sequence,
                decimal_override,
            )
        )

        core.save_debug_screenshots = (
            cached_debug_writer
        )

        # ------------------------------------------------------
        # Run existing fixed-grid decoder
        # ----------------------------------------------------------

        try:
            result = (
                fixed_app.main(
                    remaining,
                    decode_samples=(
                        decode_samples
                    ),
                )
            )

        finally:
            fixed_app.make_rectified_finder = (
                original_factory
            )

            core.save_debug_screenshots = (
                original_debug_writer
            )

            core.decode_decimal_places_sequence = (
                original_decimal_decoder
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
