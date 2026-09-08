#!/usr/bin/env python3
"""
Fixed-grid RDS-200 decoder.

Uses:
    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py

The display is first perspective-rectified using the already determined
--quad.  The user then selects ONE fixed rectangle containing all three
large numeric LCD digits.

That rectangle is used to derive the digit geometry for the entire video.
There is no frame-by-frame y-shift or geometry optimizer.

Additionally, each seven-segment bar is measured against its LOCAL
surrounding LCD background rather than against one brightness value
for the whole digit.

First run:

    python3 dosimeter_get_values_fixedgrid.py IMG_1151edited.mov out.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad 0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493 \
        --select-grid \
        --contrast auto \
        --raw-output raw.csv \
        --debug-dir debug

The selected grid is printed as:

    --grid x1,y1,x2,y2

and can then be reused.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np


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


# ======================================================================
# Types / globals
# ======================================================================

Grid = tuple[
    float,
    float,
    float,
    float,
]


@dataclass(frozen=True)
class LocalSegmentMeasurementConfig:
    segment_percentile: float
    background_percentile: float


RDS200_LOCAL_SEGMENT_MEASUREMENT = (
    LocalSegmentMeasurementConfig(
        segment_percentile=35.0,
        background_percentile=70.0,
    )
)

ORIGINAL_EXTRACT_DARKNESS = (
    core.extract_darkness
)

ORIGINAL_RDS200_PROFILE = (
    core.PROFILES["rds200"]
)

ORIGINAL_ROI_FINDER = (
    roi_app.find_display_crop_roi
)


# ======================================================================
# Parse grid
# ======================================================================


def parse_grid(
    text: str,
) -> Grid:
    try:
        values = tuple(
            float(part.strip())
            for part in text.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--grid must contain four numbers: x1,y1,x2,y2"
        ) from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "--grid must contain four numbers: x1,y1,x2,y2"
        )

    x1, y1, x2, y2 = values

    if not (
        0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "--grid must satisfy "
            "0 <= x1 < x2 <= 1 and "
            "0 <= y1 < y2 <= 1"
        )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def format_grid(
    grid: Grid,
) -> str:
    return ",".join(
        f"{value:.6f}"
        for value in grid
    )


# ======================================================================
# Parse extra arguments
# ======================================================================


def parse_extra_args(
    argv: Sequence[str] | None = None,
) -> tuple[
    argparse.Namespace,
    list[str],
]:
    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--quad",
        type=rect_app.parse_quad,
        required=True,
    )

    parser.add_argument(
        "--select-grid",
        action="store_true",
    )

    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=None,
    )

    parser.add_argument(
        "--grid-time",
        type=float,
        default=None,
    )

    return parser.parse_known_args(argv)


def parse_roi_args(
    remaining: list[str],
) -> argparse.Namespace:
    return roi_app.parse_args(
        remaining
    )


# ======================================================================
# Interactive grid selection
# ======================================================================


def select_digit_grid(
    rectified: np.ndarray,
    profile: core.Profile,
) -> Grid:
    """
    Select a tight rectangle around ALL THREE large digits.

    Do not include:
      - the scale above the digits,
      - uSv/h,
      - the black bezel.

    Including the decimal dots is not necessary.
    """

    if rectified.ndim == 2:
        display = cv2.cvtColor(
            rectified,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        display = rectified.copy()

    # Draw the OLD grid lightly as orientation help only.
    for (
        x1,
        y1,
        x2,
        y2,
    ) in profile.digit_boxes:
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 180, 0),
            1,
        )

    cv2.putText(
        display,
        "Select a tight box around ALL THREE numeric digits",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    window = (
        "Select three-digit LCD grid - ENTER/SPACE accept"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    x, y, width, height = cv2.selectROI(
        window,
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        window
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Digit-grid selection cancelled."
        )

    h, w = (
        rectified.shape[:2]
    )

    return (
        x / w,
        y / h,
        (x + width) / w,
        (y + height) / h,
    )


# ======================================================================
# Build fixed profile
# ======================================================================


def transform_box(
    box: tuple[
        int,
        int,
        int,
        int,
    ],
    source: tuple[
        float,
        float,
        float,
        float,
    ],
    destination: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    int,
    int,
    int,
    int,
]:
    sx1, sy1, sx2, sy2 = (
        source
    )

    dx1, dy1, dx2, dy2 = (
        destination
    )

    scale_x = (
        (dx2 - dx1)
        / (sx2 - sx1)
    )

    scale_y = (
        (dy2 - dy1)
        / (sy2 - sy1)
    )

    x1, y1, x2, y2 = box

    return (
        int(
            round(
                dx1
                + (x1 - sx1)
                * scale_x
            )
        ),
        int(
            round(
                dy1
                + (y1 - sy1)
                * scale_y
            )
        ),
        int(
            round(
                dx1
                + (x2 - sx1)
                * scale_x
            )
        ),
        int(
            round(
                dy1
                + (y2 - sy1)
                * scale_y
            )
        ),
    )


def make_fixed_profile(
    profile: core.Profile,
    grid: Grid,
) -> core.Profile:
    original_x1 = min(
        box[0]
        for box in profile.digit_boxes
    )

    original_y1 = min(
        box[1]
        for box in profile.digit_boxes
    )

    original_x2 = max(
        box[2]
        for box in profile.digit_boxes
    )

    original_y2 = max(
        box[3]
        for box in profile.digit_boxes
    )

    source = (
        float(original_x1),
        float(original_y1),
        float(original_x2),
        float(original_y2),
    )

    destination = (
        grid[0]
        * profile.canonical_width,
        grid[1]
        * profile.canonical_height,
        grid[2]
        * profile.canonical_width,
        grid[3]
        * profile.canonical_height,
    )

    new_digit_boxes = tuple(
        transform_box(
            box,
            source,
            destination,
        )
        for box in profile.digit_boxes
    )

    new_decimal_candidates = []

    for (
        x1,
        y1,
        x2,
        y2,
        decimal_places,
    ) in profile.decimal_candidates:
        transformed = transform_box(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            source,
            destination,
        )

        new_decimal_candidates.append(
            (
                transformed[0],
                transformed[1],
                transformed[2],
                transformed[3],
                decimal_places,
            )
        )

    return replace(
        profile,
        digit_boxes=new_digit_boxes,
        decimal_candidates=tuple(
            new_decimal_candidates
        ),

        # CRITICAL:
        # perspective + manually selected grid define geometry.
        # No per-frame shift anymore.
        adaptive_y_shift=False,
        y_shift_min=0,
        y_shift_max=0,
    )


# ======================================================================
# Local segment background masks
# ======================================================================


def make_local_masks(
    profile: core.Profile,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    segment_masks = (
        core.make_segment_masks(
            profile
        )
    )

    all_segments = np.zeros(
        (130, 65),
        dtype=np.uint8,
    )

    for mask in segment_masks:
        all_segments[
            mask
        ] = 255

    rings = []

    kernel = np.ones(
        (11, 11),
        dtype=np.uint8,
    )

    for mask in segment_masks:
        mask_u8 = (
            mask.astype(np.uint8)
            * 255
        )

        dilated = cv2.dilate(
            mask_u8,
            kernel,
            iterations=1,
        )

        ring = (
            (dilated > 0)
            & (all_segments == 0)
        )

        # Fallback if neighbouring segments consumed too much ring.
        if np.count_nonzero(
            ring
        ) < 20:
            ring = (
                (dilated > 0)
                & (~mask)
            )

        rings.append(
            ring
        )

    return (
        tuple(segment_masks),
        tuple(rings),
    )


# ======================================================================
# Local segment contrast
# ======================================================================


def local_darkness(
    patches,
    profile: core.Profile,
    config: LocalSegmentMeasurementConfig,
) -> np.ndarray:
    segment_masks, rings = (
        make_local_masks(
            profile
        )
    )

    result = np.full(
        (
            len(patches),
            7,
        ),
        np.nan,
        dtype=float,
    )

    for digit_index, patch in enumerate(
        patches
    ):
        for segment_index, (
            segment_mask,
            ring_mask,
        ) in enumerate(
            zip(
                segment_masks,
                rings,
            )
        ):
            segment_pixels = (
                patch[
                    segment_mask
                ]
            )

            background_pixels = (
                patch[
                    ring_mask
                ]
            )

            if (
                segment_pixels.size == 0
                or background_pixels.size == 0
            ):
                continue

            # Active segment contains a dark core.
            segment_level = float(
                np.percentile(
                    segment_pixels,
                    config.segment_percentile,
                )
            )

            # Local LCD immediately around the segment.
            background_level = float(
                np.percentile(
                    background_pixels,
                    config.background_percentile,
                )
            )

            result[
                digit_index,
                segment_index,
            ] = (
                background_level
                - segment_level
            )

    return result


# ======================================================================
# Fixed RDS-200 extraction
# ======================================================================


def fixed_extract_darkness(
    display: np.ndarray,
    profile: core.Profile,
    segment_masks,
    segment_percentile: float = 50.0,
    *,
    contrast_mode: str,
    measurement_config: LocalSegmentMeasurementConfig,
) -> np.ndarray:
    del segment_masks
    del segment_percentile

    if profile.name != "rds200":
        return ORIGINAL_EXTRACT_DARKNESS(
            display,
            profile,
            core.make_segment_masks(
                profile
            ),
        )

    raw_patches = (
        core.digit_patches(
            display,
            profile,
        )
    )

    raw_darkness = (
        local_darkness(
            raw_patches,
            profile,
            measurement_config,
        )
    )

    if contrast_mode == "none":
        return raw_darkness

    enhanced_patches = (
        core.clahe_patches(
            raw_patches
        )
    )

    enhanced_darkness = (
        local_darkness(
            enhanced_patches,
            profile,
            measurement_config,
        )
    )

    if contrast_mode == "clahe":
        return enhanced_darkness

    # auto:
    #
    # pattern_quality() is LOWER when segment intensities more cleanly
    # match legal seven-segment patterns.
    raw_quality = (
        core.pattern_quality(
            raw_darkness,
            profile.digit_patterns
            or core.STANDARD_DIGIT_PATTERNS,
        )
    )

    enhanced_quality = (
        core.pattern_quality(
            enhanced_darkness,
            profile.digit_patterns
            or core.STANDARD_DIGIT_PATTERNS,
        )
    )

    if (
        enhanced_quality
        < raw_quality
    ):
        return enhanced_darkness

    return raw_darkness


def make_fixed_extract_darkness(
    contrast_mode: str,
    measurement_config: LocalSegmentMeasurementConfig,
):
    def configured_fixed_extract_darkness(
        display: np.ndarray,
        profile: core.Profile,
        segment_masks,
        segment_percentile: float = 50.0,
    ) -> np.ndarray:
        return fixed_extract_darkness(
            display,
            profile,
            segment_masks,
            segment_percentile=segment_percentile,
            contrast_mode=contrast_mode,
            measurement_config=measurement_config,
        )

    return configured_fixed_extract_darkness


# ======================================================================
# Rectified finder
# ======================================================================


def make_rectified_finder(
    quad: np.ndarray,
):
    def finder(
        frame,
        profile,
        previous_box,
        roi,
    ):
        display, box = (
            ORIGINAL_ROI_FINDER(
                frame,
                profile,
                previous_box,
                roi,
            )
        )

        if display is None:
            return (
                None,
                box,
            )

        display = (
            rect_app.rectify_display(
                display,
                profile,
                quad,
            )
        )

        return (
            display,
            box,
        )

    return finder


# ======================================================================
# Main
# ======================================================================


def main(
    argv: Sequence[str] | None = None,
    decode_samples: roi_app.DecodeSamples | None = None,
    base_profile_override: core.Profile | None = None,
    debug_writer: roi_app.DebugWriter | None = None,
) -> int:
    extra, remaining = (
        parse_extra_args(argv)
    )

    args = parse_roi_args(
        remaining
    )

    if args.profile != "rds200":
        print(
            "Error: this experimental fixed-grid version "
            "currently supports only --profile rds200.",
            file=sys.stderr,
        )

        return 1

    if args.roi is None:
        print(
            "Error: --roi is required.",
            file=sys.stderr,
        )

        return 1

    if (
        extra.select_grid
        and extra.grid is not None
    ):
        print(
            "Error: use either --select-grid or --grid.",
            file=sys.stderr,
        )

        return 1

    if (
        not extra.select_grid
        and extra.grid is None
    ):
        print(
            "Error: first run requires --select-grid; "
            "later runs may use --grid.",
            file=sys.stderr,
        )

        return 1

    try:
        info = core.probe_video(
            args.video
        )

        base_profile = (
            ORIGINAL_RDS200_PROFILE
            if base_profile_override is None
            else base_profile_override
        )

        sample_fps = (
            args.sample_fps
            if args.sample_fps is not None
            else base_profile.default_sample_fps
        )

        grid = (
            extra.grid
        )

        # ==========================================================
        # Select digit grid on a rectified reference frame
        # ==========================================================

        if extra.select_grid:
            target_time = (
                extra.grid_time
                if extra.grid_time is not None
                else 0.5 * info.duration
            )

            print(
                (
                    "Scanning toward reference frame "
                    f"t={target_time:.3f} s..."
                ),
                file=sys.stderr,
            )

            reference_display, actual_time = (
                rect_app.find_reference_display(
                    args.video,
                    info,
                    base_profile,
                    args.roi,
                    args.processing_width,
                    sample_fps,
                    target_time,
                )
            )

            reference_rectified = (
                rect_app.rectify_display(
                    reference_display,
                    base_profile,
                    extra.quad,
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
                "Select a TIGHT rectangle around the THREE large digits.",
                file=sys.stderr,
            )

            print(
                "Do not include the scale or uSv/h.",
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

            grid = (
                select_digit_grid(
                    reference_rectified,
                    base_profile,
                )
            )

            print(
                "",
                file=sys.stderr,
            )

            print(
                "Selected fixed digit grid:",
                file=sys.stderr,
            )

            print(
                (
                    "  --grid "
                    + format_grid(
                        grid
                    )
                ),
                file=sys.stderr,
            )

            print(
                "",
                file=sys.stderr,
            )

        if grid is None:
            raise RuntimeError(
                "No fixed digit grid available."
            )

        fixed_profile = (
            make_fixed_profile(
                base_profile,
                grid,
            )
        )

        print(
            "Fixed digit boxes:",
            file=sys.stderr,
        )

        for index, box in enumerate(
            fixed_profile.digit_boxes,
            start=1,
        ):
            print(
                f"  digit {index}: {box}",
                file=sys.stderr,
            )

        print(
            "Fixed decimal candidates:",
            file=sys.stderr,
        )

        for item in (
            fixed_profile.decimal_candidates
        ):
            print(
                f"  {item}",
                file=sys.stderr,
            )

        print(
            "",
            file=sys.stderr,
        )

        # ==========================================================
        # Configure fixed profile and fixed extraction
        # ==========================================================

        original_finder = (
            roi_app.find_display_crop_roi
        )

        darkness_extractor = (
            make_fixed_extract_darkness(
                args.contrast,
                RDS200_LOCAL_SEGMENT_MEASUREMENT,
            )
        )

        roi_app.find_display_crop_roi = (
            make_rectified_finder(
                extra.quad
            )
        )

        # ==========================================================
        # Run normal ROI pipeline
        # ==========================================================

        try:
            result = (
                roi_app.main(
                    remaining,
                    darkness_extractor=(
                        darkness_extractor
                    ),
                    profile_override=(
                        fixed_profile
                    ),
                    decode_samples=(
                        decode_samples
                    ),
                    debug_writer=(
                        debug_writer
                    ),
                )
            )

        finally:
            roi_app.find_display_crop_roi = (
                original_finder
            )

        print()

        print(
            "Fixed digit grid used:"
        )

        print(
            (
                "  --grid "
                + format_grid(
                    grid
                )
            )
        )

        print()

        print(
            "Perspective quad used:"
        )

        print(
            (
                "  --quad "
                + rect_app.format_quad(
                    extra.quad
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
