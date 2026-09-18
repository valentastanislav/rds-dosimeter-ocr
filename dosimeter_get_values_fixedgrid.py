#!/usr/bin/env python3
"""
Fixed-grid RADOS dosimeter decoder.

Uses:
    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py

The display is first perspective-rectified using the already determined
--quad.  Digit geometry then comes from the selected Profile policy: either
one manually selected fixed rectangle, or the Profile's canonical digit
boxes.  There is no frame-by-frame y-shift or geometry optimizer.

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
from dataclasses import replace
from typing import Callable, Sequence

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
RectifiedFinderFactory = Callable[
    [np.ndarray],
    roi_app.RoiDisplayFinder,
]


ORIGINAL_EXTRACT_DARKNESS = (
    core.extract_darkness
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
    require_quad: bool = True,
    usage: str | None = None,
) -> tuple[
    argparse.Namespace,
    list[str],
]:
    parser = argparse.ArgumentParser(
        add_help=False,
        usage=usage,
    )

    parser.add_argument(
        "--quad",
        type=rect_app.parse_quad,
        required=require_quad,
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

    parser.add_argument(
        "--decoder-measurement",
        choices=(
            "profile",
            "local-p70-seg10",
        ),
        default="profile",
        help=(
            "auxiliary decoder segment measurement strategy "
            "(default: profile)"
        ),
    )

    return parser.parse_known_args(argv)


def parse_roi_args(
    remaining: list[str],
    usage: str | None = None,
) -> argparse.Namespace:
    return roi_app.parse_args(
        remaining,
        usage=usage,
    )


# ======================================================================
# Profile-driven fixed-grid geometry policy
# ======================================================================


def validate_geometry_options(
    profile: core.Profile,
    select_grid: bool,
    grid: Grid | None,
) -> None:
    mode = (
        profile.fixedgrid_geometry_mode
    )

    if mode == "manual_grid":
        if select_grid and grid is not None:
            raise ValueError(
                "Use either --select-grid or --grid."
            )

        if not select_grid and grid is None:
            raise ValueError(
                "First run requires --select-grid; "
                "later runs may use --grid."
            )

        return

    if mode == "profile":
        if select_grid or grid is not None:
            raise ValueError(
                f"Profile {profile.name} uses canonical digit geometry; "
                "do not use --select-grid or --grid."
            )

        return

    raise ValueError(
        "Unknown fixed-grid geometry mode: "
        f"{mode}"
    )


# ======================================================================
# Interactive grid selection
# ======================================================================


def select_digit_grid(
    rectified: np.ndarray,
    profile: core.Profile,
) -> Grid:
    """
    Select a tight rectangle around all configured numeric digits.

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

    digit_count = len(
        profile.digit_boxes
    )

    cv2.putText(
        display,
        f"Select a tight box around all {digit_count} numeric positions",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    window = (
        f"Select {digit_count}-position LCD grid - ENTER/SPACE accept"
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

    # RDS-200 decimal dots are part of the same physical LCD geometry as
    # the digits.  For a tight manually selected grid, derive their final
    # position from the final digit boxes instead of preserving the legacy
    # absolute profile coordinates.
    #
    # Horizontally the dot is centred on the boundary between neighbouring
    # digits and is allowed to straddle that boundary.  Vertically its lower
    # edge follows the transformed bottom segment d.
    if (
        profile.name == "rds200"
        and profile.digit_segment_polygons is not None
        and new_decimal_candidates
    ):
        bottom_edges = []

        for digit_box, polygons in zip(
            new_digit_boxes,
            profile.digit_segment_polygons,
        ):
            if "d" not in polygons:
                continue

            _bx1, by1, _bx2, by2 = digit_box
            box_height = max(1, by2 - by1)
            local_bottom = float(
                np.max(np.asarray(polygons["d"])[:, 1])
            )
            bottom_edges.append(
                by1 + local_bottom * box_height / 130.0
            )

        if bottom_edges:
            target_y2 = int(
                round(float(np.median(bottom_edges)))
            )
            target_y2 = max(
                1,
                min(profile.canonical_height, target_y2),
            )

            aligned_decimal_candidates = []
            number_digits = len(new_digit_boxes)

            for (
                x1,
                old_y1,
                x2,
                old_y2,
                decimal_places,
            ) in new_decimal_candidates:
                width = max(1, x2 - x1)
                height = max(1, old_y2 - old_y1)

                left_index = (
                    number_digits
                    - int(decimal_places)
                    - 1
                )
                right_index = left_index + 1

                if (
                    0 <= left_index < number_digits
                    and 0 <= right_index < number_digits
                ):
                    boundary_x = 0.5 * (
                        new_digit_boxes[left_index][2]
                        + new_digit_boxes[right_index][0]
                    )
                    new_x1 = int(
                        round(boundary_x - 0.5 * width)
                    )
                    new_x2 = new_x1 + width
                else:
                    new_x1 = x1
                    new_x2 = x2

                new_x1 = max(0, new_x1)
                new_x2 = min(
                    profile.canonical_width,
                    new_x2,
                )
                new_y2 = target_y2
                new_y1 = max(0, new_y2 - height)

                aligned_decimal_candidates.append(
                    (
                        new_x1,
                        new_y1,
                        new_x2,
                        new_y2,
                        decimal_places,
                    )
                )

            new_decimal_candidates = (
                aligned_decimal_candidates
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


def make_profile_geometry_fixed_profile(
    profile: core.Profile,
) -> core.Profile:
    return replace(
        profile,

        # Perspective rectification plus canonical Profile digit boxes
        # define the fixed geometry. No per-frame shift is used.
        adaptive_y_shift=False,
        y_shift_min=0,
        y_shift_max=0,
    )


# ======================================================================
# Local segment background masks
# ======================================================================


def make_local_masks(
    profile: core.Profile,
    digit_index: int | None = None,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    segment_masks = (
        core.make_segment_masks(
            profile,
            digit_index=digit_index,
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
    config: core.FixedGridMeasurementConfig,
) -> np.ndarray:
    shared_masks = (
        make_local_masks(profile)
        if profile.digit_segment_polygons is None
        else None
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
        segment_masks, rings = (
            shared_masks
            if shared_masks is not None
            else make_local_masks(
                profile,
                digit_index=digit_index,
            )
        )
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
# Fixed-grid extraction
# ======================================================================


def fixed_extract_darkness(
    display: np.ndarray,
    profile: core.Profile,
    segment_masks,
    segment_percentile: float = 50.0,
    *,
    contrast_mode: str,
    measurement_config: core.FixedGridMeasurementConfig,
) -> np.ndarray:
    del segment_masks
    del segment_percentile

    if measurement_config.mode == "core":
        return ORIGINAL_EXTRACT_DARKNESS(
            display,
            profile,
            core.make_segment_masks(
                profile
            ),
            segment_percentile=(
                measurement_config.segment_percentile
            ),
            background_percentile=(
                measurement_config.background_percentile
            ),
        )

    if measurement_config.mode != "local":
        raise ValueError(
            "Unknown fixed-grid measurement mode: "
            f"{measurement_config.mode}"
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
    measurement_config: core.FixedGridMeasurementConfig,
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


def make_decoder_measurement_extractor(
    strategy: str,
):
    if strategy == "profile":
        return None

    if strategy == "local-p70-seg10":
        return make_fixed_extract_darkness(
            "none",
            core.FixedGridMeasurementConfig(
                mode="local",
                segment_percentile=10.0,
                background_percentile=70.0,
            ),
        )

    raise ValueError(
        "Unknown decoder measurement strategy: "
        f"{strategy}"
    )


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
    rectified_finder_factory: RectifiedFinderFactory | None = None,
) -> int:
    extra, remaining = (
        parse_extra_args(argv)
    )

    args = parse_roi_args(
        remaining
    )

    if (
        args.roi is None
        and rectified_finder_factory is None
    ):
        print(
            "Error: --roi is required.",
            file=sys.stderr,
        )

        return 1

    try:
        base_profile = (
            core.PROFILES[
                args.profile
            ]
            if base_profile_override is None
            else base_profile_override
        )

        validate_geometry_options(
            base_profile,
            extra.select_grid,
            extra.grid,
        )

        info = core.probe_video(
            args.video
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
                (
                    "Select a TIGHT rectangle around all "
                    f"{len(base_profile.digit_boxes)} numeric positions."
                ),
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

        if (
            base_profile.fixedgrid_geometry_mode
            == "manual_grid"
        ):
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

        else:
            fixed_profile = (
                make_profile_geometry_fixed_profile(
                    base_profile
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

        selected_rectified_finder_factory = (
            make_rectified_finder
            if rectified_finder_factory is None
            else rectified_finder_factory
        )

        darkness_extractor = (
            make_fixed_extract_darkness(
                args.contrast,
                fixed_profile.fixedgrid_primary_measurement,
            )
        )

        auxiliary_darkness_extractor = (
            make_decoder_measurement_extractor(
                extra.decoder_measurement
            )
        )

        display_finder = (
            selected_rectified_finder_factory(
                extra.quad
            )
        )

        # ==========================================================
        # Run normal ROI pipeline
        # ==========================================================

        result = (
            roi_app.main(
                remaining,
                darkness_extractor=(
                    darkness_extractor
                ),
                auxiliary_darkness_extractor=(
                    auxiliary_darkness_extractor
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
                display_finder=(
                    display_finder
                ),
            )
        )

        if grid is not None:
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
