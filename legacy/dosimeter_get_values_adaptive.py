#!/usr/bin/env python3
"""
Adaptive RDS-200 decoder.

This is an extension of:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py

It keeps the ROI handling and all output processing from
dosimeter_get_values_roi.py, but replaces the RDS-200 segment
measurement with:

    * local segment/background contrast,
    * adaptive x/y translation of the digit grid,
    * adaptive x/y scale,
    * temporal tracking of the best geometry,
    * automatic raw/CLAHE selection.

Usage is the same as dosimeter_get_values_roi.py, for example:

    python3 dosimeter_get_values_adaptive.py IMG_1151edited.mov out.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --contrast auto \
        --raw-output raw.csv \
        --debug-dir debug
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import dosimeter_get_values as core
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values.py.\n"
        "Put dosimeter_get_values_adaptive.py in the same directory."
    ) from exc

try:
    import dosimeter_get_values_roi as roi_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_roi.py.\n"
        "Put dosimeter_get_values_adaptive.py in the same directory."
    ) from exc


# ======================================================================
# Save original functions for non-RDS200 profiles
# ======================================================================

_ORIGINAL_EXTRACT_DARKNESS_ADAPTIVE = core.extract_darkness_adaptive
_ORIGINAL_DECIMAL_SCORES = core.extract_decimal_scores


# ======================================================================
# Geometry state
# ======================================================================


@dataclass
class Geometry:
    dx: float = 0.0
    dy: float = 0.0
    sx: float = 1.0
    sy: float = 1.0


@dataclass
class AdaptiveState:
    geometry: Geometry
    quality: float
    frame_count: int


_STATE = AdaptiveState(
    geometry=Geometry(),
    quality=float("-inf"),
    frame_count=0,
)


# ======================================================================
# Masks for local segment contrast
# ======================================================================

_LOCAL_MASK_CACHE: dict[
    str,
    tuple[
        tuple[np.ndarray, ...],
        tuple[np.ndarray, ...],
    ],
] = {}


def make_local_segment_masks(
    profile: core.Profile,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """
    Return:

        segment masks
        local background-ring masks

    Each ring surrounds one seven-segment polygon but excludes all
    segment polygons.  Thus every segment is compared with its immediate
    LCD neighbourhood rather than one brightness value for the whole digit.
    """

    if profile.name in _LOCAL_MASK_CACHE:
        return _LOCAL_MASK_CACHE[
            profile.name
        ]

    segment_masks = core.make_segment_masks(
        profile
    )

    union = np.zeros(
        (130, 65),
        dtype=np.uint8,
    )

    for mask in segment_masks:
        union[mask] = 255

    rings: list[np.ndarray] = []

    kernel = np.ones(
        (9, 9),
        dtype=np.uint8,
    )

    large_kernel = np.ones(
        (13, 13),
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
            & (union == 0)
        )

        # Near neighbouring segments the strict ring can occasionally
        # become too small.
        if np.count_nonzero(ring) < 20:
            dilated = cv2.dilate(
                mask_u8,
                large_kernel,
                iterations=1,
            )

            ring = (
                (dilated > 0)
                & (~mask)
            )

        rings.append(
            ring
        )

    result = (
        tuple(segment_masks),
        tuple(rings),
    )

    _LOCAL_MASK_CACHE[
        profile.name
    ] = result

    return result


# ======================================================================
# Geometry transformation
# ======================================================================


def digit_grid_center(
    profile: core.Profile,
) -> tuple[float, float]:
    x1 = min(
        box[0]
        for box in profile.digit_boxes
    )

    y1 = min(
        box[1]
        for box in profile.digit_boxes
    )

    x2 = max(
        box[2]
        for box in profile.digit_boxes
    )

    y2 = max(
        box[3]
        for box in profile.digit_boxes
    )

    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
    )


def transform_point(
    x: float,
    y: float,
    profile: core.Profile,
    geometry: Geometry,
) -> tuple[float, float]:
    cx, cy = digit_grid_center(
        profile
    )

    tx = (
        cx
        + geometry.sx * (x - cx)
        + geometry.dx
    )

    ty = (
        cy
        + geometry.sy * (y - cy)
        + geometry.dy
    )

    return tx, ty


def transformed_digit_boxes(
    profile: core.Profile,
    geometry: Geometry,
) -> tuple[
    tuple[int, int, int, int],
    ...,
]:
    result: list[
        tuple[int, int, int, int]
    ] = []

    for (
        x1,
        y1,
        x2,
        y2,
    ) in profile.digit_boxes:
        tx1, ty1 = transform_point(
            x1,
            y1,
            profile,
            geometry,
        )

        tx2, ty2 = transform_point(
            x2,
            y2,
            profile,
            geometry,
        )

        result.append(
            (
                int(round(tx1)),
                int(round(ty1)),
                int(round(tx2)),
                int(round(ty2)),
            )
        )

    return tuple(result)


# ======================================================================
# Extract digit patches for a candidate geometry
# ======================================================================


def geometry_digit_patches(
    display: np.ndarray,
    profile: core.Profile,
    geometry: Geometry,
) -> list[np.ndarray] | None:
    height, width = display.shape

    boxes = transformed_digit_boxes(
        profile,
        geometry,
    )

    patches: list[np.ndarray] = []

    for (
        x1,
        y1,
        x2,
        y2,
    ) in boxes:
        if (
            x1 < 0
            or y1 < 0
            or x2 > width
            or y2 > height
            or x2 - x1 < 20
            or y2 - y1 < 40
        ):
            return None

        patch = display[
            y1:y2,
            x1:x2,
        ]

        if patch.size == 0:
            return None

        patch = cv2.resize(
            patch,
            (65, 130),
            interpolation=cv2.INTER_LINEAR,
        )

        patches.append(
            patch
        )

    return patches


# ======================================================================
# Local segment contrast
# ======================================================================


def local_darkness_from_patches(
    patches: list[np.ndarray],
    profile: core.Profile,
) -> np.ndarray:
    segment_masks, background_rings = (
        make_local_segment_masks(
            profile
        )
    )

    result = np.empty(
        (
            len(patches),
            7,
        ),
        dtype=float,
    )

    for digit_index, patch in enumerate(
        patches
    ):
        # A small global component prevents the local ring method from
        # becoming unstable if a reflection crosses the immediate ring.
        global_background = float(
            np.percentile(
                patch,
                85,
            )
        )

        for segment_index, (
            mask,
            ring,
        ) in enumerate(
            zip(
                segment_masks,
                background_rings,
            )
        ):
            segment_pixels = patch[
                mask
            ]

            background_pixels = patch[
                ring
            ]

            if (
                segment_pixels.size == 0
                or background_pixels.size == 0
            ):
                result[
                    digit_index,
                    segment_index,
                ] = float("nan")

                continue

            # Active LCD segments contain dark pixels.  Lower percentile
            # is deliberately used so partial mask overlap remains useful.
            segment_level = float(
                np.percentile(
                    segment_pixels,
                    35,
                )
            )

            local_background = float(
                np.percentile(
                    background_pixels,
                    75,
                )
            )

            local_contrast = (
                local_background
                - segment_level
            )

            global_contrast = (
                global_background
                - segment_level
            )

            result[
                digit_index,
                segment_index,
            ] = (
                0.82 * local_contrast
                + 0.18 * global_contrast
            )

    return result


# ======================================================================
# Pattern quality for geometry selection
# ======================================================================


def single_digit_geometry_quality(
    levels: np.ndarray,
    patterns: dict[
        int,
        tuple[int, ...],
    ],
) -> float:
    """
    Score how well seven measured segment levels can be separated into
    active/inactive groups for ANY legal digit.

    A high positive margin is good.
    """

    levels = np.asarray(
        levels,
        dtype=float,
    )

    if (
        levels.shape != (7,)
        or not np.all(
            np.isfinite(levels)
        )
    ):
        return -1000.0

    candidates: list[float] = []

    for digit, pattern in patterns.items():
        active_mask = np.asarray(
            pattern,
            dtype=bool,
        )

        active = levels[
            active_mask
        ]

        inactive = levels[
            ~active_mask
        ]

        if inactive.size:
            weakest_active = float(
                np.percentile(
                    active,
                    20,
                )
            )

            strongest_inactive = float(
                np.percentile(
                    inactive,
                    80,
                )
            )

            margin = (
                weakest_active
                - strongest_inactive
            )

            contrast = float(
                np.mean(active)
                - np.mean(inactive)
            )

            score = (
                margin
                + 0.15 * contrast
            )

        else:
            # Digit 8: all seven segments must carry genuine evidence.
            weakest_active = float(
                np.percentile(
                    active,
                    20,
                )
            )

            contrast = (
                float(np.mean(active))
                - 8.0
            )

            score = (
                weakest_active
                - 8.0
                + 0.10 * contrast
            )

        candidates.append(
            float(score)
        )

    candidates.sort(
        reverse=True
    )

    best = candidates[0]

    second = (
        candidates[1]
        if len(candidates) > 1
        else best
    )

    separation = (
        best
        - second
    )

    # Geometry should produce both a good legal digit and some separation
    # from the next-best pattern.
    return float(
        best
        + 0.12 * separation
    )


def geometry_quality(
    darkness: np.ndarray,
    profile: core.Profile,
) -> float:
    patterns = (
        profile.digit_patterns
        or core.STANDARD_DIGIT_PATTERNS
    )

    qualities = [
        single_digit_geometry_quality(
            digit,
            patterns,
        )
        for digit in darkness
    ]

    if not qualities:
        return float("-inf")

    # Require all three positions to be reasonably aligned.  The weakest
    # digit receives extra weight so one excellent digit cannot hide two
    # badly positioned ones.
    return float(
        sum(qualities)
        + 0.50 * min(qualities)
    )


# ======================================================================
# Evaluate one geometry
# ======================================================================


def evaluate_geometry(
    raw_display: np.ndarray,
    enhanced_display: np.ndarray | None,
    profile: core.Profile,
    geometry: Geometry,
    contrast_mode: str,
) -> tuple[
    float,
    np.ndarray | None,
]:
    raw_patches = geometry_digit_patches(
        raw_display,
        profile,
        geometry,
    )

    if raw_patches is None:
        return (
            float("-inf"),
            None,
        )

    raw_darkness = (
        local_darkness_from_patches(
            raw_patches,
            profile,
        )
    )

    raw_quality = geometry_quality(
        raw_darkness,
        profile,
    )

    if contrast_mode == "none":
        return (
            raw_quality,
            raw_darkness,
        )

    if enhanced_display is None:
        return (
            raw_quality,
            raw_darkness,
        )

    enhanced_patches = (
        geometry_digit_patches(
            enhanced_display,
            profile,
            geometry,
        )
    )

    if enhanced_patches is None:
        return (
            raw_quality,
            raw_darkness,
        )

    enhanced_darkness = (
        local_darkness_from_patches(
            enhanced_patches,
            profile,
        )
    )

    enhanced_quality = geometry_quality(
        enhanced_darkness,
        profile,
    )

    if contrast_mode == "clahe":
        return (
            enhanced_quality,
            enhanced_darkness,
        )

    # auto
    if enhanced_quality > raw_quality:
        return (
            enhanced_quality,
            enhanced_darkness,
        )

    return (
        raw_quality,
        raw_darkness,
    )


# ======================================================================
# Candidate generators
# ======================================================================


def clipped_unique(
    values: list[float],
    minimum: float,
    maximum: float,
    digits: int = 4,
) -> list[float]:
    result = {
        round(
            min(
                maximum,
                max(
                    minimum,
                    value,
                ),
            ),
            digits,
        )
        for value in values
    }

    return sorted(
        result
    )


def coarse_geometries() -> list[Geometry]:
    geometries: list[
        Geometry
    ] = []

    # Coarse acquisition.  This is intentionally performed rarely.
    dx_values = (
        -12,
        -8,
        -4,
        0,
        4,
        8,
        12,
    )

    dy_values = (
        -18,
        -12,
        -6,
        0,
        6,
        12,
    )

    scale_values = (
        0.92,
        0.96,
        1.00,
        1.04,
        1.08,
    )

    for dx in dx_values:
        for dy in dy_values:
            for sx in scale_values:
                for sy in scale_values:
                    geometries.append(
                        Geometry(
                            dx=float(dx),
                            dy=float(dy),
                            sx=float(sx),
                            sy=float(sy),
                        )
                    )

    return geometries


def local_geometries(
    center: Geometry,
    wide: bool = False,
) -> list[Geometry]:
    if wide:
        shift_step = 3.0
        scale_step = 0.025
    else:
        shift_step = 2.0
        scale_step = 0.015

    dx_values = clipped_unique(
        [
            center.dx - shift_step,
            center.dx,
            center.dx + shift_step,
        ],
        -16.0,
        16.0,
    )

    dy_values = clipped_unique(
        [
            center.dy - shift_step,
            center.dy,
            center.dy + shift_step,
        ],
        -24.0,
        16.0,
    )

    sx_values = clipped_unique(
        [
            center.sx - scale_step,
            center.sx,
            center.sx + scale_step,
        ],
        0.88,
        1.12,
    )

    sy_values = clipped_unique(
        [
            center.sy - scale_step,
            center.sy,
            center.sy + scale_step,
        ],
        0.88,
        1.12,
    )

    result: list[
        Geometry
    ] = []

    for dx in dx_values:
        for dy in dy_values:
            for sx in sx_values:
                for sy in sy_values:
                    result.append(
                        Geometry(
                            dx=dx,
                            dy=dy,
                            sx=sx,
                            sy=sy,
                        )
                    )

    return result


# ======================================================================
# Search
# ======================================================================


def search_geometries(
    raw_display: np.ndarray,
    enhanced_display: np.ndarray | None,
    profile: core.Profile,
    contrast_mode: str,
    geometries: list[Geometry],
    previous: Geometry | None,
) -> tuple[
    Geometry,
    float,
    np.ndarray,
]:
    best_geometry: Geometry | None = None

    best_quality = float(
        "-inf"
    )

    best_darkness: np.ndarray | None = None

    for geometry in geometries:
        quality, darkness = (
            evaluate_geometry(
                raw_display,
                enhanced_display,
                profile,
                geometry,
                contrast_mode,
            )
        )

        if darkness is None:
            continue

        # Temporal regularization: a small improvement is not enough
        # to justify a large jump of the grid between adjacent frames.
        if previous is not None:
            movement = (
                abs(
                    geometry.dx
                    - previous.dx
                )
                + abs(
                    geometry.dy
                    - previous.dy
                )
            )

            scale_change = (
                abs(
                    geometry.sx
                    - previous.sx
                )
                + abs(
                    geometry.sy
                    - previous.sy
                )
            )

            quality -= (
                0.08 * movement
                + 5.0 * scale_change
            )

        if quality > best_quality:
            best_quality = quality
            best_geometry = geometry
            best_darkness = darkness

    if (
        best_geometry is None
        or best_darkness is None
    ):
        raise RuntimeError(
            "Adaptive geometry search found no usable digit grid."
        )

    return (
        best_geometry,
        float(best_quality),
        best_darkness,
    )


# ======================================================================
# Adaptive RDS-200 extractor
# ======================================================================


def adaptive_extract_darkness(
    display: np.ndarray,
    profile: core.Profile,
    mask_bank: dict[
        int,
        tuple[np.ndarray, ...],
    ],
    previous_shift: int | None,
    contrast_mode: str,
) -> tuple[
    np.ndarray,
    int,
]:
    """
    Drop-in replacement for core.extract_darkness_adaptive().
    """

    del mask_bank
    del previous_shift

    if profile.name != "rds200":
        return _ORIGINAL_EXTRACT_DARKNESS_ADAPTIVE(
            display,
            profile,
            core.make_y_shift_mask_bank(
                profile
            ),
            None,
            contrast_mode,
        )

    _STATE.frame_count += 1

    # Apply CLAHE once to the complete normalized LCD image.
    # Candidate digit crops are then inexpensive.
    enhanced_display: np.ndarray | None = None

    if contrast_mode != "none":
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced_display = (
            clahe.apply(
                display
            )
        )

    previous_geometry = (
        _STATE.geometry
    )

    # ----------------------------------------------------------
    # First frame: broad acquisition
    # ----------------------------------------------------------

    if _STATE.frame_count == 1:
        (
            best_geometry,
            best_quality,
            best_darkness,
        ) = search_geometries(
            display,
            enhanced_display,
            profile,
            contrast_mode,
            coarse_geometries(),
            previous=None,
        )

        # Fine search around the coarse optimum.
        (
            best_geometry,
            best_quality,
            best_darkness,
        ) = search_geometries(
            display,
            enhanced_display,
            profile,
            contrast_mode,
            local_geometries(
                best_geometry,
                wide=True,
            ),
            previous=best_geometry,
        )

    else:
        # ------------------------------------------------------
        # Normal frame: small search around previous geometry
        # ------------------------------------------------------

        (
            best_geometry,
            best_quality,
            best_darkness,
        ) = search_geometries(
            display,
            enhanced_display,
            profile,
            contrast_mode,
            local_geometries(
                previous_geometry,
                wide=False,
            ),
            previous=previous_geometry,
        )

        # ------------------------------------------------------
        # Periodic / quality-driven reacquisition
        # ------------------------------------------------------

        need_reacquire = (
            _STATE.frame_count % 60 == 0
            or (
                math.isfinite(
                    _STATE.quality
                )
                and best_quality
                < _STATE.quality - 12.0
            )
        )

        if need_reacquire:
            (
                coarse_geometry,
                coarse_quality,
                coarse_darkness,
            ) = search_geometries(
                display,
                enhanced_display,
                profile,
                contrast_mode,
                coarse_geometries(),
                previous=previous_geometry,
            )

            if (
                coarse_quality
                > best_quality
            ):
                best_geometry = (
                    coarse_geometry
                )

                best_quality = (
                    coarse_quality
                )

                best_darkness = (
                    coarse_darkness
                )

                (
                    best_geometry,
                    best_quality,
                    best_darkness,
                ) = search_geometries(
                    display,
                    enhanced_display,
                    profile,
                    contrast_mode,
                    local_geometries(
                        best_geometry,
                        wide=True,
                    ),
                    previous=best_geometry,
                )

    _STATE.geometry = (
        best_geometry
    )

    _STATE.quality = (
        best_quality
    )

    if (
        _STATE.frame_count == 1
        or _STATE.frame_count % 50 == 0
    ):
        print(
            (
                "Adaptive RDS200 geometry: "
                f"frame={_STATE.frame_count}, "
                f"dx={best_geometry.dx:+.1f}, "
                f"dy={best_geometry.dy:+.1f}, "
                f"sx={best_geometry.sx:.3f}, "
                f"sy={best_geometry.sy:.3f}, "
                f"quality={best_quality:.2f}"
            ),
            file=sys.stderr,
        )

    # The legacy return value is an integer y-shift of the mask.
    # Our geometry is represented by the crop transformation instead,
    # therefore no additional legacy mask displacement is required.
    return (
        best_darkness,
        0,
    )


# ======================================================================
# Decimal point using the same transformed coordinate system
# ======================================================================


def adaptive_decimal_scores(
    display: np.ndarray,
    profile: core.Profile,
    y_shift: int = 0,
) -> np.ndarray | None:
    del y_shift

    if profile.name != "rds200":
        return _ORIGINAL_DECIMAL_SCORES(
            display,
            profile,
            y_shift=0,
        )

    if not profile.decimal_candidates:
        return None

    geometry = (
        _STATE.geometry
    )

    height, width = (
        display.shape
    )

    scores: list[
        float
    ] = []

    margin = 5

    for (
        x1,
        y1,
        x2,
        y2,
        _decimal_places,
    ) in profile.decimal_candidates:
        tx1, ty1 = transform_point(
            x1,
            y1,
            profile,
            geometry,
        )

        tx2, ty2 = transform_point(
            x2,
            y2,
            profile,
            geometry,
        )

        x1i = int(
            round(tx1)
        )

        y1i = int(
            round(ty1)
        )

        x2i = int(
            round(tx2)
        )

        y2i = int(
            round(ty2)
        )

        x1i = max(
            0,
            min(
                x1i,
                width,
            ),
        )

        y1i = max(
            0,
            min(
                y1i,
                height,
            ),
        )

        x2i = max(
            0,
            min(
                x2i,
                width,
            ),
        )

        y2i = max(
            0,
            min(
                y2i,
                height,
            ),
        )

        patch = display[
            y1i:y2i,
            x1i:x2i,
        ]

        if patch.size == 0:
            scores.append(
                float("nan")
            )

            continue

        outer_x1 = max(
            0,
            x1i - margin,
        )

        outer_y1 = max(
            0,
            y1i - margin,
        )

        outer_x2 = min(
            width,
            x2i + margin,
        )

        outer_y2 = min(
            height,
            y2i + margin,
        )

        outer = display[
            outer_y1:outer_y2,
            outer_x1:outer_x2,
        ]

        ring_mask = np.ones(
            outer.shape,
            dtype=bool,
        )

        ring_mask[
            y1i - outer_y1:
            y2i - outer_y1,
            x1i - outer_x1:
            x2i - outer_x1,
        ] = False

        ring = outer[
            ring_mask
        ]

        if ring.size:
            background = float(
                np.percentile(
                    ring,
                    85,
                )
            )
        else:
            background = float(
                np.percentile(
                    patch,
                    90,
                )
            )

        dot_level = float(
            np.percentile(
                patch,
                40,
            )
        )

        scores.append(
            background
            - dot_level
        )

    return np.asarray(
        scores,
        dtype=float,
    )


# ======================================================================
# Install replacements
# ======================================================================

core.extract_darkness_adaptive = (
    adaptive_extract_darkness
)

core.extract_decimal_scores = (
    adaptive_decimal_scores
)


# ======================================================================
# Run the existing ROI application
# ======================================================================

if __name__ == "__main__":
    raise SystemExit(
        roi_app.main()
    )