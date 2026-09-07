#!/usr/bin/env python3
"""
Generalised RDS-200 decoder for hand-held videos.

This version keeps the working geometry:

    optical-flow stabilization
    fixed display crop
    fixed perspective quad
    tight digit grid
    vertically stretched seven-segment masks

but removes all digit-specific corrections such as:

    6 -> 8

Instead, every video self-calibrates the OFF and ON darkness level of
each segment independently for:

    digit position 1 / 2 / 3
    segment a / b / c / d / e / f / g

This is useful because LCD segment visibility is strongly
position-dependent.  For example, one active vertical segment may have
darkness 13 while another active segment has darkness 45.

The adaptive classifier is deliberately conservative:

    - if a segment has a clearly bimodal OFF/ON distribution, use it;
    - if not, ignore that segment for adaptive classification;
    - if too little reliable adaptive information exists, fall back to
      the original RDS-200 pattern decoder.

No expected dose-rate values are used.

Recommended temporal smoothing:

    --mode-window 5

At 5 samples/s this suppresses isolated one-frame OCR mistakes without
hard-coding any particular displayed value.

Required existing files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_roi as roi_app
import dosimeter_get_values_fixedgrid as fixed_app
import dosimeter_get_values_flow as flow


# ======================================================================
# Original functions
# ======================================================================


ORIGINAL_DECODE_SAMPLES = (
    core.decode_samples
)

ORIGINAL_PATTERN_DECODER = (
    core.decode_pattern_digit
)


# ======================================================================
# Statistics / calibration output
# ======================================================================


STATS = {
    "digit_calls": 0,
    "adaptive_used": 0,
    "adaptive_override": 0,
    "fallback": 0,
}

LAST_CALIBRATION = None


# ======================================================================
# Wrapper arguments
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    # ----------------------------------------------------------
    # Tight-grid geometry
    # ----------------------------------------------------------

    parser.add_argument(
        "--segment-y-stretch",
        type=float,
        default=1.45,
        help=(
            "vertical stretch of RDS-200 segment masks "
            "(default: 1.45)"
        ),
    )

    parser.add_argument(
        "--segment-y-center",
        type=float,
        default=67.0,
        help=(
            "stretch center in normalized 65x130 digit coordinates "
            "(default: 67)"
        ),
    )

    parser.add_argument(
        "--segment-y-shift",
        type=float,
        default=0.0,
        help=(
            "additional vertical segment-mask shift "
            "(default: 0)"
        ),
    )

    # ----------------------------------------------------------
    # Self calibration
    # ----------------------------------------------------------

    parser.add_argument(
        "--adaptive-min-separation",
        type=float,
        default=5.0,
        help=(
            "minimum OFF/ON center separation for a calibrated "
            "segment (default: 5)"
        ),
    )

    parser.add_argument(
        "--adaptive-min-cluster-fraction",
        type=float,
        default=0.04,
        help=(
            "minimum fraction of samples in each OFF/ON cluster "
            "(default: 0.04)"
        ),
    )

    parser.add_argument(
        "--adaptive-min-reliable",
        type=int,
        default=4,
        help=(
            "minimum number of reliably calibrated segments needed "
            "for adaptive digit recognition (default: 4)"
        ),
    )

    parser.add_argument(
        "--adaptive-override-gap",
        type=float,
        default=0.055,
        help=(
            "minimum adaptive cost advantage required to override "
            "a different baseline digit (default: 0.055)"
        ),
    )

    parser.add_argument(
        "--adaptive-max-cost",
        type=float,
        default=0.24,
        help=(
            "maximum adaptive pattern cost for overriding baseline "
            "(default: 0.24)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Tight-grid segment geometry
# ======================================================================


def transform_polygon_y(
    polygon: np.ndarray,
    stretch: float,
    center: float,
    shift: float,
) -> np.ndarray:

    points = np.asarray(
        polygon,
        dtype=float,
    ).copy()

    points[:, 1] = (
        center
        + stretch
        * (
            points[:, 1]
            - center
        )
        + shift
    )

    # Digit patch is normalized to 65 x 130.
    points[:, 1] = np.clip(
        points[:, 1],
        0.0,
        129.0,
    )

    return np.rint(
        points
    ).astype(
        np.int32
    )


# ======================================================================
# Robust two-state calibration for ONE segment
# ======================================================================


def robust_mad(
    values: np.ndarray,
) -> float:

    if values.size == 0:

        return 0.0

    median = float(
        np.median(
            values
        )
    )

    mad = float(
        np.median(
            np.abs(
                values
                - median
            )
        )
    )

    return (
        1.4826
        * mad
    )


def fit_two_states(
    values: np.ndarray,
    minimum_separation: float,
    minimum_cluster_fraction: float,
) -> dict[str, float | int | bool]:

    finite = np.asarray(
        values,
        dtype=float,
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    result: dict[
        str,
        float | int | bool,
    ] = {
        "reliable": False,
        "off": float("nan"),
        "on": float("nan"),
        "threshold": float("nan"),
        "separation": 0.0,
        "quality": 0.0,
        "off_count": 0,
        "on_count": 0,
    }

    if finite.size < 20:

        return result

    # ----------------------------------------------------------
    # Trim extreme outliers.
    # ----------------------------------------------------------

    low_limit = float(
        np.percentile(
            finite,
            1.0,
        )
    )

    high_limit = float(
        np.percentile(
            finite,
            99.0,
        )
    )

    trimmed = finite[
        (
            finite
            >= low_limit
        )
        & (
            finite
            <= high_limit
        )
    ]

    if trimmed.size < 20:

        trimmed = finite

    # ----------------------------------------------------------
    # Robust 1D two-cluster iteration.
    # ----------------------------------------------------------

    center_low = float(
        np.percentile(
            trimmed,
            20.0,
        )
    )

    center_high = float(
        np.percentile(
            trimmed,
            80.0,
        )
    )

    if (
        center_high
        - center_low
        < 1.0
    ):

        return result

    for _ in range(
        40
    ):

        distance_low = np.abs(
            trimmed
            - center_low
        )

        distance_high = np.abs(
            trimmed
            - center_high
        )

        low_mask = (
            distance_low
            <= distance_high
        )

        high_mask = (
            ~low_mask
        )

        if (
            not np.any(
                low_mask
            )
            or not np.any(
                high_mask
            )
        ):

            return result

        new_low = float(
            np.median(
                trimmed[
                    low_mask
                ]
            )
        )

        new_high = float(
            np.median(
                trimmed[
                    high_mask
                ]
            )
        )

        if (
            new_low
            > new_high
        ):

            new_low, new_high = (
                new_high,
                new_low,
            )

        movement = (
            abs(
                new_low
                - center_low
            )
            + abs(
                new_high
                - center_high
            )
        )

        center_low = (
            new_low
        )

        center_high = (
            new_high
        )

        if movement < 1e-4:

            break

    # Reassign using final centers.
    distance_low = np.abs(
        trimmed
        - center_low
    )

    distance_high = np.abs(
        trimmed
        - center_high
    )

    low_cluster = trimmed[
        distance_low
        <= distance_high
    ]

    high_cluster = trimmed[
        distance_low
        > distance_high
    ]

    if (
        low_cluster.size == 0
        or high_cluster.size == 0
    ):

        return result

    minimum_count = max(
        5,
        int(
            math.ceil(
                minimum_cluster_fraction
                * trimmed.size
            )
        ),
    )

    if (
        low_cluster.size
        < minimum_count
        or high_cluster.size
        < minimum_count
    ):

        return result

    off_level = float(
        np.median(
            low_cluster
        )
    )

    on_level = float(
        np.median(
            high_cluster
        )
    )

    separation = (
        on_level
        - off_level
    )

    if (
        separation
        < minimum_separation
    ):

        return result

    off_spread = (
        robust_mad(
            low_cluster
        )
    )

    on_spread = (
        robust_mad(
            high_cluster
        )
    )

    total_spread = max(
        1.0,
        off_spread
        + on_spread,
    )

    quality = (
        separation
        / total_spread
    )

    # ----------------------------------------------------------
    # Use robust edges of the two populations when there is
    # a clean gap. Otherwise use center midpoint.
    # ----------------------------------------------------------

    off_upper = float(
        np.percentile(
            low_cluster,
            90.0,
        )
    )

    on_lower = float(
        np.percentile(
            high_cluster,
            10.0,
        )
    )

    if (
        on_lower
        > off_upper
    ):

        threshold = (
            0.5
            * (
                off_upper
                + on_lower
            )
        )

    else:

        threshold = (
            0.5
            * (
                off_level
                + on_level
            )
        )

    # Require at least modest bimodality.
    reliable = (
        quality
        >= 1.20
    )

    result.update(
        {
            "reliable": bool(
                reliable
            ),
            "off": off_level,
            "on": on_level,
            "threshold": threshold,
            "separation": separation,
            "quality": quality,
            "off_count": int(
                low_cluster.size
            ),
            "on_count": int(
                high_cluster.size
            ),
        }
    )

    return result


# ======================================================================
# Calibrate all positions and segments
# ======================================================================


def calibrate_segments(
    filtered: np.ndarray,
    minimum_separation: float,
    minimum_cluster_fraction: float,
):

    number_positions = (
        filtered.shape[1]
    )

    calibration = []

    for position in range(
        number_positions
    ):

        position_calibration = []

        for segment in range(
            7
        ):

            fitted = (
                fit_two_states(
                    filtered[
                        :,
                        position,
                        segment,
                    ],
                    minimum_separation,
                    minimum_cluster_fraction,
                )
            )

            position_calibration.append(
                fitted
            )

        calibration.append(
            position_calibration
        )

    return calibration


# ======================================================================
# Adaptive candidate costs
# ======================================================================


def adaptive_digit_costs(
    levels: np.ndarray,
    position: int,
    calibration,
    patterns,
) -> tuple[
    dict[int, float],
    int,
]:

    position_calibration = (
        calibration[
            position
        ]
    )

    probabilities = np.zeros(
        7,
        dtype=float,
    )

    weights = np.zeros(
        7,
        dtype=float,
    )

    reliable_count = 0

    for segment in range(
        7
    ):

        item = (
            position_calibration[
                segment
            ]
        )

        if not bool(
            item[
                "reliable"
            ]
        ):

            continue

        off_level = float(
            item[
                "off"
            ]
        )

        on_level = float(
            item[
                "on"
            ]
        )

        separation = max(
            1e-6,
            on_level
            - off_level,
        )

        probability = (
            (
                float(
                    levels[
                        segment
                    ]
                )
                - off_level
            )
            / separation
        )

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

        probabilities[
            segment
        ] = probability

        # Better-separated segments get somewhat larger weight,
        # but no single segment may dominate.
        weights[
            segment
        ] = float(
            np.clip(
                float(
                    item[
                        "quality"
                    ]
                ),
                1.0,
                3.0,
            )
        )

        reliable_count += 1

    costs: dict[
        int,
        float,
    ] = {}

    if (
        reliable_count == 0
    ):

        return (
            costs,
            0,
        )

    valid = (
        weights > 0.0
    )

    total_weight = float(
        np.sum(
            weights[
                valid
            ]
        )
    )

    if (
        total_weight <= 0.0
    ):

        return (
            costs,
            0,
        )

    for digit, pattern in (
        patterns.items()
    ):

        expected = np.asarray(
            pattern,
            dtype=float,
        )

        residual = (
            probabilities[
                valid
            ]
            - expected[
                valid
            ]
        )

        cost = float(
            np.sum(
                weights[
                    valid
                ]
                * residual
                * residual
            )
            / total_weight
        )

        costs[
            int(
                digit
            )
        ] = cost

    return (
        costs,
        reliable_count,
    )


# ======================================================================
# Adaptive digit decoder
# ======================================================================


def decode_adaptive_digit(
    levels: np.ndarray,
    position: int,
    calibration,
    patterns,
    minimum_reliable: int,
    override_gap: float,
    maximum_override_cost: float,
) -> tuple[
    int | None,
    float,
]:

    STATS[
        "digit_calls"
    ] += 1

    levels = np.asarray(
        levels,
        dtype=float,
    )

    baseline_digit, baseline_confidence = (
        ORIGINAL_PATTERN_DECODER(
            levels,
            patterns,
        )
    )

    if (
        levels.shape != (
            7,
        )
        or not np.all(
            np.isfinite(
                levels
            )
        )
    ):

        STATS[
            "fallback"
        ] += 1

        return (
            baseline_digit,
            baseline_confidence,
        )

    costs, reliable_count = (
        adaptive_digit_costs(
            levels,
            position,
            calibration,
            patterns,
        )
    )

    if (
        reliable_count
        < minimum_reliable
        or len(
            costs
        ) < 2
    ):

        STATS[
            "fallback"
        ] += 1

        return (
            baseline_digit,
            baseline_confidence,
        )

    order = sorted(
        costs.items(),
        key=lambda item: (
            item[
                1
            ]
        ),
    )

    adaptive_digit = int(
        order[
            0
        ][
            0
        ]
    )

    best_cost = float(
        order[
            0
        ][
            1
        ]
    )

    second_cost = float(
        order[
            1
        ][
            1
        ]
    )

    gap = (
        second_cost
        - best_cost
    )

    # Confidence from:
    #
    #   - absolute goodness of fit
    #   - distance to the second candidate
    #
    fit_confidence = float(
        np.clip(
            1.0
            - best_cost
            / 0.35,
            0.0,
            1.0,
        )
    )

    gap_confidence = float(
        np.clip(
            gap
            / 0.18,
            0.0,
            1.0,
        )
    )

    adaptive_confidence = (
        0.55
        * fit_confidence
        + 0.45
        * gap_confidence
    )

    # ----------------------------------------------------------
    # Same answer as original decoder:
    # use the stronger diagnostic confidence.
    # ----------------------------------------------------------

    if (
        baseline_digit
        == adaptive_digit
    ):

        STATS[
            "adaptive_used"
        ] += 1

        return (
            adaptive_digit,
            max(
                baseline_confidence,
                adaptive_confidence,
            ),
        )

    # ----------------------------------------------------------
    # Different answer:
    #
    # Override only when the self-calibrated pattern evidence is
    # genuinely clean.
    #
    # There is NO digit-specific condition here.
    # ----------------------------------------------------------

    if (
        best_cost
        <= maximum_override_cost
        and gap
        >= override_gap
    ):

        STATS[
            "adaptive_used"
        ] += 1

        STATS[
            "adaptive_override"
        ] += 1

        return (
            adaptive_digit,
            adaptive_confidence,
        )

    STATS[
        "fallback"
    ] += 1

    return (
        baseline_digit,
        baseline_confidence,
    )


# ======================================================================
# Adaptive decode_samples
# ======================================================================


def make_adaptive_decode_samples(
    wrapper_args: argparse.Namespace,
):

    def adaptive_decode_samples(
        samples,
        profile,
        filter_window,
        decimal_places_override=None,
        minimum_confidence=0.0,
        decimal_switch_penalty=4.0,
    ):

        global LAST_CALIBRATION

        # Keep other profiles untouched.
        if (
            profile.name
            != "rds200"
        ):

            return ORIGINAL_DECODE_SAMPLES(
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
                    decimal_switch_penalty
                ),
            )

        darkness = np.full(
            (
                len(samples),
                len(
                    profile.digit_boxes
                ),
                7,
            ),
            np.nan,
            dtype=float,
        )

        for index, sample in enumerate(
            samples
        ):

            if (
                sample.darkness
                is not None
            ):

                darkness[
                    index
                ] = np.asarray(
                    sample.darkness,
                    dtype=float,
                )

        filtered = (
            core.temporal_filter(
                darkness,
                filter_window,
                profile.temporal_filter,
            )
        )

        patterns = (
            profile.digit_patterns
            or core.STANDARD_DIGIT_PATTERNS
        )

        calibration = (
            calibrate_segments(
                filtered,
                wrapper_args.adaptive_min_separation,
                wrapper_args.adaptive_min_cluster_fraction,
            )
        )

        LAST_CALIBRATION = (
            calibration
        )

        integer_values: list[
            int | None
        ] = []

        sample_confidences: list[
            float
        ] = []

        # ------------------------------------------------------
        # Decode every sample.
        # ------------------------------------------------------

        for index in range(
            len(samples)
        ):

            if np.isnan(
                filtered[
                    index
                ]
            ).all():

                integer_values.append(
                    None
                )

                sample_confidences.append(
                    0.0
                )

                continue

            digits = []

            confidences = []

            for position, levels in enumerate(
                filtered[
                    index
                ]
            ):

                digit, confidence = (
                    decode_adaptive_digit(
                        levels,
                        position,
                        calibration,
                        patterns,
                        wrapper_args.adaptive_min_reliable,
                        wrapper_args.adaptive_override_gap,
                        wrapper_args.adaptive_max_cost,
                    )
                )

                if (
                    digit is None
                ):

                    digits = []

                    confidences.append(
                        confidence
                    )

                    break

                digits.append(
                    int(
                        digit
                    )
                )

                confidences.append(
                    confidence
                )

            confidence = min(
                confidences,
                default=0.0,
            )

            if (
                not digits
                or confidence
                < minimum_confidence
            ):

                integer_values.append(
                    None
                )

                sample_confidences.append(
                    confidence
                )

                continue

            integer_value = 0

            for digit in digits:

                integer_value = (
                    10
                    * integer_value
                    + digit
                )

            integer_values.append(
                integer_value
            )

            sample_confidences.append(
                confidence
            )

        # ------------------------------------------------------
        # Decimal position: preserve original implementation.
        # ------------------------------------------------------

        filtered_decimal_scores = (
            None
        )

        if (
            profile.decimal_candidates
            and decimal_places_override
            is None
        ):

            decimal_scores = np.full(
                (
                    len(samples),
                    len(
                        profile.decimal_candidates
                    ),
                ),
                np.nan,
                dtype=float,
            )

            for index, sample in enumerate(
                samples
            ):

                if (
                    sample.decimal_scores
                    is not None
                ):

                    decimal_scores[
                        index
                    ] = np.asarray(
                        sample.decimal_scores,
                        dtype=float,
                    )

            filtered_decimal_scores = (
                core.temporal_filter(
                    decimal_scores,
                    filter_window,
                    "median",
                )
            )

        if (
            decimal_places_override
            is not None
        ):

            decimal_places_sequence = [
                decimal_places_override
            ] * len(
                samples
            )

        elif (
            filtered_decimal_scores
            is not None
        ):

            decimal_places_sequence = (
                core.decode_decimal_places_sequence(
                    filtered_decimal_scores,
                    integer_values,
                    profile,
                    switch_penalty=(
                        decimal_switch_penalty
                    ),
                )
            )

        else:

            decimal_places_sequence = [
                profile.decimal_places
            ] * len(
                samples
            )

        decoded = []

        for (
            sample,
            integer_value,
            confidence,
            decimal_places,
        ) in zip(
            samples,
            integer_values,
            sample_confidences,
            decimal_places_sequence,
        ):

            if (
                integer_value
                is None
            ):

                decoded.append(
                    core.DecodedSample(
                        sample.time_s,
                        None,
                        confidence,
                    )
                )

            else:

                decoded.append(
                    core.DecodedSample(
                        sample.time_s,
                        integer_value
                        / (
                            10
                            ** decimal_places
                        ),
                        confidence,
                    )
                )

        return decoded

    return (
        adaptive_decode_samples
    )


# ======================================================================
# Patch all imported aliases of decode_samples
# ======================================================================


def install_decode_hook(
    replacement,
):

    patched = []

    for module in (
        core,
        roi_app,
        fixed_app,
        flow,
    ):

        if not hasattr(
            module,
            "decode_samples",
        ):

            continue

        current = getattr(
            module,
            "decode_samples",
        )

        if (
            current
            is ORIGINAL_DECODE_SAMPLES
        ):

            setattr(
                module,
                "decode_samples",
                replacement,
            )

            patched.append(
                (
                    module,
                    current,
                )
            )

    return patched


def restore_decode_hooks(
    patched,
) -> None:

    for module, original in (
        patched
    ):

        setattr(
            module,
            "decode_samples",
            original,
        )


# ======================================================================
# Print calibration table
# ======================================================================


def print_calibration() -> None:

    if (
        LAST_CALIBRATION
        is None
    ):

        return

    segment_names = (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    )

    print()

    print(
        "Adaptive RDS-200 segment calibration:"
    )

    for position, items in enumerate(
        LAST_CALIBRATION,
        start=1,
    ):

        print()

        print(
            f"  digit position {position}:"
        )

        for name, item in zip(
            segment_names,
            items,
        ):

            if bool(
                item[
                    "reliable"
                ]
            ):

                print(
                    (
                        f"    {name}: "
                        f"OFF={float(item['off']):6.2f} "
                        f"ON={float(item['on']):6.2f} "
                        f"thr={float(item['threshold']):6.2f} "
                        f"sep={float(item['separation']):6.2f} "
                        f"q={float(item['quality']):5.2f} "
                        f"n={int(item['off_count'])}/"
                        f"{int(item['on_count'])}"
                    )
                )

            else:

                print(
                    (
                        f"    {name}: "
                        "not reliably bimodal -> baseline fallback"
                    )
                )


# ======================================================================
# Main
# ======================================================================


def main() -> int:

    STATS[
        "digit_calls"
    ] = 0

    STATS[
        "adaptive_used"
    ] = 0

    STATS[
        "adaptive_override"
    ] = 0

    STATS[
        "fallback"
    ] = 0

    wrapper_args, remaining = (
        parse_wrapper_args(
            sys.argv[1:]
        )
    )

    if (
        wrapper_args.segment_y_stretch
        <= 0.0
    ):

        print(
            "Error: --segment-y-stretch must be positive.",
            file=sys.stderr,
        )

        return 1

    if not (
        0.0
        < wrapper_args.adaptive_min_cluster_fraction
        < 0.5
    ):

        print(
            (
                "Error: --adaptive-min-cluster-fraction "
                "must lie between 0 and 0.5."
            ),
            file=sys.stderr,
        )

        return 1

    # ----------------------------------------------------------
    # Install tight-grid RDS-200 segment geometry.
    # ----------------------------------------------------------

    profile = (
        core.PROFILES[
            "rds200"
        ]
    )

    polygons = (
        profile.segment_polygons
    )

    original_polygons = {
        name: polygon.copy()
        for name, polygon
        in polygons.items()
    }

    transformed_polygons = {
        name: transform_polygon_y(
            polygon,
            wrapper_args.segment_y_stretch,
            wrapper_args.segment_y_center,
            wrapper_args.segment_y_shift,
        )
        for name, polygon
        in original_polygons.items()
    }

    polygons.clear()

    polygons.update(
        transformed_polygons
    )

    print(
        "RDS-200 adaptive self-calibrating decoder:"
    )

    print(
        (
            f"  segment y stretch       : "
            f"{wrapper_args.segment_y_stretch:.3f}"
        )
    )

    print(
        (
            f"  minimum ON/OFF separation: "
            f"{wrapper_args.adaptive_min_separation:.2f}"
        )
    )

    print(
        (
            f"  minimum reliable segments: "
            f"{wrapper_args.adaptive_min_reliable}"
        )
    )

    print(
        "  digit-specific fixes     : NONE"
    )

    replacement = (
        make_adaptive_decode_samples(
            wrapper_args
        )
    )

    patched = (
        install_decode_hook(
            replacement
        )
    )

    saved_argv = (
        sys.argv
    )

    # Remove wrapper-only arguments before the ordinary pipeline
    # parses the command line.
    sys.argv = [
        saved_argv[0],
        *remaining,
    ]

    try:

        result = (
            flow.main()
        )

    finally:

        sys.argv = (
            saved_argv
        )

        restore_decode_hooks(
            patched
        )

        polygons.clear()

        polygons.update(
            original_polygons
        )

    print_calibration()

    print()

    print(
        "Adaptive decoder diagnostics:"
    )

    print(
        (
            f"  patched decode aliases : "
            f"{len(patched)}"
        )
    )

    print(
        (
            f"  digit calls            : "
            f"{STATS['digit_calls']}"
        )
    )

    print(
        (
            f"  adaptive result used   : "
            f"{STATS['adaptive_used']}"
        )
    )

    print(
        (
            f"  adaptive overrides     : "
            f"{STATS['adaptive_override']}"
        )
    )

    print(
        (
            f"  baseline fallbacks     : "
            f"{STATS['fallback']}"
        )
    )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )