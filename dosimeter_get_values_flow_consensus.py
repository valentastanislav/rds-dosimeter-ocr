#!/usr/bin/env python3
"""
General RDS-200 optical-flow decoder with binary-pattern consensus.

This version keeps:

    * optical-flow stabilization
    * fixed perspective correction
    * fixed digit grid
    * tight-grid vertical segment stretch
    * original continuous least-squares RDS-200 decoder
    * original five-frame median filtering of segment darkness

It does NOT contain digit-specific corrections such as 6 -> 8.

Instead it estimates one OFF-segment population from the complete video
and derives an activation threshold from that population.

For every digit:

    1. run the original continuous decoder;
    2. threshold the seven segment darkness values;
    3. if the resulting seven-bit pattern exactly matches a standard
       digit 0..9 and the segments lie sufficiently far from the
       threshold, use that binary digit;
    4. otherwise keep the original decoder result.

This allows generic corrections such as:

    6 -> 8
    8 -> 6
    2 -> 1
    1 -> 2
    ...

without knowing the expected displayed value.

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
from dataclasses import dataclass, replace

import numpy as np

import dosimeter_get_values as core
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
# Runtime diagnostics
# ======================================================================


@dataclass
class ConsensusDiagnostics:
    digit_calls: int = 0
    binary_exact: int = 0
    binary_agreed: int = 0
    binary_overrides: int = 0
    binary_ambiguous: int = 0
    baseline_used: int = 0
    calibration_reliable: bool = False
    off_level: float = float("nan")
    off_spread: float = float("nan")
    on_level: float = float("nan")
    separation: float = float("nan")
    threshold: float = float("nan")
    low_count: int = 0
    high_count: int = 0


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

    # ----------------------------------------------------------
    # Segment geometry already demonstrated to work with the
    # manually selected tight grid.
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
            "vertical stretch center in normalized 65x130 "
            "digit coordinates (default: 67)"
        ),
    )

    parser.add_argument(
        "--segment-y-shift",
        type=float,
        default=0.0,
        help=(
            "additional vertical mask shift "
            "(default: 0)"
        ),
    )

    # ----------------------------------------------------------
    # Video-wide OFF calibration
    # ----------------------------------------------------------

    parser.add_argument(
        "--binary-min-separation",
        type=float,
        default=6.0,
        help=(
            "minimum separation between low and high darkness "
            "populations before automatic binary decoding is enabled "
            "(default: 6)"
        ),
    )

    parser.add_argument(
        "--binary-off-margin",
        type=float,
        default=4.0,
        help=(
            "minimum threshold distance above the estimated OFF level "
            "(default: 4)"
        ),
    )

    parser.add_argument(
        "--binary-off-sigma",
        type=float,
        default=3.5,
        help=(
            "threshold offset in robust OFF-cluster sigma "
            "(default: 3.5)"
        ),
    )

    parser.add_argument(
        "--binary-threshold-fraction",
        type=float,
        default=0.30,
        help=(
            "maximum threshold position as fraction of OFF-to-ON "
            "separation (default: 0.30)"
        ),
    )

    parser.add_argument(
        "--binary-override-margin",
        type=float,
        default=1.5,
        help=(
            "minimum distance of EVERY segment from threshold before "
            "binary pattern may override the baseline decoder "
            "(default: 1.5)"
        ),
    )

    parser.add_argument(
        "--binary-agree-margin",
        type=float,
        default=0.5,
        help=(
            "minimum segment margin when binary and baseline decoders "
            "already agree (default: 0.5)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Tight-grid geometry
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
# Robust statistics
# ======================================================================


def robust_sigma(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

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


# ======================================================================
# Find low/high populations in ALL segment measurements
# ======================================================================


def calibrate_binary_threshold(
    filtered: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int | bool]:

    values = np.asarray(
        filtered,
        dtype=float,
    ).reshape(
        -1
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    result: dict[
        str,
        float | int | bool,
    ] = {
        "reliable": False,
        "off_level": float("nan"),
        "off_spread": float("nan"),
        "on_level": float("nan"),
        "separation": float("nan"),
        "threshold": float("nan"),
        "low_count": 0,
        "high_count": 0,
    }

    if values.size < 100:

        return result

    # ----------------------------------------------------------
    # Remove only extreme outliers.
    # ----------------------------------------------------------

    lower_limit = float(
        np.percentile(
            values,
            0.5,
        )
    )

    upper_limit = float(
        np.percentile(
            values,
            99.5,
        )
    )

    values = values[
        (
            values
            >= lower_limit
        )
        & (
            values
            <= upper_limit
        )
    ]

    if values.size < 100:

        return result

    # ----------------------------------------------------------
    # Robust 1D two-cluster fit.
    #
    # This is intentionally GLOBAL over every digit position and
    # every segment.  A segment does not have to switch between
    # ON and OFF during the video.
    # ----------------------------------------------------------

    low_center = float(
        np.percentile(
            values,
            15.0,
        )
    )

    high_center = float(
        np.percentile(
            values,
            75.0,
        )
    )

    if (
        high_center
        <= low_center
    ):

        return result

    for _ in range(
        50
    ):

        distance_low = np.abs(
            values
            - low_center
        )

        distance_high = np.abs(
            values
            - high_center
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
                values[
                    low_mask
                ]
            )
        )

        new_high = float(
            np.median(
                values[
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
                - low_center
            )
            + abs(
                new_high
                - high_center
            )
        )

        low_center = (
            new_low
        )

        high_center = (
            new_high
        )

        if movement < 1e-4:

            break

    distance_low = np.abs(
        values
        - low_center
    )

    distance_high = np.abs(
        values
        - high_center
    )

    low_cluster = values[
        distance_low
        <= distance_high
    ]

    high_cluster = values[
        distance_low
        > distance_high
    ]

    if (
        low_cluster.size < 20
        or high_cluster.size < 20
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

    off_spread = (
        robust_sigma(
            low_cluster
        )
    )

    separation = (
        on_level
        - off_level
    )

    if (
        separation
        < args.binary_min_separation
    ):

        return result

    # ----------------------------------------------------------
    # Crucial point:
    #
    # Do NOT put threshold halfway between OFF and ON.
    #
    # Some genuine active RDS-200 segments are much weaker than
    # others.  We only need to get safely outside the OFF
    # population.
    # ----------------------------------------------------------

    noise_threshold = (
        off_level
        + max(
            args.binary_off_margin,
            args.binary_off_sigma
            * max(
                off_spread,
                0.5,
            ),
        )
    )

    separation_threshold = (
        off_level
        + args.binary_threshold_fraction
        * separation
    )

    threshold = min(
        noise_threshold,
        separation_threshold,
    )

    # Never allow it to sit essentially on the OFF cluster.
    threshold = max(
        threshold,
        off_level
        + 2.0,
    )

    result.update(
        {
            "reliable": True,
            "off_level": off_level,
            "off_spread": off_spread,
            "on_level": on_level,
            "separation": separation,
            "threshold": threshold,
            "low_count": int(
                low_cluster.size
            ),
            "high_count": int(
                high_cluster.size
            ),
        }
    )

    return result


# ======================================================================
# Generic binary-pattern classifier
# ======================================================================


def binary_pattern_digit(
    levels: np.ndarray,
    patterns: dict[
        int,
        tuple[int, ...],
    ],
    threshold: float,
) -> tuple[
    int | None,
    float,
]:
    """
    Return:

        exact binary digit or None,
        minimum distance of any segment from threshold.

    The margin is positive by construction when an exact pattern exists.
    """

    values = np.asarray(
        levels,
        dtype=float,
    )

    if (
        values.shape != (
            7,
        )
        or not np.all(
            np.isfinite(
                values
            )
        )
    ):

        return (
            None,
            float(
                "-inf"
            ),
        )

    observed = tuple(
        (
            values
            >= threshold
        ).astype(
            int
        )
    )

    pattern_to_digit = {
        tuple(
            pattern
        ): int(
            digit
        )
        for digit, pattern
        in patterns.items()
    }

    digit = (
        pattern_to_digit.get(
            observed
        )
    )

    if digit is None:

        return (
            None,
            float(
                "-inf"
            ),
        )

    pattern = np.asarray(
        patterns[
            digit
        ],
        dtype=bool,
    )

    active_values = (
        values[
            pattern
        ]
    )

    inactive_values = (
        values[
            ~pattern
        ]
    )

    if (
        active_values.size
        == 0
    ):

        return (
            None,
            float(
                "-inf"
            ),
        )

    active_margin = float(
        np.min(
            active_values
            - threshold
        )
    )

    if (
        inactive_values.size
        > 0
    ):

        inactive_margin = float(
            np.min(
                threshold
                - inactive_values
            )
        )

        margin = min(
            active_margin,
            inactive_margin,
        )

    else:

        # Digit 8: every segment is active.
        # There is no inactive-side margin.
        margin = (
            active_margin
        )

    return (
        digit,
        margin,
    )


# ======================================================================
# Consensus decoder
# ======================================================================


def decode_consensus_digit(
    levels: np.ndarray,
    patterns,
    threshold: float,
    calibration_reliable: bool,
    args: argparse.Namespace,
    diagnostics: ConsensusDiagnostics,
) -> tuple[
    int | None,
    float,
]:

    diagnostics.digit_calls += 1

    (
        baseline_digit,
        baseline_confidence,
    ) = ORIGINAL_PATTERN_DECODER(
        levels,
        patterns,
    )

    if not calibration_reliable:

        diagnostics.baseline_used += 1

        return (
            baseline_digit,
            baseline_confidence,
        )

    (
        binary_digit,
        margin,
    ) = binary_pattern_digit(
        levels,
        patterns,
        threshold,
    )

    if (
        binary_digit
        is None
    ):

        diagnostics.binary_ambiguous += 1

        diagnostics.baseline_used += 1

        return (
            baseline_digit,
            baseline_confidence,
        )

    diagnostics.binary_exact += 1

    # Convert geometric distance from threshold to diagnostic
    # confidence.  This does NOT depend on digit identity.
    binary_confidence = float(
        np.clip(
            0.55
            + 0.06
            * margin,
            0.0,
            0.95,
        )
    )

    # ----------------------------------------------------------
    # Both classifiers agree.
    # ----------------------------------------------------------

    if (
        binary_digit
        == baseline_digit
    ):

        if (
            margin
            >= args.binary_agree_margin
        ):

            diagnostics.binary_agreed += 1

            return (
                binary_digit,
                max(
                    baseline_confidence,
                    binary_confidence,
                ),
            )

        diagnostics.baseline_used += 1

        return (
            baseline_digit,
            baseline_confidence,
        )

    # ----------------------------------------------------------
    # Classifiers disagree.
    #
    # An exact seven-bit pattern may override the continuous
    # decoder ONLY if every active and inactive segment has a
    # comfortable distance from the automatically calibrated
    # threshold.
    #
    # No digit-specific exceptions exist here.
    # ----------------------------------------------------------

    if (
        margin
        >= args.binary_override_margin
    ):

        diagnostics.binary_overrides += 1

        return (
            binary_digit,
            binary_confidence,
        )

    diagnostics.binary_ambiguous += 1

    diagnostics.baseline_used += 1

    return (
        baseline_digit,
        baseline_confidence,
    )


# ======================================================================
# Consensus decode_samples()
# ======================================================================


def make_consensus_decode_samples(
    args: argparse.Namespace,
    diagnostics: ConsensusDiagnostics,
):

    def consensus_decode_samples(
        samples,
        profile,
        filter_window,
        decimal_places_override=None,
        minimum_confidence=0.0,
        decimal_switch_penalty=4.0,
        decimal_sequence_observer: (
            core.DecimalSequenceObserver | None
        ) = None,
    ):

        # Other dosimeter profiles keep their original decoder.
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
                decimal_sequence_observer=(
                    decimal_sequence_observer
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

        # Preserve the original RDS200 temporal segment filter.
        filtered = (
            core.temporal_filter(
                darkness,
                filter_window,
                profile.temporal_filter,
            )
        )

        # ------------------------------------------------------
        # ONE video-wide OFF calibration.
        # ------------------------------------------------------

        calibration = (
            calibrate_binary_threshold(
                filtered,
                args,
            )
        )

        diagnostics.calibration_reliable = (
            calibration[
                "reliable"
            ]
        )
        diagnostics.off_level = (
            calibration[
                "off_level"
            ]
        )
        diagnostics.off_spread = (
            calibration[
                "off_spread"
            ]
        )
        diagnostics.on_level = (
            calibration[
                "on_level"
            ]
        )
        diagnostics.separation = (
            calibration[
                "separation"
            ]
        )
        diagnostics.threshold = (
            calibration[
                "threshold"
            ]
        )
        diagnostics.low_count = (
            calibration[
                "low_count"
            ]
        )
        diagnostics.high_count = (
            calibration[
                "high_count"
            ]
        )

        patterns = (
            profile.digit_patterns
            or core.STANDARD_DIGIT_PATTERNS
        )

        integer_values = []

        sample_confidences = []

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

            for levels in (
                filtered[
                    index
                ]
            ):

                (
                    digit,
                    confidence,
                ) = (
                    decode_consensus_digit(
                        levels,
                        patterns,
                        float(
                            calibration[
                                "threshold"
                            ]
                        ),
                        bool(
                            calibration[
                                "reliable"
                            ]
                        ),
                        args,
                        diagnostics,
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
        # Decimal-point handling copied from the original logic.
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

        if (
            decimal_sequence_observer
            is not None
        ):

            decimal_sequence_observer(
                list(
                    decimal_places_sequence
                )
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

    return consensus_decode_samples


# ======================================================================
# Main
# ======================================================================


def main() -> int:

    diagnostics = (
        ConsensusDiagnostics()
    )

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
        < wrapper_args.binary_threshold_fraction
        < 1.0
    ):

        print(
            (
                "Error: --binary-threshold-fraction must "
                "lie between 0 and 1."
            ),
            file=sys.stderr,
        )

        return 1

    # ----------------------------------------------------------
    # Construct the tight-grid geometry.
    # ----------------------------------------------------------

    original_profile = (
        core.PROFILES[
            "rds200"
        ]
    )

    original_polygons = {
        name: polygon.copy()
        for name, polygon
        in original_profile.segment_polygons.items()
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

    tight_profile = replace(
        original_profile,
        segment_polygons=(
            transformed_polygons
        ),
    )

    consensus_decoder = (
        make_consensus_decode_samples(
            wrapper_args,
            diagnostics,
        )
    )

    print(
        "RDS-200 generic binary-consensus decoder:"
    )

    print(
        (
            f"  segment y stretch   : "
            f"{wrapper_args.segment_y_stretch:.3f}"
        )
    )

    print(
        "  digit-specific rules: NONE"
    )

    print(
        "  expected values used: NONE"
    )

    result = (
        flow.main(
            remaining,
            decode_samples=(
                consensus_decoder
            ),
            profile_override=(
                tight_profile
            ),
        )
    )

    print()

    print(
        "Binary OFF calibration:"
    )

    if bool(
        diagnostics.calibration_reliable
    ):

        print(
            (
                f"  OFF level       : "
                f"{float(diagnostics.off_level):.3f}"
            )
        )

        print(
            (
                f"  OFF robust sigma: "
                f"{float(diagnostics.off_spread):.3f}"
            )
        )

        print(
            (
                f"  high population : "
                f"{float(diagnostics.on_level):.3f}"
            )
        )

        print(
            (
                f"  separation      : "
                f"{float(diagnostics.separation):.3f}"
            )
        )

        print(
            (
                f"  ACTIVE threshold: "
                f"{float(diagnostics.threshold):.3f}"
            )
        )

        print(
            (
                f"  cluster sizes   : "
                f"{int(diagnostics.low_count)} / "
                f"{int(diagnostics.high_count)}"
            )
        )

    else:

        print(
            (
                "  automatic OFF calibration was not reliable; "
                "original decoder was used."
            )
        )

    print()

    print(
        "Binary-consensus diagnostics:"
    )

    print(
        "  decoder injection : explicit"
    )

    print(
        (
            f"  digit calls       : "
            f"{diagnostics.digit_calls}"
        )
    )

    print(
        (
            f"  exact patterns    : "
            f"{diagnostics.binary_exact}"
        )
    )

    print(
        (
            f"  agreement used    : "
            f"{diagnostics.binary_agreed}"
        )
    )

    print(
        (
            f"  generic overrides : "
            f"{diagnostics.binary_overrides}"
        )
    )

    print(
        (
            f"  ambiguous binary  : "
            f"{diagnostics.binary_ambiguous}"
        )
    )

    print(
        (
            f"  baseline used     : "
            f"{diagnostics.baseline_used}"
        )
    )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
