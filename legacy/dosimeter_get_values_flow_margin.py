#!/usr/bin/env python3
"""
RDS-200 optical-flow decoder with active/inactive segment-margin
classification.

Required existing files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py

This wrapper keeps the COMPLETE optical-flow / geometry pipeline from
dosimeter_get_values_flow.py unchanged.

The ONLY change is the RDS-200 digit classifier.

Instead of primarily fitting absolute seven-segment darkness values,
each candidate digit is judged by how well its expected ACTIVE segments
can be separated from its expected INACTIVE (ghost) segments.

This is intended for LCD video where all seven physical segments may
remain faintly visible.

All command-line arguments are passed unchanged to
dosimeter_get_values_flow.py.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_flow as flow


# ======================================================================
# Keep original decoder as conservative fallback
# ======================================================================

ORIGINAL_DECODER = (
    core.decode_pattern_digit
)


# ======================================================================
# Diagnostics
# ======================================================================

STATS = {
    "calls": 0,
    "margin": 0,
    "fallback": 0,
    "rejected": 0,
}


# ======================================================================
# Helpers
# ======================================================================


def logistic(
    value: float,
) -> float:

    # Avoid overflow for extreme values.
    value = max(
        -30.0,
        min(
            30.0,
            value,
        ),
    )

    return (
        1.0
        / (
            1.0
            + math.exp(
                -value
            )
        )
    )


# ======================================================================
# RDS-200 margin decoder
# ======================================================================


def decode_rds200_margin_digit(
    darkness: np.ndarray,
    patterns: dict[
        int,
        tuple[int, ...],
    ],
) -> tuple[
    int | None,
    float,
]:
    """
    Decode one seven-segment digit from local segment darkness.

    darkness:

        [a, b, c, d, e, f, g]

    Larger value = segment is darker / more likely active.

    For a candidate digit with both active and inactive segments:

        margin =
            weakest expected-active segment
            -
            strongest expected-inactive segment

    Positive margin therefore means that ONE threshold exists which
    perfectly separates the candidate's active bars from ghost bars.

    This is much more appropriate for a ghosty LCD than relying mainly
    on absolute intensity.
    """

    STATS[
        "calls"
    ] += 1

    levels = np.asarray(
        darkness,
        dtype=float,
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
            "rejected"
        ] += 1

        return (
            None,
            0.0,
        )

    # Almost no usable segment evidence.
    if float(
        np.max(
            levels
        )
    ) < 4.0:

        STATS[
            "rejected"
        ] += 1

        return (
            None,
            0.0,
        )

    candidates = []

    for (
        digit,
        pattern,
    ) in patterns.items():

        active_mask = np.asarray(
            pattern,
            dtype=bool,
        )

        active = (
            levels[
                active_mask
            ]
        )

        inactive = (
            levels[
                ~active_mask
            ]
        )

        if (
            active.size == 0
        ):

            continue

        weakest_active = float(
            np.min(
                active
            )
        )

        median_active = float(
            np.median(
                active
            )
        )

        # ------------------------------------------------------
        # Ordinary digits: active vs inactive separation
        # ------------------------------------------------------

        if (
            inactive.size > 0
        ):

            strongest_inactive = float(
                np.max(
                    inactive
                )
            )

            median_inactive = float(
                np.median(
                    inactive
                )
            )

            margin = (
                weakest_active
                - strongest_inactive
            )

            contrast = (
                median_active
                - median_inactive
            )

            # Margin is the primary quantity.
            # Median contrast is only a weak tie-breaker.
            score = (
                margin
                + 0.12
                * contrast
            )

        # ------------------------------------------------------
        # Digit 8 has no inactive bars.
        #
        # A faint ghost pattern must NOT become an 8 merely because
        # every physical segment is visible.
        # ------------------------------------------------------

        else:

            absolute_on_level = (
                12.0
            )

            margin = (
                weakest_active
                - absolute_on_level
            )

            contrast = (
                median_active
                - absolute_on_level
            )

            score = (
                margin
                + 0.05
                * contrast
            )

        candidates.append(
            (
                float(
                    score
                ),
                float(
                    margin
                ),
                int(
                    digit
                ),
                float(
                    contrast
                ),
            )
        )

    if (
        len(
            candidates
        ) < 2
    ):

        STATS[
            "rejected"
        ] += 1

        return (
            None,
            0.0,
        )

    candidates.sort(
        reverse=True
    )

    best = (
        candidates[
            0
        ]
    )

    second = (
        candidates[
            1
        ]
    )

    (
        best_score,
        best_margin,
        best_digit,
        best_contrast,
    ) = best

    score_gap = (
        best_score
        - second[
            0
        ]
    )

    # ----------------------------------------------------------
    # If there is no remotely plausible active/inactive separation,
    # retain the existing continuous decoder as fallback.
    # ----------------------------------------------------------

    if (
        best_margin < -4.0
    ):

        STATS[
            "fallback"
        ] += 1

        return ORIGINAL_DECODER(
            levels,
            patterns,
        )

    # ----------------------------------------------------------
    # Confidence
    #
    # Two independent pieces of evidence:
    #
    #   1. active/inactive margin
    #   2. separation from second-best digit
    # ----------------------------------------------------------

    margin_confidence = (
        logistic(
            best_margin
            / 2.5
        )
    )

    gap_confidence = (
        logistic(
            score_gap
            / 1.5
        )
    )

    confidence = (
        0.58
        * margin_confidence
        + 0.42
        * gap_confidence
    )

    # Negative margin means there is some overlap between ghost and
    # active bars.  We may still have the correct best candidate,
    # but confidence must reflect that ambiguity.
    if (
        best_margin < 0.0
    ):

        confidence *= (
            0.55
        )

    # Also penalize very weak overall active contrast.
    if (
        best_contrast < 4.0
    ):

        confidence *= max(
            0.25,
            (
                best_contrast
                + 4.0
            )
            / 8.0,
        )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    STATS[
        "margin"
    ] += 1

    return (
        best_digit,
        confidence,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> int:

    # Monkey-patch ONLY the RDS-200 pattern decoder.
    #
    # Everything else, especially the optical-flow tracking and cached
    # debug images, remains exactly as in dosimeter_get_values_flow.py.
    core.decode_pattern_digit = (
        decode_rds200_margin_digit
    )

    try:

        result = (
            flow.main()
        )

    finally:

        core.decode_pattern_digit = (
            ORIGINAL_DECODER
        )

    print()

    print(
        "RDS-200 margin decoder diagnostics:"
    )

    print(
        (
            f"  digit calls : "
            f"{STATS['calls']}"
        )
    )

    print(
        (
            f"  margin used : "
            f"{STATS['margin']}"
        )
    )

    print(
        (
            f"  fallback    : "
            f"{STATS['fallback']}"
        )
    )

    print(
        (
            f"  rejected    : "
            f"{STATS['rejected']}"
        )
    )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )