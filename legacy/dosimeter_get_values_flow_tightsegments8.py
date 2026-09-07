#!/usr/bin/env python3
"""
RDS-200 flow + tight segment geometry + conservative 6 -> 8 correction.

Required existing files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py
    dosimeter_get_values_flow_diag.py
    dosimeter_get_values_flow_tightsegments.py

This wrapper changes ONLY one decoder decision:

    baseline digit == 6

may be corrected to

    digit == 8

when ALL seven measured segments are clearly active.

Why:
For a true seven-segment 8 all seven segments must be active.
In the present RDS-200 video a real 8 is sometimes classified by the
continuous pattern decoder as 6 because the upper-right segment b is
weaker than the other segments.

The correction is deliberately narrow:

    6 -> 8 only

It does not alter 0,1,2,3,4,5,7,9 decisions.

All geometry, optical flow, temporal filtering, decimal handling and
tight-segment masks remain unchanged.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_flow_tightsegments as tight


# ======================================================================
# Original decoder
# ======================================================================


ORIGINAL_DECODER = (
    core.decode_pattern_digit
)


# ======================================================================
# Statistics
# ======================================================================


STATS = {
    "calls": 0,
    "baseline_6": 0,
    "corrected_6_to_8": 0,
}


# ======================================================================
# Wrapper command-line arguments
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--eight-min-segment",
        type=float,
        default=10.0,
        help=(
            "minimum darkness required on EACH of the seven segments "
            "before baseline 6 may be corrected to 8 "
            "(default: 10.0)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Decoder factory
# ======================================================================


def make_decoder(
    minimum_eight_segment: float,
):

    def decode_pattern_digit_with_eight_fix(
        darkness: np.ndarray,
        patterns: dict[
            int,
            tuple[int, ...],
        ],
    ) -> tuple[
        int | None,
        float,
    ]:

        STATS[
            "calls"
        ] += 1

        # ----------------------------------------------------------
        # Run the normal RDS-200 decoder first.
        # ----------------------------------------------------------

        (
            baseline_digit,
            baseline_confidence,
        ) = ORIGINAL_DECODER(
            darkness,
            patterns,
        )

        if (
            baseline_digit == 6
        ):

            STATS[
                "baseline_6"
            ] += 1

        # ----------------------------------------------------------
        # Correction is intentionally ONLY 6 -> 8.
        # ----------------------------------------------------------

        if (
            baseline_digit != 6
        ):

            return (
                baseline_digit,
                baseline_confidence,
            )

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

            return (
                baseline_digit,
                baseline_confidence,
            )

        # ----------------------------------------------------------
        # Verify that the configured digit-8 pattern really means
        # all seven bars active.
        # ----------------------------------------------------------

        pattern_8 = (
            patterns.get(
                8
            )
        )

        if (
            pattern_8 is None
        ):

            return (
                baseline_digit,
                baseline_confidence,
            )

        mask_8 = np.asarray(
            pattern_8,
            dtype=bool,
        )

        if (
            mask_8.shape != (
                7,
            )
            or not np.all(
                mask_8
            )
        ):

            return (
                baseline_digit,
                baseline_confidence,
            )

        # ----------------------------------------------------------
        # A genuine 8 must have ALL seven bars active.
        #
        # For our problematic frame around 20.4 s we measured:
        #
        #   a ~= 37
        #   b ~= 13
        #   c ~= 13
        #   d ~= 33
        #   e ~= 45
        #   f ~= 47
        #   g ~= 38
        #
        # so the weakest active bar is still clearly above 10.
        #
        # A normal 6 has upper-right segment b near the inactive
        # background, so it will NOT satisfy this condition.
        # ----------------------------------------------------------

        minimum_level = float(
            np.min(
                levels
            )
        )

        if (
            minimum_level
            < minimum_eight_segment
        ):

            return (
                baseline_digit,
                baseline_confidence,
            )

        # ----------------------------------------------------------
        # Exact strong-seven-segment evidence:
        # correct 6 -> 8.
        # ----------------------------------------------------------

        STATS[
            "corrected_6_to_8"
        ] += 1

        # Confidence should represent the weakest bar.
        #
        # At threshold:
        #       confidence ~= 0.55
        #
        # Three levels above threshold:
        #       confidence ~= 0.70
        #
        # Do not artificially make this 1.0 merely because of the
        # correction.
        correction_confidence = float(
            np.clip(
                0.55
                + 0.05
                * (
                    minimum_level
                    - minimum_eight_segment
                ),
                0.55,
                0.90,
            )
        )

        return (
            8,
            max(
                baseline_confidence,
                correction_confidence,
            ),
        )

    return (
        decode_pattern_digit_with_eight_fix
    )


# ======================================================================
# Main
# ======================================================================


def main() -> int:

    wrapper_args, remaining = (
        parse_wrapper_args(
            sys.argv[1:]
        )
    )

    if (
        wrapper_args.eight_min_segment
        <= 0.0
    ):

        print(
            "Error: --eight-min-segment must be positive.",
            file=sys.stderr,
        )

        return 1

    print(
        "RDS-200 conservative 6 -> 8 correction:"
    )

    print(
        (
            "  minimum level on every segment : "
            f"{wrapper_args.eight_min_segment:.2f}"
        )
    )

    print(
        "  correction allowed              : 6 -> 8 only"
    )

    print()

    replacement_decoder = (
        make_decoder(
            wrapper_args.eight_min_segment
        )
    )

    saved_argv = (
        sys.argv
    )

    # Remove our wrapper-only option before the normal scripts parse
    # the command line.
    sys.argv = [
        saved_argv[0],
        *remaining,
    ]

    core.decode_pattern_digit = (
        replacement_decoder
    )

    try:

        result = (
            tight.main()
        )

    finally:

        core.decode_pattern_digit = (
            ORIGINAL_DECODER
        )

        sys.argv = (
            saved_argv
        )

    print()

    print(
        "RDS-200 6 -> 8 diagnostics:"
    )

    print(
        (
            f"  digit decoder calls : "
            f"{STATS['calls']}"
        )
    )

    print(
        (
            f"  baseline digit 6    : "
            f"{STATS['baseline_6']}"
        )
    )

    print(
        (
            f"  corrected 6 -> 8    : "
            f"{STATS['corrected_6_to_8']}"
        )
    )

    print(
        (
            f"  minimum segment     : "
            f"{wrapper_args.eight_min_segment:.2f}"
        )
    )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )