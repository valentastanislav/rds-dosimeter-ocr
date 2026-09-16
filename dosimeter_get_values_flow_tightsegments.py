#!/usr/bin/env python3
"""
Test RDS-200 segment geometry for a tightly selected digit grid.

This wrapper changes ONLY the geometry of the seven-segment masks.

It keeps unchanged:

    - optical-flow tracking
    - ROI
    - perspective quad
    - fixed digit boxes
    - decimal position
    - temporal filtering
    - standard RDS-200 decoder

The original RDS200 segment geometry was designed for digit boxes with
considerable vertical margins.  With a newly selected TIGHT digit grid,
the top and bottom segment masks therefore fall too far toward the
middle of each digit.

We correct this by stretching all segment polygons vertically around
the center of the middle segment.

The wrapper calls dosimeter_get_values_flow_diag.py so the same detailed
RAW/FILTERED segment diagnostics are produced.

Required files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py
    dosimeter_get_values_flow_diag.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_flow_diag as diag


# ======================================================================
# Wrapper arguments
# ======================================================================


def parse_digit_x_offsets(
    text: str,
) -> tuple[int, int, int]:
    try:
        values = tuple(
            int(value.strip())
            for value in text.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--digit-x-offsets must contain three integers: DX1,DX2,DX3"
        ) from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--digit-x-offsets must contain three integers: DX1,DX2,DX3"
        )
    return values


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--segment-y-stretch",
        type=float,
        default=1.45,
        help=(
            "vertical stretch of the seven-segment mask around "
            "the middle bar (default: 1.45)"
        ),
    )

    parser.add_argument(
        "--segment-y-center",
        type=float,
        default=67.0,
        help=(
            "vertical center in the normalized 65x130 digit patch "
            "(default: 67)"
        ),
    )

    parser.add_argument(
        "--segment-y-shift",
        type=float,
        default=0.0,
        help=(
            "additional vertical shift after stretching "
            "(default: 0)"
        ),
    )

    parser.add_argument(
        "--segment-x-stretch",
        type=float,
        default=1.0,
        help=(
            "horizontal stretch; keep 1.0 for the first test "
            "(default: 1.0)"
        ),
    )

    parser.add_argument(
        "--segment-x-center",
        type=float,
        default=32.0,
        help=(
            "horizontal stretch center in the normalized 65x130 "
            "digit patch (default: 32)"
        ),
    )

    parser.add_argument(
        "--segment-x-shift",
        type=float,
        default=0.0,
        help=(
            "additional horizontal shift after stretching "
            "(default: 0)"
        ),
    )

    parser.add_argument(
        "--digit-x-offsets",
        type=parse_digit_x_offsets,
        default=(0, 0, 0),
        help=(
            "per-digit integer horizontal segment-mask offsets "
            "DX1,DX2,DX3 (default: 0,0,0)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Geometry transformation
# ======================================================================


def transform_polygon(
    polygon: np.ndarray,
    x_stretch: float,
    x_center: float,
    x_shift: float,
    y_stretch: float,
    y_center: float,
    y_shift: float,
) -> np.ndarray:

    points = np.asarray(
        polygon,
        dtype=float,
    ).copy()

    points[:, 0] = (
        x_center
        + x_stretch
        * (
            points[:, 0]
            - x_center
        )
        + x_shift
    )

    points[:, 1] = (
        y_center
        + y_stretch
        * (
            points[:, 1]
            - y_center
        )
        + y_shift
    )

    # The normalized digit patch is exactly 65 x 130.
    points[:, 0] = np.clip(
        points[:, 0],
        0.0,
        64.0,
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


def make_digit_segment_polygons(
    polygons: dict[str, np.ndarray],
    x_offsets: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], ...]:
    return tuple(
        {
            name: polygon
            + np.asarray(
                (x_offset, 0),
                dtype=polygon.dtype,
            )
            for name, polygon in polygons.items()
        }
        for x_offset in x_offsets
    )


def make_digit_offset_profile(
    profile: core.Profile,
    polygons: dict[str, np.ndarray],
    x_offsets: tuple[int, int, int],
) -> core.Profile | None:
    if not any(x_offsets):
        return None
    return replace(
        profile,
        digit_segment_polygons=make_digit_segment_polygons(
            polygons,
            x_offsets,
        ),
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
        wrapper_args.segment_y_stretch
        <= 0.0
    ):

        print(
            "Error: --segment-y-stretch must be positive.",
            file=sys.stderr,
        )

        return 1

    if (
        wrapper_args.segment_x_stretch
        <= 0.0
    ):

        print(
            "Error: --segment-x-stretch must be positive.",
            file=sys.stderr,
        )

        return 1

    # ----------------------------------------------------------
    # Important:
    #
    # mutate the existing dictionary IN PLACE.
    #
    # Other imported modules may already hold a Profile object that
    # points to this same dictionary.  Keeping the dictionary identity
    # guarantees that all layers see the transformed polygons.
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
        name: transform_polygon(
            polygon,
            x_stretch=(
                wrapper_args.segment_x_stretch
            ),
            x_center=(
                wrapper_args.segment_x_center
            ),
            x_shift=(
                wrapper_args.segment_x_shift
            ),
            y_stretch=(
                wrapper_args.segment_y_stretch
            ),
            y_center=(
                wrapper_args.segment_y_center
            ),
            y_shift=(
                wrapper_args.segment_y_shift
            ),
        )
        for name, polygon
        in original_polygons.items()
    }

    profile_override = make_digit_offset_profile(
        profile,
        transformed_polygons,
        wrapper_args.digit_x_offsets,
    )

    print(
        "Temporary RDS-200 tight-grid segment geometry:"
    )

    print(
        (
            f"  y stretch : "
            f"{wrapper_args.segment_y_stretch:.3f}"
        )
    )

    print(
        (
            f"  y center  : "
            f"{wrapper_args.segment_y_center:.2f}"
        )
    )

    print(
        (
            f"  y shift   : "
            f"{wrapper_args.segment_y_shift:+.2f}"
        )
    )

    print(
        (
            f"  x stretch : "
            f"{wrapper_args.segment_x_stretch:.3f}"
        )
    )

    print(
        (
            f"  x center  : "
            f"{wrapper_args.segment_x_center:.2f}"
        )
    )

    print(
        (
            f"  x shift   : "
            f"{wrapper_args.segment_x_shift:+.2f}"
        )
    )

    if profile_override is not None:
        print(
            (
                "  digit x offsets: "
                + ",".join(
                    f"{value:+d}"
                    for value in wrapper_args.digit_x_offsets
                )
            )
        )

    print()

    print(
        "Segment polygons, old -> new:"
    )

    for name in (
        core.SEGMENT_ORDER
    ):

        old = (
            original_polygons[
                name
            ]
        )

        new = (
            transformed_polygons[
                name
            ]
        )

        print(
            (
                f"  {name}: "
                f"{old.tolist()} "
                f"-> "
                f"{new.tolist()}"
            )
        )

    # Install transformed polygons.
    polygons.clear()

    polygons.update(
        transformed_polygons
    )

    saved_argv = (
        sys.argv
    )

    # Remove our wrapper-only arguments before the ordinary parser
    # sees the command line.
    sys.argv = [
        saved_argv[0],
        *remaining,
    ]

    try:

        result = (
            diag.main(
                profile_override=profile_override
            )
        )

    finally:

        sys.argv = (
            saved_argv
        )

        # Restore the original profile, so importing/running things
        # later in the same Python process cannot inherit this test.
        polygons.clear()

        polygons.update(
            original_polygons
        )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
