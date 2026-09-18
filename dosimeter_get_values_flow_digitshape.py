#!/usr/bin/env python3
"""Experimental anchored per-digit mask scaling for RDS-200.

This wrapper sits above ``dosimeter_get_values_flow_ringtrack.py`` and applies
one coherent affine-like expansion to all seven sampling polygons of each
digit after the existing tight-grid geometry has been constructed.

The intended RDS-200 geometry is:

- left digit: anchor at bottom-right, expand left/up
- middle digit: anchor at bottom-centre, expand left/right/up
- right digit: anchor at bottom-left, expand right/up

This preserves the internal seven-segment geometry of each digit and avoids
per-segment tuning against one video's error pattern.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

import dosimeter_get_values_flow_ringtrack as ringtrack


def parse_three_floats(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected three comma-separated numbers"
        ) from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "expected three comma-separated numbers"
        )
    return values


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--digit-mask-x-stretch",
        type=float,
        default=1.15,
        help=(
            "shared horizontal expansion of each digit mask around its own "
            "x anchor (default: 1.15)"
        ),
    )
    parser.add_argument(
        "--digit-mask-y-stretch",
        type=float,
        default=1.10,
        help=(
            "shared vertical expansion of each digit mask around the bottom "
            "anchor (default: 1.10)"
        ),
    )
    parser.add_argument(
        "--digit-mask-anchor-xs",
        type=parse_three_floats,
        default=(64.0, 32.0, 0.0),
        metavar="LEFT,MIDDLE,RIGHT",
        help=(
            "normalized x anchors for the three digit masks; defaults to "
            "right edge, centre, left edge: 64,32,0"
        ),
    )
    parser.add_argument(
        "--digit-mask-anchor-y",
        type=float,
        default=129.0,
        help=(
            "normalized y anchor shared by all digit masks "
            "(default: bottom edge 129)"
        ),
    )
    return parser.parse_known_args(argv)


def main() -> int:
    args, remaining = parse_args(sys.argv[1:])

    if args.digit_mask_x_stretch <= 0.0:
        print(
            "Error: --digit-mask-x-stretch must be positive.",
            file=sys.stderr,
        )
        return 1
    if args.digit_mask_y_stretch <= 0.0:
        print("Error: --digit-mask-y-stretch must be positive.", file=sys.stderr)
        return 1

    tight = ringtrack.segmentshape.tight
    original_make_digit_offset_profile = tight.make_digit_offset_profile

    def make_anchored_digit_profile(
        profile,
        polygons,
        x_offsets,
    ):
        per_digit = []

        for digit_index, (anchor_x, x_offset) in enumerate(
            zip(args.digit_mask_anchor_xs, x_offsets)
        ):
            digit_polygons = {}

            for name, polygon in polygons.items():
                points = np.asarray(polygon, dtype=float).copy()

                # Coherent per-digit expansion.  Bottom geometry remains
                # anchored while the top expands upward.  The x anchors fan
                # the three masks outward from the centre of the display.
                points[:, 0] = (
                    anchor_x
                    + args.digit_mask_x_stretch
                    * (points[:, 0] - anchor_x)
                    + x_offset
                )
                points[:, 1] = (
                    args.digit_mask_anchor_y
                    + args.digit_mask_y_stretch
                    * (points[:, 1] - args.digit_mask_anchor_y)
                )

                points[:, 0] = np.clip(points[:, 0], 0.0, 64.0)
                points[:, 1] = np.clip(points[:, 1], 0.0, 129.0)
                digit_polygons[name] = np.rint(points).astype(np.int32)

            per_digit.append(digit_polygons)

        return replace(
            profile,
            digit_segment_polygons=tuple(per_digit),
        )

    print("Experimental anchored per-digit mask scaling:")
    print(f"  x stretch : {args.digit_mask_x_stretch:.3f}")
    print(f"  y stretch : {args.digit_mask_y_stretch:.3f}")
    print(
        "  x anchors : "
        + ",".join(f"{value:.2f}" for value in args.digit_mask_anchor_xs)
    )
    print(f"  y anchor  : {args.digit_mask_anchor_y:.2f}")
    print()

    saved_argv = sys.argv
    tight.make_digit_offset_profile = make_anchored_digit_profile
    sys.argv = [saved_argv[0], *remaining]

    try:
        return ringtrack.main()
    finally:
        tight.make_digit_offset_profile = original_make_digit_offset_profile
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
