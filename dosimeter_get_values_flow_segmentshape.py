#!/usr/bin/env python3
"""Experimental RDS-200 segment-shape refinement wrapper.

This wrapper sits on top of ``dosimeter_get_values_flow_tightsegments.py``
and changes only the shape of the seven-segment sampling polygons after the
existing tight-grid stretch/shift transform has been applied.

It is intended for visual geometry diagnostics, not production use.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import dosimeter_get_values_flow_tightsegments as tight


def parse_shape_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--segment-x-thickness",
        type=float,
        default=1.0,
        help=(
            "scale each segment polygon horizontally about its own centroid "
            "after the tight-grid transform (default: 1.0)"
        ),
    )

    parser.add_argument(
        "--segment-y-thickness",
        type=float,
        default=1.0,
        help=(
            "scale each segment polygon vertically about its own centroid "
            "after the tight-grid transform (default: 1.0)"
        ),
    )

    parser.add_argument(
        "--segment-x-shear",
        type=float,
        default=0.0,
        help=(
            "additional x shear versus normalized digit y; negative values "
            "move the lower part left (default: 0.0)"
        ),
    )

    parser.add_argument(
        "--segment-shear-center",
        type=float,
        default=67.0,
        help=(
            "normalized digit y coordinate about which x shear is applied "
            "(default: 67)"
        ),
    )

    return parser.parse_known_args(argv)


def main() -> int:
    shape_args, remaining = parse_shape_args(sys.argv[1:])

    if shape_args.segment_x_thickness <= 0.0:
        print(
            "Error: --segment-x-thickness must be positive.",
            file=sys.stderr,
        )
        return 1

    if shape_args.segment_y_thickness <= 0.0:
        print(
            "Error: --segment-y-thickness must be positive.",
            file=sys.stderr,
        )
        return 1

    original_transform = tight.transform_polygon

    def transform_polygon_with_shape(
        polygon: np.ndarray,
        x_stretch: float,
        x_center: float,
        x_shift: float,
        y_stretch: float,
        y_center: float,
        y_shift: float,
    ) -> np.ndarray:
        points = original_transform(
            polygon,
            x_stretch=x_stretch,
            x_center=x_center,
            x_shift=x_shift,
            y_stretch=y_stretch,
            y_center=y_center,
            y_shift=y_shift,
        ).astype(float)

        centroid = points.mean(axis=0)

        points[:, 0] = (
            centroid[0]
            + shape_args.segment_x_thickness
            * (points[:, 0] - centroid[0])
        )

        points[:, 1] = (
            centroid[1]
            + shape_args.segment_y_thickness
            * (points[:, 1] - centroid[1])
        )

        points[:, 0] += (
            shape_args.segment_x_shear
            * (points[:, 1] - shape_args.segment_shear_center)
        )

        points[:, 0] = np.clip(points[:, 0], 0.0, 64.0)
        points[:, 1] = np.clip(points[:, 1], 0.0, 129.0)

        return np.rint(points).astype(np.int32)

    print("Experimental RDS-200 segment shape refinement:")
    print(f"  x thickness : {shape_args.segment_x_thickness:.3f}")
    print(f"  y thickness : {shape_args.segment_y_thickness:.3f}")
    print(f"  x shear     : {shape_args.segment_x_shear:+.3f}")
    print(f"  shear center: {shape_args.segment_shear_center:.2f}")
    print()

    saved_argv = sys.argv
    tight.transform_polygon = transform_polygon_with_shape
    sys.argv = [saved_argv[0], *remaining]

    try:
        return tight.main()
    finally:
        tight.transform_polygon = original_transform
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
