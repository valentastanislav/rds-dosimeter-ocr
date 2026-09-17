#!/usr/bin/env python3
"""Experimental per-segment offsets for the RDS-200 flow pipeline.

This wrapper sits above ``dosimeter_get_values_flow_ringtrack.py`` and adds
small independent x/y offsets to the seven sampling polygons in normalized
65x130 digit coordinates.  It is intended for targeted geometry diagnostics,
for example moving only segment e without disturbing the other six segments.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import dosimeter_get_values_flow_ringtrack as ringtrack


SEGMENT_ORDER = tuple(ringtrack.segmentshape.tight.core.SEGMENT_ORDER)


def parse_offsets(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "offset list must contain seven numbers for a,b,c,d,e,f,g"
        ) from exc
    if len(values) != 7:
        raise argparse.ArgumentTypeError(
            "offset list must contain seven numbers for a,b,c,d,e,f,g"
        )
    return values


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--segment-x-offsets",
        type=parse_offsets,
        default=(0.0,) * 7,
        metavar="A,B,C,D,E,F,G",
        help=(
            "independent x offsets of segment polygons in normalized digit "
            "coordinates (default: all zero)"
        ),
    )
    parser.add_argument(
        "--segment-y-offsets",
        type=parse_offsets,
        default=(0.0,) * 7,
        metavar="A,B,C,D,E,F,G",
        help=(
            "independent y offsets of segment polygons in normalized digit "
            "coordinates (default: all zero)"
        ),
    )
    return parser.parse_known_args(argv)


def main() -> int:
    args, remaining = parse_args(sys.argv[1:])

    tight = ringtrack.segmentshape.tight
    original_transform = tight.transform_polygon

    base_polygons = {
        name: np.asarray(polygon).copy()
        for name, polygon in tight.core.PROFILES["rds200"].segment_polygons.items()
    }
    x_offsets = dict(zip(SEGMENT_ORDER, args.segment_x_offsets))
    y_offsets = dict(zip(SEGMENT_ORDER, args.segment_y_offsets))

    def transform_polygon_with_offsets(
        polygon,
        x_stretch,
        x_center,
        x_shift,
        y_stretch,
        y_center,
        y_shift,
    ):
        transformed = original_transform(
            polygon,
            x_stretch=x_stretch,
            x_center=x_center,
            x_shift=x_shift,
            y_stretch=y_stretch,
            y_center=y_center,
            y_shift=y_shift,
        ).astype(float)

        segment_name = None
        source = np.asarray(polygon)
        for name, reference in base_polygons.items():
            if np.array_equal(source, reference):
                segment_name = name
                break

        if segment_name is not None:
            transformed[:, 0] += x_offsets[segment_name]
            transformed[:, 1] += y_offsets[segment_name]

        transformed[:, 0] = np.clip(transformed[:, 0], 0.0, 64.0)
        transformed[:, 1] = np.clip(transformed[:, 1], 0.0, 129.0)
        return np.rint(transformed).astype(np.int32)

    print("Experimental per-segment offsets (a,b,c,d,e,f,g):")
    print("  x: " + ",".join(f"{value:+.2f}" for value in args.segment_x_offsets))
    print("  y: " + ",".join(f"{value:+.2f}" for value in args.segment_y_offsets))
    print()

    saved_argv = sys.argv
    tight.transform_polygon = transform_polygon_with_offsets
    sys.argv = [saved_argv[0], *remaining]
    try:
        return ringtrack.main()
    finally:
        tight.transform_polygon = original_transform
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
