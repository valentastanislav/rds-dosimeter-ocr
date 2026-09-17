#!/usr/bin/env python3
"""Experimental ring-only optical-flow diagnostic for RDS-200.

This wrapper keeps the normal RDS-200 geometry/decoder pipeline but restricts
optical-flow feature detection to a ring OUTSIDE the manually selected
reference display box.  The entire interior of the reference box is excluded,
so changing LCD graphics cannot contribute tracking features.

It delegates all other experimental geometry/preview options to
``dosimeter_get_values_flow_segmentshape.py``.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import dosimeter_get_values_flow_segmentshape as segmentshape


def parse_ring_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--flow-ring-padding",
        type=float,
        default=0.25,
        help=(
            "outer optical-flow tracking-ring width as a fraction of the "
            "reference-box size (default: 0.25)"
        ),
    )
    return parser.parse_known_args(argv)


def main() -> int:
    ring_args, remaining = parse_ring_args(sys.argv[1:])

    if ring_args.flow_ring_padding <= 0.0:
        print(
            "Error: --flow-ring-padding must be positive.",
            file=sys.stderr,
        )
        return 1

    flow = segmentshape.tight.diag.flow
    original_make_feature_mask = flow.make_feature_mask

    def make_ring_feature_mask(
        frame_shape,
        box,
        profile,
    ):
        del profile

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        x, y, width, height = box

        pad_x = int(round(ring_args.flow_ring_padding * width))
        pad_y = int(round(ring_args.flow_ring_padding * height))

        outer_x1 = max(0, x - pad_x)
        outer_y1 = max(0, y - pad_y)
        outer_x2 = min(frame_width, x + width + pad_x)
        outer_y2 = min(frame_height, y + height + pad_y)

        mask = np.zeros(
            (frame_height, frame_width),
            dtype=np.uint8,
        )

        # Allow features in the larger region around the display.
        cv2.rectangle(
            mask,
            (outer_x1, outer_y1),
            (outer_x2 - 1, outer_y2 - 1),
            255,
            -1,
        )

        # Critical diagnostic change: exclude the ENTIRE reference-box
        # interior.  Only the outside ring may supply optical-flow features.
        cv2.rectangle(
            mask,
            (x, y),
            (x + width - 1, y + height - 1),
            0,
            -1,
        )

        return mask

    print("Experimental ring-only optical-flow tracking:")
    print(f"  ring padding : {ring_args.flow_ring_padding:.3f}")
    print("  box interior : fully excluded")
    print()

    saved_argv = sys.argv
    flow.make_feature_mask = make_ring_feature_mask
    sys.argv = [saved_argv[0], *remaining]

    try:
        return segmentshape.main()
    finally:
        flow.make_feature_mask = original_make_feature_mask
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
