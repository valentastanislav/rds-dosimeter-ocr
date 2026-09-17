#!/usr/bin/env python3
"""Experimental ring-only optical-flow diagnostic for RDS-200.

This wrapper keeps the normal RDS-200 geometry/decoder pipeline but restricts
optical-flow feature detection to a ring OUTSIDE the manually selected
reference display box.  The entire interior of the reference box is excluded,
so changing LCD graphics cannot contribute tracking features.

It can also print the production optical-flow step and cumulative similarity
transform at selected sampled times.  This is diagnostic only; it does not
change the transform used by the OCR pipeline.

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
    parser.add_argument(
        "--motion-diagnostic-time",
        type=float,
        action="append",
        default=None,
        metavar="SECONDS",
        help=(
            "print the accepted frame-to-frame step and cumulative optical-"
            "flow transform at this sampled time; may be repeated"
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
    original_estimate_step = flow.estimate_step
    original_precompute_motion = flow.precompute_motion

    diagnostic_context: dict[str, object] = {
        "active": False,
        "forward_call": 0,
        "reference_index": 0,
        "target_indices": set(),
        "maximum_target": -1,
        "steps": {},
    }

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

    def estimate_step_with_diagnostics(*args, **kwargs):
        result = original_estimate_step(*args, **kwargs)

        if not diagnostic_context["active"]:
            return result

        diagnostic_context["forward_call"] = int(
            diagnostic_context["forward_call"]
        ) + 1

        current_index = (
            int(diagnostic_context["reference_index"])
            + int(diagnostic_context["forward_call"])
        )

        # Target times used for this experiment are after the reference time.
        # Once the forward call index has passed the largest requested target,
        # later calls (including the backward pass) are irrelevant here.
        if current_index > int(diagnostic_context["maximum_target"]):
            return result

        target_indices = diagnostic_context["target_indices"]
        if current_index not in target_indices:
            return result

        (
            accepted,
            delta,
            _tracked_points,
            number_tracked,
            number_inliers,
        ) = result

        dx, dy, rotation, scale = flow.decompose_similarity(delta)
        diagnostic_context["steps"][current_index] = {
            "accepted": bool(accepted),
            "tracked": int(number_tracked),
            "inliers": int(number_inliers),
            "dx": float(dx),
            "dy": float(dy),
            "rotation": float(rotation),
            "scale": float(scale),
        }

        return result

    def precompute_motion_with_diagnostics(*args, **kwargs):
        if not ring_args.motion_diagnostic_time:
            return original_precompute_motion(*args, **kwargs)

        sample_fps = (
            kwargs.get("sample_fps")
            if "sample_fps" in kwargs
            else args[5]
        )
        reference_time = (
            kwargs.get("reference_time")
            if "reference_time" in kwargs
            else args[6]
        )

        reference_index = int(round(float(reference_time) * float(sample_fps)))
        target_indices = {
            int(round(float(value) * float(sample_fps)))
            for value in ring_args.motion_diagnostic_time
            if float(value) >= float(reference_time)
        }

        diagnostic_context["active"] = True
        diagnostic_context["forward_call"] = 0
        diagnostic_context["reference_index"] = reference_index
        diagnostic_context["target_indices"] = target_indices
        diagnostic_context["maximum_target"] = max(
            target_indices,
            default=-1,
        )
        diagnostic_context["steps"] = {}

        try:
            result = original_precompute_motion(*args, **kwargs)
        finally:
            diagnostic_context["active"] = False

        transforms = result[0]
        steps = diagnostic_context["steps"]

        print()
        print("Target-time optical-flow diagnostics:")
        print(
            f"  reference index/time : {reference_index} / "
            f"{reference_index / float(sample_fps):.3f} s"
        )

        for requested_time in ring_args.motion_diagnostic_time:
            frame_index = int(round(float(requested_time) * float(sample_fps)))
            if frame_index < 0 or frame_index >= len(transforms):
                print(
                    f"  requested {requested_time:.3f}s -> frame "
                    f"{frame_index}: outside sampled video"
                )
                continue

            cumulative = transforms[frame_index]
            cdx, cdy, crot, cscale = flow.decompose_similarity(cumulative)
            step = steps.get(frame_index)

            print(
                f"  frame {frame_index:4d} t={frame_index / float(sample_fps):7.3f}s"
            )
            if step is None:
                print("    step       : not captured")
            else:
                print(
                    "    step       : "
                    f"accepted={int(step['accepted'])} "
                    f"tracked={step['tracked']} inliers={step['inliers']} "
                    f"dx={step['dx']:+.3f}px dy={step['dy']:+.3f}px "
                    f"rot={step['rotation']:+.4f}deg "
                    f"scale={step['scale']:.6f}"
                )
            print(
                "    cumulative : "
                f"dx={cdx:+.3f}px dy={cdy:+.3f}px "
                f"rot={crot:+.4f}deg scale={cscale:.6f}"
            )
            print(
                "    matrix     : "
                + np.array2string(
                    cumulative,
                    precision=6,
                    suppress_small=False,
                ).replace("\n", " ")
            )

        return result

    print("Experimental ring-only optical-flow tracking:")
    print(f"  ring padding : {ring_args.flow_ring_padding:.3f}")
    print("  box interior : fully excluded")
    if ring_args.motion_diagnostic_time:
        print(
            "  motion diagnostics: "
            + ", ".join(
                f"{value:.3f}s"
                for value in ring_args.motion_diagnostic_time
            )
        )
    print()

    saved_argv = sys.argv
    flow.make_feature_mask = make_ring_feature_mask
    flow.estimate_step = estimate_step_with_diagnostics
    flow.precompute_motion = precompute_motion_with_diagnostics
    sys.argv = [saved_argv[0], *remaining]

    try:
        return segmentshape.main()
    finally:
        flow.make_feature_mask = original_make_feature_mask
        flow.estimate_step = original_estimate_step
        flow.precompute_motion = original_precompute_motion
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
