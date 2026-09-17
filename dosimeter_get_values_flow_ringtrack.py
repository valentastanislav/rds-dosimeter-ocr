#!/usr/bin/env python3
"""Experimental ring-only optical-flow diagnostic for RDS-200.

This wrapper keeps the normal RDS-200 geometry/decoder pipeline but restricts
optical-flow feature detection to a ring OUTSIDE the manually selected
reference display box.  The entire interior of the reference box is excluded,
so changing LCD graphics cannot contribute tracking features.

For selected sampled times it can compare the production cumulative transform
against a fresh DIRECT registration from the reference frame to that target
frame.  It also writes visual diagnostics showing reference features, tracked
points, transformed reference-box outlines, and fixed-coordinate crops after
cumulative versus direct stabilization.

It delegates all other experimental geometry/preview options to
``dosimeter_get_values_flow_segmentshape.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
            "compare production cumulative flow with direct reference-to-"
            "target registration at this sampled time; may be repeated"
        ),
    )
    parser.add_argument(
        "--flow-diagnostic-dir",
        type=Path,
        default=Path("rds200_flow_diagnostics"),
        help=(
            "directory for direct-vs-cumulative tracking diagnostics "
            "(default: rds200_flow_diagnostics)"
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
    core = flow.core

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

    def ring_bounds(frame_shape, box):
        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        x, y, width, height = box

        pad_x = int(round(ring_args.flow_ring_padding * width))
        pad_y = int(round(ring_args.flow_ring_padding * height))

        return (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_width, x + width + pad_x),
            min(frame_height, y + height + pad_y),
        )

    def make_ring_feature_mask(
        frame_shape,
        box,
        profile,
    ):
        del profile

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        x, y, width, height = box
        outer_x1, outer_y1, outer_x2, outer_y2 = ring_bounds(
            frame_shape,
            box,
        )

        mask = np.zeros(
            (frame_height, frame_width),
            dtype=np.uint8,
        )

        cv2.rectangle(
            mask,
            (outer_x1, outer_y1),
            (outer_x2 - 1, outer_y2 - 1),
            255,
            -1,
        )

        # Exclude the ENTIRE selected display box.  Only the outside ring may
        # supply optical-flow features.
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

    def direct_registration(
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
        reference_points: np.ndarray,
    ):
        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            reference_gray,
            target_gray,
            reference_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )

        if current_points is None or status is None:
            return None

        valid = status.reshape(-1) == 1
        previous = reference_points.reshape(-1, 2)[valid]
        current = current_points.reshape(-1, 2)[valid]

        if error is not None:
            good_error = error.reshape(-1)[valid]
            keep = good_error < 35.0
            previous = previous[keep]
            current = current[keep]

        if len(previous) < 3:
            return None

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            previous,
            current,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )

        if affine is None or inlier_mask is None:
            return None

        inliers = inlier_mask.reshape(-1) == 1
        return (
            flow.affine_to_homogeneous(affine),
            previous,
            current,
            inliers,
        )

    def transform_box(box, matrix):
        x, y, width, height = box
        corners = np.asarray(
            [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
            ],
            dtype=np.float64,
        )
        homogeneous = np.column_stack(
            (corners, np.ones(len(corners), dtype=np.float64))
        )
        transformed = (matrix @ homogeneous.T).T[:, :2]
        return np.rint(transformed).astype(np.int32)

    def stabilized_frame(frame, matrix):
        height, width = frame.shape[:2]
        return cv2.warpAffine(
            frame,
            matrix[0:2].astype(np.float32),
            (width, height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def crop_box(frame, box):
        x, y, width, height = box
        return frame[y:y + height, x:x + width].copy()

    def labeled(image, text):
        header = 28
        canvas = np.full(
            (image.shape[0] + header, image.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        canvas[header:] = image
        cv2.putText(
            canvas,
            text,
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def write_visual_diagnostics(
        frames,
        reference_index,
        reference_box,
        reference_points,
        transforms,
        sample_fps,
        target_indices,
    ):
        output_dir = ring_args.flow_diagnostic_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        reference_frame = frames[reference_index]
        reference_visual = reference_frame.copy()
        x, y, width, height = reference_box
        outer_x1, outer_y1, outer_x2, outer_y2 = ring_bounds(
            reference_frame.shape,
            reference_box,
        )
        cv2.rectangle(
            reference_visual,
            (outer_x1, outer_y1),
            (outer_x2 - 1, outer_y2 - 1),
            (0, 180, 180),
            2,
        )
        cv2.rectangle(
            reference_visual,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 0, 220),
            2,
        )
        for point in reference_points.reshape(-1, 2):
            cv2.circle(
                reference_visual,
                tuple(np.rint(point).astype(int)),
                4,
                (0, 220, 0),
                -1,
            )
        cv2.imwrite(
            str(output_dir / "reference_features.jpg"),
            reference_visual,
        )

        reference_gray = cv2.cvtColor(
            reference_frame,
            cv2.COLOR_BGR2GRAY,
        )

        direct_results = {}

        for frame_index in sorted(target_indices):
            if frame_index < 0 or frame_index >= len(frames):
                continue

            target_frame = frames[frame_index]
            target_gray = cv2.cvtColor(
                target_frame,
                cv2.COLOR_BGR2GRAY,
            )
            direct = direct_registration(
                reference_gray,
                target_gray,
                reference_points,
            )
            if direct is None:
                direct_results[frame_index] = None
                continue

            direct_matrix, previous, current, inliers = direct
            cumulative = transforms[frame_index]
            residual = np.linalg.inv(direct_matrix) @ cumulative
            direct_results[frame_index] = {
                "matrix": direct_matrix,
                "tracked": len(previous),
                "inliers": int(np.sum(inliers)),
                "residual": residual,
            }

            tracks = target_frame.copy()
            direct_polygon = transform_box(reference_box, direct_matrix)
            cumulative_polygon = transform_box(reference_box, cumulative)

            cv2.polylines(
                tracks,
                [direct_polygon],
                True,
                (0, 220, 0),
                3,
            )
            cv2.polylines(
                tracks,
                [cumulative_polygon],
                True,
                (0, 0, 220),
                3,
            )

            for previous_point, current_point, is_inlier in zip(
                previous,
                current,
                inliers,
            ):
                p0 = tuple(np.rint(previous_point).astype(int))
                p1 = tuple(np.rint(current_point).astype(int))
                color = (0, 200, 0) if is_inlier else (0, 165, 255)
                cv2.line(tracks, p0, p1, color, 1, cv2.LINE_AA)
                cv2.circle(tracks, p1, 3, color, -1)

            cv2.putText(
                tracks,
                "GREEN=direct box  RED=cumulative box",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                tracks,
                "GREEN=direct box  RED=cumulative box",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(
                    output_dir
                    / f"frame_{frame_index:04d}_tracks.jpg"
                ),
                tracks,
            )

            raw_crop = crop_box(target_frame, reference_box)
            cumulative_crop = crop_box(
                stabilized_frame(target_frame, cumulative),
                reference_box,
            )
            direct_crop = crop_box(
                stabilized_frame(target_frame, direct_matrix),
                reference_box,
            )

            comparison = np.hstack(
                (
                    labeled(raw_crop, "raw fixed crop"),
                    labeled(cumulative_crop, "cumulative stabilization"),
                    labeled(direct_crop, "direct reference stabilization"),
                )
            )
            cv2.imwrite(
                str(
                    output_dir
                    / f"frame_{frame_index:04d}_crops.jpg"
                ),
                comparison,
            )

        return direct_results

    def precompute_motion_with_diagnostics(*args, **kwargs):
        if not ring_args.motion_diagnostic_time:
            return original_precompute_motion(*args, **kwargs)

        def argument(name, position):
            return kwargs[name] if name in kwargs else args[position]

        video = argument("video", 0)
        info = argument("info", 1)
        profile = argument("profile", 2)
        processing_width = int(argument("processing_width", 4))
        sample_fps = float(argument("sample_fps", 5))
        reference_time = float(argument("reference_time", 6))

        reference_index = int(round(reference_time * sample_fps))
        target_indices = {
            int(round(float(value) * sample_fps))
            for value in ring_args.motion_diagnostic_time
            if float(value) >= reference_time
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
        reference_box = result[1]
        steps = diagnostic_context["steps"]

        # Re-read the same sampled BGR frames only for diagnostics.  This does
        # not alter the production transforms or OCR path.
        frames = list(
            core.iter_ffmpeg_frames(
                video,
                info,
                sample_fps,
                processing_width,
            )
        )

        if reference_index >= len(frames):
            print(
                "Direct flow diagnostics skipped: reference frame missing.",
                file=sys.stderr,
            )
            return result

        reference_gray = cv2.cvtColor(
            frames[reference_index],
            cv2.COLOR_BGR2GRAY,
        )
        reference_mask = make_ring_feature_mask(
            reference_gray.shape,
            reference_box,
            profile,
        )
        reference_points = flow.detect_features(
            reference_gray,
            reference_mask,
        )

        direct_results = {}
        if reference_points is not None:
            direct_results = write_visual_diagnostics(
                frames,
                reference_index,
                reference_box,
                reference_points,
                transforms,
                sample_fps,
                target_indices,
            )

        print()
        print("Target-time optical-flow diagnostics:")
        print(
            f"  reference index/time : {reference_index} / "
            f"{reference_index / sample_fps:.3f} s"
        )
        print(
            f"  diagnostic images    : {ring_args.flow_diagnostic_dir}"
        )

        for requested_time in ring_args.motion_diagnostic_time:
            frame_index = int(round(float(requested_time) * sample_fps))
            if frame_index < 0 or frame_index >= len(transforms):
                print(
                    f"  requested {requested_time:.3f}s -> frame "
                    f"{frame_index}: outside sampled video"
                )
                continue

            cumulative = transforms[frame_index]
            cdx, cdy, crot, cscale = flow.decompose_similarity(cumulative)
            step = steps.get(frame_index)
            direct = direct_results.get(frame_index)

            print(
                f"  frame {frame_index:4d} "
                f"t={frame_index / sample_fps:7.3f}s"
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

            if direct is None:
                print("    direct     : unavailable")
            else:
                direct_matrix = direct["matrix"]
                ddx, ddy, drot, dscale = flow.decompose_similarity(
                    direct_matrix
                )
                rdx, rdy, rrot, rscale = flow.decompose_similarity(
                    direct["residual"]
                )
                print(
                    "    direct     : "
                    f"tracked={direct['tracked']} "
                    f"inliers={direct['inliers']} "
                    f"dx={ddx:+.3f}px dy={ddy:+.3f}px "
                    f"rot={drot:+.4f}deg scale={dscale:.6f}"
                )
                print(
                    "    cum/direct : "
                    f"dx={rdx:+.3f}px dy={rdy:+.3f}px "
                    f"rot={rrot:+.4f}deg scale={rscale:.6f}"
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
        print(f"  diagnostic dir: {ring_args.flow_diagnostic_dir}")
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
