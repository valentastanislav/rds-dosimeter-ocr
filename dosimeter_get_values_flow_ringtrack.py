#!/usr/bin/env python3
"""Experimental ring-only optical-flow diagnostic for RDS-200.

This wrapper keeps the normal RDS-200 geometry/decoder pipeline but restricts
optical-flow feature detection to a ring OUTSIDE the manually selected
reference display box.  The entire interior of the reference box is excluded,
so changing LCD graphics cannot contribute tracking features.

It also exposes an experimental registration mode:

    --flow-registration cumulative   # current shared flow behaviour
    --flow-registration direct       # reference -> every sampled frame

The direct mode deliberately reuses the same LK + RANSAC step estimator as the
normal flow code, but always compares the fixed reference frame directly with
the target frame instead of composing frame-to-frame transforms.  This keeps
the experiment focused on accumulation drift rather than changing the motion
estimator itself.

For selected sampled times it can compare the transform used by the pipeline
against a fresh DIRECT registration from the reference frame to that target
frame.  It also writes visual diagnostics showing reference features, tracked
points, transformed reference-box outlines, and fixed-coordinate crops.

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
        "--flow-registration",
        choices=("cumulative", "direct"),
        default="cumulative",
        help=(
            "optical-flow registration strategy: cumulative keeps the current "
            "frame-to-frame composition; direct estimates reference-to-frame "
            "motion independently for every sampled frame"
        ),
    )
    parser.add_argument(
        "--motion-diagnostic-time",
        type=float,
        action="append",
        default=None,
        metavar="SECONDS",
        help=(
            "print flow diagnostics at this sampled time; may be repeated"
        ),
    )
    parser.add_argument(
        "--flow-diagnostic-dir",
        type=Path,
        default=Path("rds200_flow_diagnostics"),
        help=(
            "directory for tracking diagnostics "
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
    original_precompute_motion = flow.precompute_motion

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

        # Only the outside ring may supply optical-flow features.
        cv2.rectangle(
            mask,
            (x, y),
            (x + width - 1, y + height - 1),
            0,
            -1,
        )

        return mask

    def direct_registration(
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
        reference_points: np.ndarray,
        maximum_translation: float,
        maximum_rotation: float,
        minimum_scale: float,
        maximum_scale: float,
        minimum_inliers: int,
    ):
        return flow.estimate_step(
            reference_gray,
            target_gray,
            reference_points,
            maximum_translation,
            maximum_rotation,
            minimum_scale,
            maximum_scale,
            minimum_inliers,
        )

    def replace_with_direct_transforms(
        original_result,
        video,
        info,
        profile,
        processing_width,
        sample_fps,
        reference_time,
        maximum_translation,
        maximum_rotation,
        minimum_scale,
        maximum_scale,
        minimum_inliers,
    ):
        cumulative_transforms, reference_box, processing_size, _stats = (
            original_result
        )

        frames = list(
            core.iter_ffmpeg_frames(
                video,
                info,
                sample_fps,
                processing_width,
            )
        )

        if not frames:
            raise RuntimeError("No sampled frames available for direct flow.")

        reference_index = int(round(reference_time * sample_fps))
        reference_index = max(0, min(reference_index, len(frames) - 1))

        gray_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for frame in frames
        ]
        reference_gray = gray_frames[reference_index]
        reference_mask = make_ring_feature_mask(
            reference_gray.shape,
            reference_box,
            profile,
        )
        reference_points = flow.detect_features(
            reference_gray,
            reference_mask,
        )

        if reference_points is None or len(reference_points) < minimum_inliers:
            raise RuntimeError(
                "Not enough reference features for direct registration."
            )

        transforms = [
            flow.identity_matrix()
            for _ in gray_frames
        ]

        stats: dict[str, float | int] = {
            "accepted": 0,
            "rejected": 0,
            "tracked_points_sum": 0,
            "inliers_sum": 0,
            "steps": 0,
            "redetections": 0,
            "direct_fallbacks": 0,
        }

        for frame_index, current_gray in enumerate(gray_frames):
            if frame_index == reference_index:
                transforms[frame_index] = flow.identity_matrix()
                continue

            (
                accepted,
                matrix,
                _tracked_points,
                number_tracked,
                number_inliers,
            ) = direct_registration(
                reference_gray,
                current_gray,
                reference_points,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            )

            stats["steps"] += 1
            stats["tracked_points_sum"] += number_tracked
            stats["inliers_sum"] += number_inliers

            if accepted:
                transforms[frame_index] = matrix
                stats["accepted"] += 1
            else:
                # Experimental safety fallback: preserve the old cumulative
                # transform only for targets where the direct fit fails.
                transforms[frame_index] = cumulative_transforms[frame_index]
                stats["rejected"] += 1
                stats["direct_fallbacks"] += 1

        print()
        print("Direct reference-registration diagnostics:")
        print(f"  reference frame      : {reference_index}")
        print(f"  reference features   : {len(reference_points)}")
        print(f"  accepted direct fits : {stats['accepted']}")
        print(f"  rejected direct fits : {stats['rejected']}")
        print(f"  cumulative fallbacks : {stats['direct_fallbacks']}")

        return transforms, reference_box, processing_size, stats

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
        maximum_translation,
        maximum_rotation,
        minimum_scale,
        maximum_scale,
        minimum_inliers,
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

        for frame_index in sorted(target_indices):
            if frame_index < 0 or frame_index >= len(frames):
                continue

            target_frame = frames[frame_index]
            target_gray = cv2.cvtColor(
                target_frame,
                cv2.COLOR_BGR2GRAY,
            )
            (
                accepted,
                direct_matrix,
                tracked_points,
                _number_tracked,
                _number_inliers,
            ) = direct_registration(
                reference_gray,
                target_gray,
                reference_points,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            )
            if not accepted:
                continue

            used_matrix = transforms[frame_index]
            tracks = target_frame.copy()
            direct_polygon = transform_box(reference_box, direct_matrix)
            used_polygon = transform_box(reference_box, used_matrix)

            cv2.polylines(
                tracks,
                [direct_polygon],
                True,
                (0, 220, 0),
                3,
            )
            cv2.polylines(
                tracks,
                [used_polygon],
                True,
                (0, 0, 220),
                3,
            )

            if tracked_points is not None:
                for point in tracked_points.reshape(-1, 2):
                    cv2.circle(
                        tracks,
                        tuple(np.rint(point).astype(int)),
                        3,
                        (0, 220, 0),
                        -1,
                    )

            cv2.putText(
                tracks,
                "GREEN=direct  RED=pipeline-used transform",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                tracks,
                "GREEN=direct  RED=pipeline-used transform",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(output_dir / f"frame_{frame_index:04d}_tracks.jpg"),
                tracks,
            )

            raw_crop = crop_box(target_frame, reference_box)
            used_crop = crop_box(
                stabilized_frame(target_frame, used_matrix),
                reference_box,
            )
            direct_crop = crop_box(
                stabilized_frame(target_frame, direct_matrix),
                reference_box,
            )

            comparison = np.hstack(
                (
                    labeled(raw_crop, "raw fixed crop"),
                    labeled(
                        used_crop,
                        f"pipeline: {ring_args.flow_registration}",
                    ),
                    labeled(direct_crop, "direct reference stabilization"),
                )
            )
            cv2.imwrite(
                str(output_dir / f"frame_{frame_index:04d}_crops.jpg"),
                comparison,
            )

    def precompute_motion_with_registration(*args, **kwargs):
        def argument(name, position):
            return kwargs[name] if name in kwargs else args[position]

        video = argument("video", 0)
        info = argument("info", 1)
        profile = argument("profile", 2)
        processing_width = int(argument("processing_width", 4))
        sample_fps = float(argument("sample_fps", 5))
        reference_time = float(argument("reference_time", 6))
        maximum_translation = float(argument("maximum_translation", 7))
        maximum_rotation = float(argument("maximum_rotation", 8))
        minimum_scale = float(argument("minimum_scale", 9))
        maximum_scale = float(argument("maximum_scale", 10))
        minimum_inliers = int(argument("minimum_inliers", 11))

        original_result = original_precompute_motion(*args, **kwargs)

        if ring_args.flow_registration == "direct":
            result = replace_with_direct_transforms(
                original_result,
                video,
                info,
                profile,
                processing_width,
                sample_fps,
                reference_time,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            )
        else:
            result = original_result

        if not ring_args.motion_diagnostic_time:
            return result

        transforms, reference_box, _processing_size, _stats = result
        frames = list(
            core.iter_ffmpeg_frames(
                video,
                info,
                sample_fps,
                processing_width,
            )
        )
        reference_index = int(round(reference_time * sample_fps))
        reference_index = max(0, min(reference_index, len(frames) - 1))
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

        target_indices = {
            int(round(float(value) * sample_fps))
            for value in ring_args.motion_diagnostic_time
        }

        if reference_points is not None:
            write_visual_diagnostics(
                frames,
                reference_index,
                reference_box,
                reference_points,
                transforms,
                sample_fps,
                target_indices,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            )

        print()
        print("Target-time optical-flow diagnostics:")
        print(f"  registration mode    : {ring_args.flow_registration}")
        print(
            f"  reference index/time : {reference_index} / "
            f"{reference_index / sample_fps:.3f} s"
        )
        print(f"  diagnostic images    : {ring_args.flow_diagnostic_dir}")

        for requested_time in ring_args.motion_diagnostic_time:
            frame_index = int(round(float(requested_time) * sample_fps))
            if frame_index < 0 or frame_index >= len(transforms):
                continue

            used_matrix = transforms[frame_index]
            udx, udy, urot, uscale = flow.decompose_similarity(used_matrix)
            (
                accepted,
                direct_matrix,
                _tracked_points,
                number_tracked,
                number_inliers,
            ) = direct_registration(
                reference_gray,
                cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY),
                reference_points,
                maximum_translation,
                maximum_rotation,
                minimum_scale,
                maximum_scale,
                minimum_inliers,
            ) if reference_points is not None else (
                False,
                flow.identity_matrix(),
                None,
                0,
                0,
            )

            print(
                f"  frame {frame_index:4d} "
                f"t={frame_index / sample_fps:7.3f}s"
            )
            print(
                "    pipeline   : "
                f"dx={udx:+.3f}px dy={udy:+.3f}px "
                f"rot={urot:+.4f}deg scale={uscale:.6f}"
            )
            if accepted:
                ddx, ddy, drot, dscale = flow.decompose_similarity(
                    direct_matrix
                )
                print(
                    "    direct     : "
                    f"tracked={number_tracked} inliers={number_inliers} "
                    f"dx={ddx:+.3f}px dy={ddy:+.3f}px "
                    f"rot={drot:+.4f}deg scale={dscale:.6f}"
                )
                residual = np.linalg.inv(direct_matrix) @ used_matrix
                rdx, rdy, rrot, rscale = flow.decompose_similarity(residual)
                print(
                    "    used/direct: "
                    f"dx={rdx:+.3f}px dy={rdy:+.3f}px "
                    f"rot={rrot:+.4f}deg scale={rscale:.6f}"
                )
            else:
                print("    direct     : unavailable")

        return result

    print("Experimental ring-only optical-flow tracking:")
    print(f"  ring padding     : {ring_args.flow_ring_padding:.3f}")
    print("  box interior     : fully excluded")
    print(f"  registration mode: {ring_args.flow_registration}")
    if ring_args.motion_diagnostic_time:
        print(
            "  motion diagnostics: "
            + ", ".join(
                f"{value:.3f}s"
                for value in ring_args.motion_diagnostic_time
            )
        )
        print(f"  diagnostic dir   : {ring_args.flow_diagnostic_dir}")
    print()

    saved_argv = sys.argv
    flow.make_feature_mask = make_ring_feature_mask
    flow.precompute_motion = precompute_motion_with_registration
    sys.argv = [saved_argv[0], *remaining]

    try:
        return segmentshape.main()
    finally:
        flow.make_feature_mask = original_make_feature_mask
        flow.precompute_motion = original_precompute_motion
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
