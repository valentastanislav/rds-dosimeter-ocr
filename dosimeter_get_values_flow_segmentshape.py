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
from pathlib import Path

import cv2
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

    parser.add_argument(
        "--flow-feature-padding",
        type=float,
        default=None,
        help=(
            "override the optical-flow feature-mask padding around the "
            "reference display box as a fraction of box size; the normal "
            "flow code uses 0.08"
        ),
    )

    parser.add_argument(
        "--preview-time",
        type=float,
        action="append",
        default=None,
        metavar="SECONDS",
        help=(
            "write an exact cached-main-pass geometry preview at this time; "
            "may be repeated"
        ),
    )

    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("rds200_segmentshape_previews"),
        help=(
            "directory for --preview-time images "
            "(default: rds200_segmentshape_previews)"
        ),
    )

    return parser.parse_known_args(argv)


def draw_preview_geometry(
    display: np.ndarray,
    profile,
) -> np.ndarray:
    """Draw the exact digit boxes and per-digit segment polygons."""

    overlay = cv2.cvtColor(
        display,
        cv2.COLOR_GRAY2BGR,
    )

    per_digit = getattr(
        profile,
        "digit_segment_polygons",
        None,
    )

    for digit_index, (x1, y1, x2, y2) in enumerate(
        profile.digit_boxes
    ):
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 180, 0),
            1,
        )

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        scale_x = box_width / 65.0
        scale_y = box_height / 130.0

        polygons = (
            per_digit[digit_index]
            if per_digit is not None
            else profile.segment_polygons
        )

        for name in tight.core.SEGMENT_ORDER:
            local = np.asarray(
                polygons[name],
                dtype=float,
            )
            points = np.empty_like(
                local,
                dtype=np.int32,
            )
            points[:, 0] = np.rint(
                x1 + local[:, 0] * scale_x
            ).astype(np.int32)
            points[:, 1] = np.rint(
                y1 + local[:, 1] * scale_y
            ).astype(np.int32)

            cv2.polylines(
                overlay,
                [points],
                True,
                (0, 180, 0),
                1,
            )

    return overlay


def write_exact_previews(
    display_cache: dict[int, np.ndarray],
    sample_fps: float,
    profile,
    times: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not display_cache:
        print("Exact preview: display cache is empty.")
        return

    maximum_index = max(display_cache)

    print()
    print("Exact cached-main-pass previews:")

    for requested_time in times:
        frame_index = int(
            round(requested_time * sample_fps)
        )
        frame_index = max(
            0,
            min(frame_index, maximum_index),
        )
        actual_time = frame_index / sample_fps

        display = display_cache.get(frame_index)
        if display is None:
            print(
                f"  t={requested_time:.3f}s -> frame {frame_index}: missing"
            )
            continue

        overlay = draw_preview_geometry(
            display,
            profile,
        )

        header_height = 54
        canvas = np.full(
            (
                overlay.shape[0] + header_height,
                overlay.shape[1],
                3,
            ),
            255,
            dtype=np.uint8,
        )
        canvas[header_height:] = overlay

        cv2.putText(
            canvas,
            (
                f"EXACT CACHED MAIN PASS frame={frame_index} "
                f"t={actual_time:.3f}s"
            ),
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 80, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"requested t={requested_time:.3f}s",
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        filename = (
            f"frame_{frame_index:04d}_"
            f"t{actual_time:08.3f}.jpg"
        )
        path = output_dir / filename

        if not cv2.imwrite(
            str(path),
            canvas,
        ):
            raise RuntimeError(
                f"Could not write exact preview: {path}"
            )

        print(
            f"  requested {requested_time:.3f}s -> "
            f"{actual_time:.3f}s: {path}"
        )


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

    if (
        shape_args.flow_feature_padding is not None
        and shape_args.flow_feature_padding < 0.0
    ):
        print(
            "Error: --flow-feature-padding must be non-negative.",
            file=sys.stderr,
        )
        return 1

    original_transform = tight.transform_polygon
    original_flow_main = tight.diag.flow.main
    original_make_fixed_profile = (
        tight.diag.flow.fixed_app.make_fixed_profile
    )
    original_make_feature_mask = (
        tight.diag.flow.make_feature_mask
    )
    captured_fixed_profile = {
        "profile": None,
    }

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

    def make_fixed_profile_with_capture(profile, grid):
        fixed_profile = original_make_fixed_profile(
            profile,
            grid,
        )
        captured_fixed_profile["profile"] = fixed_profile
        return fixed_profile

    def make_feature_mask_with_padding(
        frame_shape,
        box,
        profile,
    ):
        if shape_args.flow_feature_padding is None:
            return original_make_feature_mask(
                frame_shape,
                box,
                profile,
            )

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        x, y, width, height = box
        fraction = shape_args.flow_feature_padding

        pad_x = int(round(fraction * width))
        pad_y = int(round(fraction * height))

        outer_x1 = max(0, x - pad_x)
        outer_y1 = max(0, y - pad_y)
        outer_x2 = min(frame_width, x + width + pad_x)
        outer_y2 = min(frame_height, y + height + pad_y)

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

        (
            exclusion_x1,
            exclusion_y1,
            exclusion_x2,
            exclusion_y2,
        ) = profile.flow_feature_exclusion_box

        inner_x1 = int(round(x + exclusion_x1 * width))
        inner_x2 = int(round(x + exclusion_x2 * width))
        inner_y1 = int(round(y + exclusion_y1 * height))
        inner_y2 = int(round(y + exclusion_y2 * height))

        cv2.rectangle(
            mask,
            (inner_x1, inner_y1),
            (inner_x2, inner_y2),
            0,
            -1,
        )

        return mask

    def flow_main_with_exact_previews(*args, **kwargs):
        if not shape_args.preview_time:
            return original_flow_main(*args, **kwargs)

        source_profile = kwargs.get("profile_override")
        if source_profile is None:
            source_profile = tight.core.PROFILES["rds200"]

        previous_observer = kwargs.get(
            "display_cache_observer"
        )

        def observer(
            display_cache,
            total_frames,
            sample_fps,
        ):
            del total_frames

            preview_profile = (
                captured_fixed_profile["profile"]
                if captured_fixed_profile["profile"] is not None
                else source_profile
            )

            write_exact_previews(
                display_cache,
                sample_fps,
                preview_profile,
                shape_args.preview_time,
                shape_args.preview_dir,
            )

            if previous_observer is not None:
                previous_observer(
                    display_cache,
                    len(display_cache),
                    sample_fps,
                )

        kwargs["display_cache_observer"] = observer
        return original_flow_main(*args, **kwargs)

    print("Experimental RDS-200 segment shape refinement:")
    print(f"  x thickness : {shape_args.segment_x_thickness:.3f}")
    print(f"  y thickness : {shape_args.segment_y_thickness:.3f}")
    print(f"  x shear     : {shape_args.segment_x_shear:+.3f}")
    print(f"  shear center: {shape_args.segment_shear_center:.2f}")
    if shape_args.flow_feature_padding is not None:
        print(
            "  flow padding : "
            f"{shape_args.flow_feature_padding:.3f}"
        )
    if shape_args.preview_time:
        print(
            "  preview times: "
            + ", ".join(
                f"{value:.3f}s"
                for value in shape_args.preview_time
            )
        )
        print(f"  preview dir  : {shape_args.preview_dir}")
    print()

    saved_argv = sys.argv
    tight.transform_polygon = transform_polygon_with_shape
    tight.diag.flow.main = flow_main_with_exact_previews
    tight.diag.flow.fixed_app.make_fixed_profile = (
        make_fixed_profile_with_capture
    )
    tight.diag.flow.make_feature_mask = (
        make_feature_mask_with_padding
    )
    sys.argv = [saved_argv[0], *remaining]

    try:
        return tight.main()
    finally:
        tight.transform_polygon = original_transform
        tight.diag.flow.main = original_flow_main
        tight.diag.flow.fixed_app.make_fixed_profile = (
            original_make_fixed_profile
        )
        tight.diag.flow.make_feature_mask = (
            original_make_feature_mask
        )
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
