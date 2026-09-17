#!/usr/bin/env python3
"""Ground-truth-blind experimental sweep of an RDS-200 digit grid.

The video pipeline is run once.  Every candidate is then measured from the
exact cached, stabilized, and rectified main-pass displays.  Candidate
generation and ranking use no expected values or ground-truth data.

The experimental ranking is lexicographic, in this order:

1. recognized complete samples (descending),
2. p10 complete-sample minimum pattern margin (descending),
3. median complete-sample minimum pattern margin (descending),
4. recognized individual digits (descending),
5. median decoder confidence (descending),
6. mean decoder confidence (descending),
7. total absolute edge perturbation (ascending), then edge offsets.

The final tie-breakers prefer the least perturbed geometry, not any known
displayed value.  All component measurements are retained in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_fixedgrid as fixed
import dosimeter_get_values_flow as flow
import dosimeter_get_values_flow_diag as flow_diag
import dosimeter_get_values_flow_tightsegments as tightsegments
import dosimeter_get_values_rectified as rectified


FROZEN_SEGMENT_Y_STRETCH = 1.45
FROZEN_SEGMENT_Y_CENTER = 67.0
FROZEN_SEGMENT_X_STRETCH = 1.0
FROZEN_SEGMENT_X_CENTER = 32.0
FROZEN_SEGMENT_X_SHIFT = 4.0
FROZEN_DIGIT_X_OFFSETS = (0, 0, 3)
FROZEN_FILTER_WINDOW = 1
FROZEN_MODE_WINDOW = 1
FROZEN_SAMPLE_FPS = 5.0


@dataclass(frozen=True)
class GridCandidate:
    dx1: int
    dy1: int
    dx2: int
    dy2: int
    grid: fixed.Grid

    @property
    def perturbation(self) -> int:
        return abs(self.dx1) + abs(self.dy1) + abs(self.dx2) + abs(self.dy2)


@dataclass(frozen=True)
class CandidateMetrics:
    candidate: GridCandidate
    total_samples: int
    display_samples: int
    recognized_samples: int
    recognized_fraction: float
    recognized_digits: int
    total_digits: int
    digit_recognition_fraction: float
    ambiguous_digits: int
    ambiguous_digit_fraction: float
    p10_min_margin: float
    median_min_margin: float
    mean_min_margin: float
    mean_digit_confidence: float
    median_digit_confidence: float
    recognized_digit1: int
    recognized_digit2: int
    recognized_digit3: int
    p10_margin_digit1: float
    p10_margin_digit2: float
    p10_margin_digit3: float
    median_confidence_digit1: float
    median_confidence_digit2: float
    median_confidence_digit3: float
    evaluation_seconds: float
    experimental_rank: int = 0


def offset_grid_by_pixels(
    base_grid: fixed.Grid,
    display_width: int,
    display_height: int,
    dx1: int,
    dy1: int,
    dx2: int,
    dy2: int,
) -> fixed.Grid | None:
    """Apply integer rectified-display pixel offsets to normalized edges."""
    if display_width <= 0 or display_height <= 0:
        raise ValueError("display dimensions must be positive")
    if (dx1, dy1, dx2, dy2) == (0, 0, 0, 0):
        return base_grid

    x1 = base_grid[0] * display_width + dx1
    y1 = base_grid[1] * display_height + dy1
    x2 = base_grid[2] * display_width + dx2
    y2 = base_grid[3] * display_height + dy2

    if not (
        0.0 <= x1 < x2 <= display_width
        and 0.0 <= y1 < y2 <= display_height
    ):
        return None

    return (
        x1 / display_width,
        y1 / display_height,
        x2 / display_width,
        y2 / display_height,
    )


def generate_candidates(
    base_grid: fixed.Grid,
    display_width: int,
    display_height: int,
    radius_px: int,
) -> list[GridCandidate]:
    if radius_px < 0:
        raise ValueError("search radius must be non-negative")

    offsets = range(-radius_px, radius_px + 1)
    candidates: list[GridCandidate] = []
    for dx1, dy1, dx2, dy2 in itertools.product(offsets, repeat=4):
        grid = offset_grid_by_pixels(
            base_grid,
            display_width,
            display_height,
            dx1,
            dy1,
            dx2,
            dy2,
        )
        if grid is None:
            continue
        candidates.append(GridCandidate(dx1, dy1, dx2, dy2, grid))

    return candidates


def make_frozen_rds200_profile(profile: core.Profile) -> core.Profile:
    """Return the frozen tight-segment profile without mutating the input."""
    transformed_polygons = {
        name: tightsegments.transform_polygon(
            polygon,
            x_stretch=FROZEN_SEGMENT_X_STRETCH,
            x_center=FROZEN_SEGMENT_X_CENTER,
            x_shift=FROZEN_SEGMENT_X_SHIFT,
            y_stretch=FROZEN_SEGMENT_Y_STRETCH,
            y_center=FROZEN_SEGMENT_Y_CENTER,
            y_shift=0.0,
        )
        for name, polygon in profile.segment_polygons.items()
    }
    digit_polygons = tightsegments.make_digit_segment_polygons(
        transformed_polygons,
        FROZEN_DIGIT_X_OFFSETS,
    )
    return replace(
        profile,
        segment_polygons={
            name: polygon.copy()
            for name, polygon in transformed_polygons.items()
        },
        digit_segment_polygons=tuple(
            {
                name: polygon.copy()
                for name, polygon in digit.items()
            }
            for digit in digit_polygons
        ),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def evaluate_candidate(
    candidate: GridCandidate,
    displays: dict[int, np.ndarray],
    total_samples: int,
    base_profile: core.Profile,
) -> CandidateMetrics:
    started = time.perf_counter()
    profile = fixed.make_fixed_profile(base_profile, candidate.grid)
    extractor = fixed.make_fixed_extract_darkness(
        "auto",
        profile.fixedgrid_primary_measurement,
    )
    digit_count = len(profile.digit_boxes)
    if digit_count != 3:
        raise ValueError("RDS-200 grid sweep requires exactly three digits")

    darkness = np.full((total_samples, digit_count, 7), np.nan, dtype=float)
    for frame_index, display in displays.items():
        if 0 <= frame_index < total_samples:
            darkness[frame_index] = extractor(display, profile, ())

    filtered = core.temporal_filter(
        darkness,
        FROZEN_FILTER_WINDOW,
        profile.temporal_filter,
    )
    patterns = profile.digit_patterns or core.STANDARD_DIGIT_PATTERNS
    minimum_confidence = profile.default_min_confidence

    recognized_samples = 0
    recognized_by_position = [0, 0, 0]
    confidence_by_position: list[list[float]] = [[], [], []]
    margin_by_position: list[list[float]] = [[], [], []]
    all_confidences: list[float] = []
    complete_sample_min_margins: list[float] = []

    for sample_levels in filtered:
        sample_recognized: list[bool] = []
        sample_margins: list[float] = []
        for position, levels in enumerate(sample_levels):
            if not np.all(np.isfinite(levels)):
                all_confidences.append(0.0)
                sample_recognized.append(False)
                continue

            digit, confidence = core.decode_pattern_digit(
                levels,
                patterns,
                rds200_pattern_refinement=True,
            )
            confidence = float(confidence)
            all_confidences.append(confidence)
            recognized = digit is not None and confidence >= minimum_confidence
            sample_recognized.append(recognized)
            if not recognized:
                continue

            _margin_digit, margin = flow_diag.margin_candidate(levels, patterns)
            recognized_by_position[position] += 1
            confidence_by_position[position].append(confidence)
            margin_by_position[position].append(float(margin))
            sample_margins.append(float(margin))

        if len(sample_recognized) == digit_count and all(sample_recognized):
            recognized_samples += 1
            complete_sample_min_margins.append(min(sample_margins))

    recognized_digits = sum(recognized_by_position)
    total_digits = total_samples * digit_count
    ambiguous_digits = total_digits - recognized_digits

    return CandidateMetrics(
        candidate=candidate,
        total_samples=total_samples,
        display_samples=len(displays),
        recognized_samples=recognized_samples,
        recognized_fraction=recognized_samples / total_samples,
        recognized_digits=recognized_digits,
        total_digits=total_digits,
        digit_recognition_fraction=recognized_digits / total_digits,
        ambiguous_digits=ambiguous_digits,
        ambiguous_digit_fraction=ambiguous_digits / total_digits,
        p10_min_margin=_percentile(complete_sample_min_margins, 10.0),
        median_min_margin=_percentile(complete_sample_min_margins, 50.0),
        mean_min_margin=_mean(complete_sample_min_margins),
        mean_digit_confidence=_mean(all_confidences),
        median_digit_confidence=_percentile(all_confidences, 50.0),
        recognized_digit1=recognized_by_position[0],
        recognized_digit2=recognized_by_position[1],
        recognized_digit3=recognized_by_position[2],
        p10_margin_digit1=_percentile(margin_by_position[0], 10.0),
        p10_margin_digit2=_percentile(margin_by_position[1], 10.0),
        p10_margin_digit3=_percentile(margin_by_position[2], 10.0),
        median_confidence_digit1=_percentile(confidence_by_position[0], 50.0),
        median_confidence_digit2=_percentile(confidence_by_position[1], 50.0),
        median_confidence_digit3=_percentile(confidence_by_position[2], 50.0),
        evaluation_seconds=time.perf_counter() - started,
    )


def _rank_value(value: float) -> float:
    return value if math.isfinite(value) else float("-inf")


def ranking_key(metrics: CandidateMetrics) -> tuple[float | int, ...]:
    candidate = metrics.candidate
    return (
        -metrics.recognized_samples,
        -_rank_value(metrics.p10_min_margin),
        -_rank_value(metrics.median_min_margin),
        -metrics.recognized_digits,
        -_rank_value(metrics.median_digit_confidence),
        -_rank_value(metrics.mean_digit_confidence),
        candidate.perturbation,
        candidate.dx1,
        candidate.dy1,
        candidate.dx2,
        candidate.dy2,
    )


def rank_candidates(metrics: Sequence[CandidateMetrics]) -> list[CandidateMetrics]:
    return [
        replace(item, experimental_rank=rank)
        for rank, item in enumerate(sorted(metrics, key=ranking_key), start=1)
    ]


def _format_float(value: float) -> str:
    return "" if not math.isfinite(value) else format(value, ".12g")


def write_candidate_csv(path: Path, ranked: Sequence[CandidateMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "experimental_rank",
        "dx1",
        "dy1",
        "dx2",
        "dy2",
        "grid_x1",
        "grid_y1",
        "grid_x2",
        "grid_y2",
        "total_samples",
        "display_samples",
        "recognized_samples",
        "recognized_fraction",
        "recognized_digits",
        "total_digits",
        "digit_recognition_fraction",
        "ambiguous_digits",
        "ambiguous_digit_fraction",
        "p10_min_margin",
        "median_min_margin",
        "mean_min_margin",
        "mean_digit_confidence",
        "median_digit_confidence",
        "recognized_digit1",
        "recognized_digit2",
        "recognized_digit3",
        "p10_margin_digit1",
        "p10_margin_digit2",
        "p10_margin_digit3",
        "median_confidence_digit1",
        "median_confidence_digit2",
        "median_confidence_digit3",
        "evaluation_seconds",
        "experimental_score_tuple",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranked:
            candidate = item.candidate
            grid = candidate.grid
            score_tuple = (
                f"{item.recognized_samples}|"
                f"{_format_float(item.p10_min_margin)}|"
                f"{_format_float(item.median_min_margin)}|"
                f"{item.recognized_digits}|"
                f"{_format_float(item.median_digit_confidence)}|"
                f"{_format_float(item.mean_digit_confidence)}"
            )
            writer.writerow(
                {
                    "experimental_rank": item.experimental_rank,
                    "dx1": candidate.dx1,
                    "dy1": candidate.dy1,
                    "dx2": candidate.dx2,
                    "dy2": candidate.dy2,
                    "grid_x1": _format_float(grid[0]),
                    "grid_y1": _format_float(grid[1]),
                    "grid_x2": _format_float(grid[2]),
                    "grid_y2": _format_float(grid[3]),
                    "total_samples": item.total_samples,
                    "display_samples": item.display_samples,
                    "recognized_samples": item.recognized_samples,
                    "recognized_fraction": _format_float(item.recognized_fraction),
                    "recognized_digits": item.recognized_digits,
                    "total_digits": item.total_digits,
                    "digit_recognition_fraction": _format_float(
                        item.digit_recognition_fraction
                    ),
                    "ambiguous_digits": item.ambiguous_digits,
                    "ambiguous_digit_fraction": _format_float(
                        item.ambiguous_digit_fraction
                    ),
                    "p10_min_margin": _format_float(item.p10_min_margin),
                    "median_min_margin": _format_float(item.median_min_margin),
                    "mean_min_margin": _format_float(item.mean_min_margin),
                    "mean_digit_confidence": _format_float(
                        item.mean_digit_confidence
                    ),
                    "median_digit_confidence": _format_float(
                        item.median_digit_confidence
                    ),
                    "recognized_digit1": item.recognized_digit1,
                    "recognized_digit2": item.recognized_digit2,
                    "recognized_digit3": item.recognized_digit3,
                    "p10_margin_digit1": _format_float(item.p10_margin_digit1),
                    "p10_margin_digit2": _format_float(item.p10_margin_digit2),
                    "p10_margin_digit3": _format_float(item.p10_margin_digit3),
                    "median_confidence_digit1": _format_float(
                        item.median_confidence_digit1
                    ),
                    "median_confidence_digit2": _format_float(
                        item.median_confidence_digit2
                    ),
                    "median_confidence_digit3": _format_float(
                        item.median_confidence_digit3
                    ),
                    "evaluation_seconds": _format_float(item.evaluation_seconds),
                    "experimental_score_tuple": score_tuple,
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground-truth-blind experimental RDS-200 digit-grid sweep. "
            "All non-grid OCR and tracking parameters are frozen."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--track-time", type=float, required=True)
    parser.add_argument(
        "--reference-box",
        type=flow.parse_reference_box,
        required=True,
    )
    parser.add_argument("--quad", type=rectified.parse_quad, required=True)
    parser.add_argument("--grid", type=fixed.parse_grid, required=True)
    parser.add_argument("--radius-px", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    if args.radius_px < 0:
        parser.error("--radius-px must be non-negative")
    if args.top_n <= 0:
        parser.error("--top-n must be positive")
    return args


def acquire_cached_displays(
    args: argparse.Namespace,
    profile: core.Profile,
) -> tuple[dict[int, np.ndarray], int, float]:
    captured_displays: dict[int, np.ndarray] = {}
    captured_total = 0
    captured_fps = 0.0

    def observe(
        displays: dict[int, np.ndarray],
        total_samples: int,
        sample_fps: float,
    ) -> None:
        nonlocal captured_total, captured_fps
        captured_displays.update(displays)
        captured_total = total_samples
        captured_fps = sample_fps

    with tempfile.TemporaryDirectory(prefix="rds200-grid-refinement-") as directory:
        temporary = Path(directory)
        flow_argv = [
            str(args.video),
            str(temporary / "base_intervals.csv"),
            "--profile",
            "rds200",
            "--track-time",
            format(args.track_time, ".17g"),
            "--reference-box",
            flow.format_reference_box(args.reference_box),
            "--quad",
            rectified.format_quad(args.quad),
            "--grid",
            fixed.format_grid(args.grid),
            "--flow-redetect-every",
            "1",
            "--filter-window",
            str(FROZEN_FILTER_WINDOW),
            "--mode-window",
            str(FROZEN_MODE_WINDOW),
            "--sample-fps",
            format(FROZEN_SAMPLE_FPS, "g"),
            "--decimal-places",
            "auto",
            "--contrast",
            "auto",
            "--rds200-pattern-refinement",
            "--raw-output",
            str(temporary / "base_raw.csv"),
        ]
        result = flow.main(
            flow_argv,
            profile_override=profile,
            display_cache_observer=observe,
        )
        if result != 0:
            raise RuntimeError(f"base flow run failed with exit status {result}")

    if captured_total <= 0 or not captured_displays:
        raise RuntimeError("base flow run did not expose cached rectified displays")
    return captured_displays, captured_total, captured_fps


def print_summary(
    ranked: Sequence[CandidateMetrics],
    base: CandidateMetrics,
    display_width: int,
    display_height: int,
    radius_px: int,
    output_csv: Path,
    elapsed: float,
    top_n: int,
) -> None:
    print()
    print("Experimental ground-truth-blind grid sweep:")
    print(f"  candidates evaluated : {len(ranked)}")
    print(f"  rectified display     : {display_width} x {display_height}")
    print(f"  search radius         : {radius_px} px per edge")
    print(f"  base-grid rank        : {base.experimental_rank}")
    print(
        "  base recognized      : "
        f"{base.recognized_samples}/{base.total_samples} "
        f"({100.0 * base.recognized_fraction:.2f}%)"
    )
    print(f"  output CSV            : {output_csv}")
    print(f"  elapsed               : {elapsed:.2f} s")
    print(
        "  ranking               : recognized samples, p10/median minimum "
        "margin, recognized digits, confidence; then least perturbation"
    )
    print()
    print(f"Top {min(top_n, len(ranked))} internally ranked candidates:")
    print(" rank   dx1 dy1 dx2 dy2   recognized       p10 margin   median margin")
    for item in ranked[:top_n]:
        candidate = item.candidate
        print(
            f" {item.experimental_rank:4d}  "
            f"{candidate.dx1:+3d} {candidate.dy1:+3d} "
            f"{candidate.dx2:+3d} {candidate.dy2:+3d}   "
            f"{item.recognized_samples:4d}/{item.total_samples:<4d} "
            f"{100.0 * item.recognized_fraction:7.2f}%   "
            f"{item.p10_min_margin:10.4f}   "
            f"{item.median_min_margin:13.4f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        source_profile = core.PROFILES["rds200"]
        frozen_profile = make_frozen_rds200_profile(source_profile)
        displays, total_samples, sample_fps = acquire_cached_displays(
            args,
            frozen_profile,
        )
        first_display = displays[min(displays)]
        display_height, display_width = first_display.shape[:2]
        if any(
            display.shape[:2] != (display_height, display_width)
            for display in displays.values()
        ):
            raise RuntimeError("cached rectified displays have inconsistent sizes")
        if not math.isclose(sample_fps, FROZEN_SAMPLE_FPS):
            raise RuntimeError(
                f"unexpected sampled frame rate {sample_fps:g}; "
                f"expected {FROZEN_SAMPLE_FPS:g}"
            )

        candidates = generate_candidates(
            args.grid,
            display_width,
            display_height,
            args.radius_px,
        )
        metrics: list[CandidateMetrics] = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            metrics.append(evaluate_candidate(
                candidate,
                displays,
                total_samples,
                frozen_profile,
            ))
            if candidate_index % 50 == 0 or candidate_index == len(candidates):
                print(
                    f"Evaluated {candidate_index}/{len(candidates)} grid candidates...",
                    file=sys.stderr,
                )
        ranked = rank_candidates(metrics)
        base = next(
            item
            for item in ranked
            if (
                item.candidate.dx1,
                item.candidate.dy1,
                item.candidate.dx2,
                item.candidate.dy2,
            )
            == (0, 0, 0, 0)
        )
        write_candidate_csv(args.output_csv, ranked)
        print_summary(
            ranked,
            base,
            display_width,
            display_height,
            args.radius_px,
            args.output_csv,
            time.perf_counter() - started,
            args.top_n,
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
