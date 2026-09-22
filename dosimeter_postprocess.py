#!/usr/bin/env python3
"""Post-process dosimeter interval CSV files with confidence and sigma cuts."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Interval:
    row_number: int
    start_s: float
    end_s: float
    value: float
    confidence: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class WeightedStats:
    duration_s: float
    mean: float
    variance: float
    std_dev: float
    minimum: float
    maximum: float


def _parse_finite_float(raw: str | None, column: str, row_number: int) -> float:
    if raw is None or raw.strip() == "":
        raise ValueError(f"row {row_number}: missing {column}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: invalid {column} value {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: non-finite {column} value {raw!r}")
    return value


def read_intervals(path: Path) -> list[Interval]:
    required = {"start_s", "end_s", "value", "confidence"}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                "input CSV is missing required column(s): " + ", ".join(missing)
            )

        intervals: list[Interval] = []
        for row_number, row in enumerate(reader, start=2):
            start_s = _parse_finite_float(row.get("start_s"), "start_s", row_number)
            end_s = _parse_finite_float(row.get("end_s"), "end_s", row_number)
            value = _parse_finite_float(row.get("value"), "value", row_number)
            confidence = _parse_finite_float(
                row.get("confidence"), "confidence", row_number
            )

            if end_s <= start_s:
                raise ValueError(
                    f"row {row_number}: end_s must be greater than start_s"
                )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"row {row_number}: confidence must be between 0 and 1"
                )

            intervals.append(
                Interval(
                    row_number=row_number,
                    start_s=start_s,
                    end_s=end_s,
                    value=value,
                    confidence=confidence,
                )
            )

    if not intervals:
        raise ValueError("input CSV contains no data rows")
    return intervals


def calculate_weighted_stats(intervals: Sequence[Interval]) -> WeightedStats:
    if not intervals:
        raise ValueError("cannot calculate statistics with no intervals")

    total_duration = sum(interval.duration_s for interval in intervals)
    if total_duration <= 0.0:
        raise ValueError("total included interval duration must be positive")

    mean = (
        sum(interval.duration_s * interval.value for interval in intervals)
        / total_duration
    )
    variance = (
        sum(
            interval.duration_s * (interval.value - mean) ** 2
            for interval in intervals
        )
        / total_duration
    )

    return WeightedStats(
        duration_s=total_duration,
        mean=mean,
        variance=variance,
        std_dev=math.sqrt(variance),
        minimum=min(interval.value for interval in intervals),
        maximum=max(interval.value for interval in intervals),
    )


def sigma_clip(
    intervals: Sequence[Interval],
    sigma: float,
    max_iterations: int = 100,
) -> tuple[list[Interval], list[Interval]]:
    current = list(intervals)
    rejected: list[Interval] = []

    for _ in range(max_iterations):
        if len(current) < 2:
            break

        stats = calculate_weighted_stats(current)
        if stats.std_dev == 0.0:
            break

        limit = sigma * stats.std_dev
        kept = [
            interval
            for interval in current
            if abs(interval.value - stats.mean) <= limit
        ]
        removed = [
            interval
            for interval in current
            if abs(interval.value - stats.mean) > limit
        ]

        if not removed:
            break
        if not kept:
            raise ValueError(
                "sigma clipping rejected all intervals; "
                "choose a larger --sigma-clip value"
            )

        rejected.extend(removed)
        current = kept
    else:
        raise RuntimeError(
            f"sigma clipping did not converge after {max_iterations} iterations"
        )

    return current, rejected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate time-weighted dosimeter statistics after a confidence "
            "cut and iterative time-weighted sigma clipping."
        )
    )
    parser.add_argument(
        "csv",
        type=Path,
        help=(
            "interval/debug-index CSV containing start_s, end_s, value, "
            "and confidence columns"
        ),
    )
    parser.add_argument(
        "--summary-min-confidence",
        type=float,
        default=0.20,
        help=(
            "exclude intervals below this confidence before sigma clipping "
            "(default: 0.20)"
        ),
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        required=True,
        metavar="X",
        help=(
            "iteratively reject values farther than X time-weighted standard "
            "deviations from the time-weighted mean"
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.csv.is_file():
        raise ValueError(f"input CSV does not exist: {args.csv}")
    if not 0.0 <= args.summary_min_confidence <= 1.0:
        raise ValueError("--summary-min-confidence must be between 0 and 1")
    if not math.isfinite(args.sigma_clip) or args.sigma_clip <= 0.0:
        raise ValueError("--sigma-clip must be a positive finite number")


def format_metric(value: float) -> str:
    return f"{value:10.5g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
        intervals = read_intervals(args.csv)
        total_duration = sum(interval.duration_s for interval in intervals)

        confidence_kept = [
            interval
            for interval in intervals
            if interval.confidence >= args.summary_min_confidence
        ]
        confidence_rejected = len(intervals) - len(confidence_kept)

        if not confidence_kept:
            raise ValueError(
                "no intervals remain after --summary-min-confidence cut"
            )

        selected, sigma_rejected = sigma_clip(
            confidence_kept,
            args.sigma_clip,
        )
        stats = calculate_weighted_stats(selected)

        print("After post-processing:")
        print(
            f"Summary confidence cut:  {args.summary_min_confidence:10.3f}"
        )
        print(f"Sigma clip:              {args.sigma_clip:10.3f} sigma")
        print(
            f"Intervals used:          {len(selected):10d} / {len(intervals)}"
        )
        print(
            f"Duration used:           {stats.duration_s:10.3f} s / "
            f"{total_duration:.3f} s "
            f"({100.0 * stats.duration_s / total_duration:.2f} %)"
        )
        print(
            f"Rejected by confidence:  {confidence_rejected:10d}"
        )
        print(
            f"Rejected by sigma clip:  {len(sigma_rejected):10d}"
        )
        print()
        print(
            f"Time-weighted mean:      {format_metric(stats.mean)}"
        )
        print(
            f"Time-weighted variance:  {format_metric(stats.variance)}"
        )
        print(
            f"Time-weighted std. dev.: {format_metric(stats.std_dev)}"
        )
        print(
            f"Minimum:                 {format_metric(stats.minimum)}"
        )
        print(
            f"Maximum:                 {format_metric(stats.maximum)}"
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
