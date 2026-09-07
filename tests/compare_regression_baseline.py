#!/usr/bin/env python3
"""Compare raw OCR samples with the committed RDS-200 regression baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


BASELINE_PATH = (
    Path(__file__).resolve().parent
    / "baseline"
    / "IMG_1151_consensus2_values.csv"
)
MAX_MISMATCHES_TO_SHOW = 10


@dataclass(frozen=True)
class RawSample:
    time_s: Decimal
    value: Decimal | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a raw OCR CSV with the committed IMG_1151 "
            "consensus2 sample sequence."
        )
    )
    parser.add_argument(
        "raw_csv",
        type=Path,
        help="newly produced raw OCR CSV containing time_s and value columns",
    )
    return parser.parse_args()


def parse_decimal(text: str, path: Path, row_number: int, column: str) -> Decimal:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path}:{row_number}: blank {column}")

    try:
        result = Decimal(stripped)
    except InvalidOperation as exc:
        raise ValueError(
            f"{path}:{row_number}: invalid {column}: {text!r}"
        ) from exc

    if not result.is_finite():
        raise ValueError(f"{path}:{row_number}: non-finite {column}: {text!r}")
    return result


def read_samples(path: Path) -> list[RawSample]:
    if not path.is_file():
        raise ValueError(f"CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        fieldnames = [name.strip() for name in reader.fieldnames]
        for required in ("time_s", "value"):
            if fieldnames.count(required) != 1:
                raise ValueError(
                    f"{path}: expected exactly one {required!r} column"
                )

        samples: list[RawSample] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number}: too many CSV fields")

            time_text = row.get("time_s")
            value_text = row.get("value")
            if time_text is None or value_text is None:
                raise ValueError(f"{path}:{row_number}: missing required CSV field")

            time_s = parse_decimal(time_text, path, row_number, "time_s")
            value = (
                None
                if not value_text.strip()
                else parse_decimal(value_text, path, row_number, "value")
            )
            samples.append(RawSample(time_s=time_s, value=value))

    return samples


def format_sample(sample: RawSample | None) -> tuple[str, str]:
    if sample is None:
        return "<missing>", "<missing>"
    value = "<blank>" if sample.value is None else str(sample.value)
    return str(sample.time_s), value


def main() -> int:
    args = parse_args()

    try:
        expected = read_samples(BASELINE_PATH)
        actual = read_samples(args.raw_csv)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    mismatches: list[tuple[int, RawSample | None, RawSample | None]] = []
    for index in range(max(len(expected), len(actual))):
        expected_sample = expected[index] if index < len(expected) else None
        actual_sample = actual[index] if index < len(actual) else None
        if expected_sample != actual_sample:
            mismatches.append((index, expected_sample, actual_sample))

    print(f"Baseline:             {BASELINE_PATH}")
    print(f"Actual:               {args.raw_csv}")
    print(f"Expected samples:     {len(expected)}")
    print(f"Actual samples:       {len(actual)}")
    print(f"Mismatching samples:  {len(mismatches)}")

    if mismatches:
        print()
        print("First mismatches:")
        for index, expected_sample, actual_sample in mismatches[
            :MAX_MISMATCHES_TO_SHOW
        ]:
            expected_time, expected_value = format_sample(expected_sample)
            actual_time, actual_value = format_sample(actual_sample)
            print(
                f"  sample {index}: "
                f"time expected={expected_time} actual={actual_time}; "
                f"value expected={expected_value} actual={actual_value}"
            )
        return 1

    print("Result:               exact parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
