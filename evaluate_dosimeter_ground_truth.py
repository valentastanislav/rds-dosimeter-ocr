#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TruthInterval:
    number: int
    start_s: float
    value: float


@dataclass
class RawSample:
    time_s: float
    value: float | None
    confidence: float


@dataclass
class EvaluatedSample:
    time_s: float
    interval: int
    truth: float
    predicted: float | None
    confidence: float
    status: str


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare dosimeter raw OCR samples against manually "
            "determined ground-truth intervals."
        )
    )

    parser.add_argument(
        "truth",
        type=Path,
        help=(
            "text file containing: "
            "true_interval start_s true_value"
        ),
    )

    parser.add_argument(
        "raw_csv",
        type=Path,
        nargs="+",
        help=(
            "one or more raw OCR CSV files containing "
            "time_s,value,confidence"
        ),
    )

    parser.add_argument(
        "--guard",
        type=float,
        default=0.4,
        help=(
            "ignore +/- this many seconds around each real "
            "transition, except t=0 (default: 0.4)"
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0049,
        help=(
            "absolute difference still considered an exact "
            "display-value match (default: 0.0049)"
        ),
    )

    parser.add_argument(
        "--show-errors",
        type=int,
        default=15,
        help=(
            "number of most frequent wrong true->OCR pairs "
            "to print (default: 15)"
        ),
    )

    return parser.parse_args()


def read_truth(
    path: Path,
) -> list[TruthInterval]:

    if not path.is_file():

        raise RuntimeError(
            f"Ground-truth file does not exist: {path}"
        )

    result: list[TruthInterval] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            stripped = line.strip()

            if not stripped:

                continue

            fields = stripped.split()

            # Header.
            if fields[0].lower().startswith(
                "true"
            ):

                continue

            if len(fields) < 3:

                raise RuntimeError(
                    f"{path}:{line_number}: "
                    "expected at least 3 columns"
                )

            try:

                number = int(
                    fields[0]
                )

                start_s = float(
                    fields[1]
                )

                value = float(
                    fields[2]
                )

            except ValueError as exc:

                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"could not parse line: {stripped}"
                ) from exc

            result.append(
                TruthInterval(
                    number=number,
                    start_s=start_s,
                    value=value,
                )
            )

    if not result:

        raise RuntimeError(
            "Ground-truth file contains no intervals."
        )

    result.sort(
        key=lambda item: item.start_s
    )

    for index in range(
        1,
        len(result),
    ):

        if (
            result[index].start_s
            <= result[index - 1].start_s
        ):

            raise RuntimeError(
                "Ground-truth start times must be strictly increasing."
            )

    return result


def read_raw(
    path: Path,
) -> list[RawSample]:

    if not path.is_file():

        raise RuntimeError(
            f"Raw CSV does not exist: {path}"
        )

    result: list[RawSample] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:

            raise RuntimeError(
                f"{path}: missing CSV header"
            )

        required = {
            "time_s",
            "value",
            "confidence",
        }

        missing = (
            required
            - set(
                reader.fieldnames
            )
        )

        if missing:

            raise RuntimeError(
                f"{path}: missing columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        for row in reader:

            time_s = float(
                row[
                    "time_s"
                ]
            )

            value_text = (
                row[
                    "value"
                ].strip()
            )

            value = (
                None
                if value_text == ""
                else float(
                    value_text
                )
            )

            confidence = float(
                row[
                    "confidence"
                ]
            )

            result.append(
                RawSample(
                    time_s=time_s,
                    value=value,
                    confidence=confidence,
                )
            )

    return result


def truth_at_time(
    time_s: float,
    truth: list[TruthInterval],
) -> TruthInterval | None:

    selected = None

    for interval in truth:

        if (
            interval.start_s
            <= time_s
        ):

            selected = interval

        else:

            break

    return selected


def near_transition(
    time_s: float,
    truth: list[TruthInterval],
    guard: float,
) -> bool:

    # Deliberately skip truth[0].
    #
    # t=0 is not a transition from one known stable
    # display value to another.  Therefore the known
    # geometry problem at the beginning of the video
    # remains visible in the benchmark.
    for interval in truth[
        1:
    ]:

        if (
            abs(
                time_s
                - interval.start_s
            )
            <= guard
        ):

            return True

    return False


def values_equal(
    a: float,
    b: float,
    tolerance: float,
) -> bool:

    return (
        abs(
            a - b
        )
        <= tolerance
    )


def evaluate(
    samples: list[RawSample],
    truth: list[TruthInterval],
    guard: float,
    tolerance: float,
) -> list[EvaluatedSample]:

    result: list[
        EvaluatedSample
    ] = []

    for sample in samples:

        expected = (
            truth_at_time(
                sample.time_s,
                truth,
            )
        )

        if expected is None:

            continue

        if near_transition(
            sample.time_s,
            truth,
            guard,
        ):

            status = (
                "ignored_transition"
            )

        elif sample.value is None:

            status = (
                "missing"
            )

        elif values_equal(
            sample.value,
            expected.value,
            tolerance,
        ):

            status = (
                "correct"
            )

        else:

            status = (
                "wrong"
            )

        result.append(
            EvaluatedSample(
                time_s=sample.time_s,
                interval=expected.number,
                truth=expected.value,
                predicted=sample.value,
                confidence=sample.confidence,
                status=status,
            )
        )

    return result


def write_evaluation_csv(
    raw_path: Path,
    evaluated: list[EvaluatedSample],
) -> Path:

    output_path = (
        raw_path.with_name(
            raw_path.stem
            + ".eval.csv"
        )
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "time_s",
                "true_interval",
                "true_value",
                "predicted_value",
                "confidence",
                "status",
            ]
        )

        for item in evaluated:

            writer.writerow(
                [
                    f"{item.time_s:.3f}",
                    item.interval,
                    f"{item.truth:.6g}",
                    (
                        ""
                        if item.predicted
                        is None
                        else f"{item.predicted:.6g}"
                    ),
                    f"{item.confidence:.4f}",
                    item.status,
                ]
            )

    return output_path


def print_interval_table(
    evaluated: list[EvaluatedSample],
    truth: list[TruthInterval],
) -> None:

    print()

    print(
        "Per true interval:"
    )

    print(
        (
            "  int   start    truth   "
            "eval   correct   wrong   missing   accuracy"
        )
    )

    for truth_interval in truth:

        rows = [
            item
            for item in evaluated
            if (
                item.interval
                == truth_interval.number
                and item.status
                != "ignored_transition"
            )
        ]

        count = len(
            rows
        )

        correct = sum(
            item.status
            == "correct"
            for item in rows
        )

        wrong = sum(
            item.status
            == "wrong"
            for item in rows
        )

        missing = sum(
            item.status
            == "missing"
            for item in rows
        )

        accuracy = (
            correct
            / count
            if count
            else float(
                "nan"
            )
        )

        print(
            (
                f"  {truth_interval.number:3d}"
                f"  {truth_interval.start_s:6.1f}"
                f"  {truth_interval.value:7.2f}"
                f"  {count:5d}"
                f"  {correct:8d}"
                f"  {wrong:6d}"
                f"  {missing:8d}"
                f"  {accuracy:9.1%}"
            )
        )


def print_confusion(
    evaluated: list[EvaluatedSample],
    maximum: int,
) -> None:

    counter = Counter()

    for item in evaluated:

        if (
            item.status
            != "wrong"
            or item.predicted
            is None
        ):

            continue

        counter[
            (
                item.truth,
                item.predicted,
            )
        ] += 1

    print()

    print(
        "Most frequent wrong value pairs:"
    )

    if not counter:

        print(
            "  none"
        )

        return

    for (
        truth_value,
        predicted_value,
    ), count in counter.most_common(
        maximum
    ):

        print(
            (
                f"  true {truth_value:7.3f}"
                f" -> OCR {predicted_value:7.3f}"
                f" : {count:4d} samples"
            )
        )


def print_error_runs(
    evaluated: list[EvaluatedSample],
    maximum: int = 15,
) -> None:

    usable = [
        item
        for item in evaluated
        if item.status
        != "ignored_transition"
    ]

    if len(
        usable
    ) < 2:

        return

    time_steps = [
        b.time_s
        - a.time_s
        for a, b in zip(
            usable[:-1],
            usable[1:],
        )
        if (
            b.time_s
            > a.time_s
        )
    ]

    expected_step = (
        statistics.median(
            time_steps
        )
        if time_steps
        else 0.2
    )

    maximum_gap = (
        1.6
        * expected_step
    )

    runs = []

    current = None

    for item in usable:

        if (
            item.status
            != "wrong"
            or item.predicted
            is None
        ):

            if current is not None:

                runs.append(
                    current
                )

                current = None

            continue

        key = (
            item.truth,
            item.predicted,
        )

        if current is None:

            current = {
                "start": item.time_s,
                "end": item.time_s,
                "truth": item.truth,
                "predicted": item.predicted,
                "count": 1,
            }

            continue

        same_pair = (
            current[
                "truth"
            ]
            == key[0]
            and current[
                "predicted"
            ]
            == key[1]
        )

        close_in_time = (
            item.time_s
            - current[
                "end"
            ]
            <= maximum_gap
        )

        if (
            same_pair
            and close_in_time
        ):

            current[
                "end"
            ] = item.time_s

            current[
                "count"
            ] += 1

        else:

            runs.append(
                current
            )

            current = {
                "start": item.time_s,
                "end": item.time_s,
                "truth": item.truth,
                "predicted": item.predicted,
                "count": 1,
            }

    if current is not None:

        runs.append(
            current
        )

    runs.sort(
        key=lambda item: (
            item[
                "count"
            ]
        ),
        reverse=True,
    )

    print()

    print(
        "Longest stable-region OCR error runs:"
    )

    if not runs:

        print(
            "  none"
        )

        return

    for item in runs[
        :maximum
    ]:

        duration = (
            item[
                "end"
            ]
            - item[
                "start"
            ]
        )

        print(
            (
                f"  {item['start']:6.2f}"
                f" .. {item['end']:6.2f} s"
                f"  ({duration:5.2f} s,"
                f" {item['count']:3d} samples)"
                f" : true {item['truth']:.3f}"
                f" -> OCR {item['predicted']:.3f}"
            )
        )


def summarize(
    raw_path: Path,
    evaluated: list[EvaluatedSample],
    truth: list[TruthInterval],
    show_errors: int,
) -> dict[str, float]:

    ignored = [
        item
        for item in evaluated
        if item.status
        == "ignored_transition"
    ]

    usable = [
        item
        for item in evaluated
        if item.status
        != "ignored_transition"
    ]

    correct = [
        item
        for item in usable
        if item.status
        == "correct"
    ]

    wrong = [
        item
        for item in usable
        if item.status
        == "wrong"
    ]

    missing = [
        item
        for item in usable
        if item.status
        == "missing"
    ]

    recognized = [
        item
        for item in usable
        if item.predicted
        is not None
    ]

    total = len(
        usable
    )

    coverage = (
        len(
            recognized
        )
        / total
        if total
        else 0.0
    )

    overall_accuracy = (
        len(
            correct
        )
        / total
        if total
        else 0.0
    )

    recognized_accuracy = (
        len(
            correct
        )
        / len(
            recognized
        )
        if recognized
        else 0.0
    )

    absolute_errors = [
        abs(
            item.predicted
            - item.truth
        )
        for item in recognized
        if item.predicted
        is not None
    ]

    mean_absolute_error = (
        statistics.mean(
            absolute_errors
        )
        if absolute_errors
        else float(
            "nan"
        )
    )

    print()

    print(
        "=" * 74
    )

    print(
        raw_path
    )

    print(
        "=" * 74
    )

    print(
        (
            f"Raw samples in truth range : "
            f"{len(evaluated)}"
        )
    )

    print(
        (
            f"Ignored near transitions   : "
            f"{len(ignored)}"
        )
    )

    print(
        (
            f"Evaluated stable samples   : "
            f"{total}"
        )
    )

    print(
        (
            f"Recognized stable samples  : "
            f"{len(recognized)} "
            f"({coverage:.2%})"
        )
    )

    print(
        (
            f"Correct stable samples     : "
            f"{len(correct)}"
        )
    )

    print(
        (
            f"Wrong stable samples       : "
            f"{len(wrong)}"
        )
    )

    print(
        (
            f"Missing stable samples     : "
            f"{len(missing)}"
        )
    )

    print(
        (
            f"Overall exact accuracy     : "
            f"{overall_accuracy:.2%}"
        )
    )

    print(
        (
            f"Accuracy when recognized   : "
            f"{recognized_accuracy:.2%}"
        )
    )

    print(
        (
            f"Mean absolute error        : "
            f"{mean_absolute_error:.5g}"
        )
    )

    print_interval_table(
        evaluated,
        truth,
    )

    print_confusion(
        evaluated,
        show_errors,
    )

    print_error_runs(
        evaluated,
        maximum=show_errors,
    )

    return {
        "coverage": coverage,
        "accuracy": overall_accuracy,
        "recognized_accuracy": recognized_accuracy,
        "mae": mean_absolute_error,
        "wrong": float(
            len(
                wrong
            )
        ),
    }


def main() -> int:

    args = parse_args()

    if args.guard < 0:

        raise SystemExit(
            "--guard cannot be negative"
        )

    if args.tolerance < 0:

        raise SystemExit(
            "--tolerance cannot be negative"
        )

    truth = read_truth(
        args.truth
    )

    print(
        "Ground truth:"
    )

    for item in truth:

        print(
            (
                f"  interval {item.number:2d}: "
                f"start={item.start_s:5.1f} s  "
                f"value={item.value:.3f}"
            )
        )

    print()

    print(
        (
            f"Transition guard: +/- "
            f"{args.guard:.3f} s"
        )
    )

    summaries = []

    for raw_path in (
        args.raw_csv
    ):

        samples = read_raw(
            raw_path
        )

        evaluated = evaluate(
            samples,
            truth,
            args.guard,
            args.tolerance,
        )

        output_path = (
            write_evaluation_csv(
                raw_path,
                evaluated,
            )
        )

        metrics = summarize(
            raw_path,
            evaluated,
            truth,
            args.show_errors,
        )

        metrics[
            "path"
        ] = str(
            raw_path
        )

        summaries.append(
            metrics
        )

        print()

        print(
            (
                f"Detailed evaluation CSV: "
                f"{output_path}"
            )
        )

    if len(
        summaries
    ) > 1:

        print()

        print(
            "=" * 74
        )

        print(
            "COMPARISON"
        )

        print(
            "=" * 74
        )

        print(
            (
                "file"
                "                         "
                "coverage   accuracy   recognized-acc   wrong"
            )
        )

        for item in summaries:

            name = Path(
                item[
                    "path"
                ]
            ).name

            print(
                (
                    f"{name:28s} "
                    f"{item['coverage']:8.2%} "
                    f"{item['accuracy']:10.2%} "
                    f"{item['recognized_accuracy']:16.2%} "
                    f"{int(item['wrong']):6d}"
                )
            )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )