from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import dosimeter_postprocess as post


class PostprocessIntervalsTests(unittest.TestCase):
    def write_csv(self, rows: list[dict[str, object]]) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            suffix=".csv",
            delete=False,
        )
        path = Path(tmp.name)
        with tmp:
            writer = csv.DictWriter(
                tmp,
                fieldnames=(
                    "start_s",
                    "end_s",
                    "value",
                    "confidence",
                    "extra",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_reads_interval_or_debug_index_shape(self) -> None:
        path = self.write_csv(
            [
                {
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "value": 3.5,
                    "confidence": 0.8,
                    "extra": "ignored",
                }
            ]
        )

        intervals = post.read_intervals(path)

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].value, 3.5)
        self.assertEqual(intervals[0].duration_s, 2.0)

    def test_weighted_statistics_match_interval_durations(self) -> None:
        intervals = [
            post.Interval(2, 0.0, 1.0, 10.0, 0.9),
            post.Interval(3, 1.0, 4.0, 20.0, 0.9),
        ]

        stats = post.calculate_weighted_stats(intervals)

        self.assertAlmostEqual(stats.mean, 17.5)
        self.assertAlmostEqual(stats.variance, 18.75)
        self.assertAlmostEqual(stats.std_dev, 18.75 ** 0.5)
        self.assertEqual(stats.minimum, 10.0)
        self.assertEqual(stats.maximum, 20.0)

    def test_iterative_weighted_sigma_clip_removes_gross_outliers(self) -> None:
        intervals = [
            post.Interval(2, 0.0, 1.0, 9.9, 0.9),
            post.Interval(3, 1.0, 2.0, 10.0, 0.9),
            post.Interval(4, 2.0, 3.0, 10.1, 0.9),
            post.Interval(5, 3.0, 3.1, 30.0, 0.9),
            post.Interval(6, 3.1, 3.2, 100.0, 0.9),
        ]

        kept, rejected = post.sigma_clip(intervals, 2.0)

        self.assertEqual([x.value for x in kept], [9.9, 10.0, 10.1])
        self.assertEqual(sorted(x.value for x in rejected), [30.0, 100.0])

    def test_confidence_cut_is_inclusive(self) -> None:
        intervals = [
            post.Interval(2, 0.0, 1.0, 10.0, 0.20),
            post.Interval(3, 1.0, 2.0, 20.0, 0.19),
        ]

        kept = [
            interval
            for interval in intervals
            if interval.confidence >= 0.20
        ]

        self.assertEqual([x.value for x in kept], [10.0])


if __name__ == "__main__":
    unittest.main()
