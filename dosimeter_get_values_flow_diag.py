#!/usr/bin/env python3
"""
Diagnostic wrapper for dosimeter_get_values_flow.py.

This script DOES NOT change OCR behaviour.

It captures the exact Sample.darkness arrays immediately before
decode_samples(), reproduces the same temporal filtering, and writes:

    segment_diagnostics.csv

It also prints detailed diagnostics for selected frames.

This avoids relying on monkey-patching temporal_filter(), because the
ROI/fixed-grid layers may hold imported aliases of decoding functions.

Required files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_roi as roi_app
import dosimeter_get_values_fixedgrid as fixed_app
import dosimeter_get_values_flow as flow


# ======================================================================
# Original functions
# ======================================================================

ORIGINAL_CORE_DECODE_SAMPLES = (
    core.decode_samples
)

ORIGINAL_TEMPORAL_FILTER = (
    core.temporal_filter
)


# ======================================================================
# Capture
# ======================================================================

CAPTURES: list[
    dict[str, object]
] = []


# ======================================================================
# Reconstruct segment arrays directly from Sample objects
# ======================================================================


def reconstruct_darkness(
    samples,
    profile,
) -> np.ndarray:

    darkness = np.full(
        (
            len(samples),
            len(profile.digit_boxes),
            7,
        ),
        np.nan,
        dtype=float,
    )

    for index, sample in enumerate(
        samples
    ):

        if (
            sample.darkness
            is not None
        ):

            darkness[
                index
            ] = np.asarray(
                sample.darkness,
                dtype=float,
            )

    return darkness


# ======================================================================
# decode_samples hook
# ======================================================================


def capturing_decode_samples(
    samples,
    profile,
    filter_window,
    decimal_places_override=None,
    minimum_confidence=0.0,
    decimal_switch_penalty=4.0,
    decimal_sequence_observer: (
        core.DecimalSequenceObserver | None
    ) = None,
    rds200_pattern_refinement: bool = False,
):

    raw = (
        reconstruct_darkness(
            samples,
            profile,
        )
    )

    filtered = (
        ORIGINAL_TEMPORAL_FILTER(
            raw,
            filter_window,
            profile.temporal_filter,
        )
    )

    CAPTURES.append(
        {
            "raw": raw.copy(),
            "filtered": filtered.copy(),
            "filter_window": int(
                filter_window
            ),
            "filter_method": str(
                profile.temporal_filter
            ),
            "profile": str(
                profile.name
            ),
        }
    )

    # Run the ORIGINAL decoder unchanged.
    return ORIGINAL_CORE_DECODE_SAMPLES(
        samples,
        profile,
        filter_window,
        decimal_places_override=(
            decimal_places_override
        ),
        minimum_confidence=(
            minimum_confidence
        ),
        decimal_switch_penalty=(
            decimal_switch_penalty
        ),
        decimal_sequence_observer=(
            decimal_sequence_observer
        ),
        rds200_pattern_refinement=(
            rds200_pattern_refinement
        ),
    )


# ======================================================================
# Command-line helpers
# ======================================================================


def get_argument(
    name: str,
    default=None,
):

    arguments = (
        sys.argv[1:]
    )

    for index, item in enumerate(
        arguments
    ):

        if (
            item == name
            and index + 1
            < len(arguments)
        ):

            return arguments[
                index + 1
            ]

        prefix = (
            name + "="
        )

        if item.startswith(
            prefix
        ):

            return item[
                len(prefix):
            ]

    return default


# ======================================================================
# Margin candidate -- DIAGNOSTIC ONLY
# ======================================================================


def margin_candidate(
    levels: np.ndarray,
    patterns,
) -> tuple[
    int,
    float,
]:

    best_digit = -1

    best_margin = float(
        "-inf"
    )

    for digit, pattern in (
        patterns.items()
    ):

        active_mask = (
            np.asarray(
                pattern,
                dtype=bool,
            )
        )

        active = (
            levels[
                active_mask
            ]
        )

        inactive = (
            levels[
                ~active_mask
            ]
        )

        if (
            active.size == 0
        ):

            continue

        weakest_active = float(
            np.min(
                active
            )
        )

        if (
            inactive.size > 0
        ):

            strongest_inactive = float(
                np.max(
                    inactive
                )
            )

            margin = (
                weakest_active
                - strongest_inactive
            )

        else:

            # Digit 8: no inactive segments.
            margin = (
                weakest_active
                - 12.0
            )

        if (
            margin
            > best_margin
        ):

            best_margin = (
                margin
            )

            best_digit = int(
                digit
            )

    return (
        best_digit,
        best_margin,
    )


# ======================================================================
# Existing decoder diagnostics
# ======================================================================


def decode_one_digit(
    levels: np.ndarray,
    patterns,
) -> tuple[
    int | None,
    float,
]:

    if not np.all(
        np.isfinite(
            levels
        )
    ):

        return (
            None,
            0.0,
        )

    return (
        core.decode_pattern_digit(
            levels,
            patterns,
        )
    )


def score_information(
    levels: np.ndarray,
    patterns,
) -> tuple[
    int,
    float,
    int,
    float,
]:

    if not np.all(
        np.isfinite(
            levels
        )
    ):

        return (
            -1,
            float("nan"),
            -1,
            float("nan"),
        )

    scores = (
        core.digit_scores(
            levels,
            patterns,
        )
    )

    if (
        len(scores) < 2
    ):

        return (
            -1,
            float("nan"),
            -1,
            float("nan"),
        )

    best = (
        scores[0]
    )

    second = (
        scores[1]
    )

    return (
        int(
            best[1]
        ),
        float(
            best[0]
        ),
        int(
            second[1]
        ),
        float(
            second[0]
        ),
    )


# ======================================================================
# CSV
# ======================================================================


def write_diagnostics(
    raw: np.ndarray,
    filtered: np.ndarray,
    sample_fps: float,
    filename: str,
) -> None:

    patterns = (
        core.STANDARD_DIGIT_PATTERNS
    )

    segment_names = (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = (
            csv.writer(
                handle
            )
        )

        header = [
            "sample_index",
            "time_s",
            "digit_position",
        ]

        header += [
            f"raw_{name}"
            for name
            in segment_names
        ]

        header += [
            f"filtered_{name}"
            for name
            in segment_names
        ]

        header += [
            "raw_decoder_digit",
            "raw_decoder_confidence",
            "filtered_decoder_digit",
            "filtered_decoder_confidence",
            "score_best_digit",
            "score_best",
            "score_second_digit",
            "score_second",
            "margin_best_digit",
            "margin_best",
        ]

        writer.writerow(
            header
        )

        for sample_index in range(
            filtered.shape[0]
        ):

            time_s = (
                sample_index
                / sample_fps
            )

            for digit_position in range(
                filtered.shape[1]
            ):

                raw_levels = np.asarray(
                    raw[
                        sample_index,
                        digit_position,
                    ],
                    dtype=float,
                )

                filtered_levels = np.asarray(
                    filtered[
                        sample_index,
                        digit_position,
                    ],
                    dtype=float,
                )

                (
                    raw_digit,
                    raw_confidence,
                ) = decode_one_digit(
                    raw_levels,
                    patterns,
                )

                (
                    filtered_digit,
                    filtered_confidence,
                ) = decode_one_digit(
                    filtered_levels,
                    patterns,
                )

                (
                    score_best_digit,
                    score_best,
                    score_second_digit,
                    score_second,
                ) = score_information(
                    filtered_levels,
                    patterns,
                )

                if np.all(
                    np.isfinite(
                        filtered_levels
                    )
                ):

                    (
                        margin_digit,
                        margin,
                    ) = margin_candidate(
                        filtered_levels,
                        patterns,
                    )

                else:

                    margin_digit = -1

                    margin = float(
                        "nan"
                    )

                row = [
                    sample_index,
                    f"{time_s:.3f}",
                    digit_position + 1,
                ]

                row += [
                    (
                        ""
                        if not np.isfinite(
                            value
                        )
                        else f"{value:.4f}"
                    )
                    for value in raw_levels
                ]

                row += [
                    (
                        ""
                        if not np.isfinite(
                            value
                        )
                        else f"{value:.4f}"
                    )
                    for value
                    in filtered_levels
                ]

                row += [
                    (
                        ""
                        if raw_digit
                        is None
                        else raw_digit
                    ),
                    f"{raw_confidence:.5f}",
                    (
                        ""
                        if filtered_digit
                        is None
                        else filtered_digit
                    ),
                    f"{filtered_confidence:.5f}",
                    score_best_digit,
                    f"{score_best:.6f}",
                    score_second_digit,
                    f"{score_second:.6f}",
                    margin_digit,
                    f"{margin:.4f}",
                ]

                writer.writerow(
                    row
                )


# ======================================================================
# Pretty terminal diagnostics
# ======================================================================


def format_segments(
    levels: np.ndarray,
) -> str:

    names = (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    )

    parts = []

    for name, value in zip(
        names,
        levels,
    ):

        if np.isfinite(
            value
        ):

            parts.append(
                f"{name}={value:7.2f}"
            )

        else:

            parts.append(
                f"{name}=    NaN"
            )

    return " ".join(
        parts
    )


def print_selected_times(
    raw: np.ndarray,
    filtered: np.ndarray,
    sample_fps: float,
) -> None:

    patterns = (
        core.STANDARD_DIGIT_PATTERNS
    )

    # Match the useful debug images plus the exact reference.
    interesting_times = (
        0.6,
        2.0,
        5.4,
        6.8,
        8.8,
        9.8,
        10.8,
    )

    print()

    print(
        "======================================================================"
    )

    print(
        "SELECTED SEGMENT DIAGNOSTICS"
    )

    print(
        "segment order: a=top, b=upper-right, c=lower-right,"
    )

    print(
        "               d=bottom, e=lower-left, f=upper-left, g=middle"
    )

    print(
        "======================================================================"
    )

    for requested_time in (
        interesting_times
    ):

        sample_index = int(
            round(
                requested_time
                * sample_fps
            )
        )

        if not (
            0
            <= sample_index
            < filtered.shape[0]
        ):

            continue

        actual_time = (
            sample_index
            / sample_fps
        )

        print()

        print(
            "----------------------------------------------------------------------"
        )

        print(
            (
                f"t = {actual_time:.3f} s "
                f"(sample {sample_index})"
            )
        )

        for digit_position in range(
            filtered.shape[1]
        ):

            raw_levels = np.asarray(
                raw[
                    sample_index,
                    digit_position,
                ],
                dtype=float,
            )

            filtered_levels = np.asarray(
                filtered[
                    sample_index,
                    digit_position,
                ],
                dtype=float,
            )

            (
                raw_digit,
                raw_confidence,
            ) = decode_one_digit(
                raw_levels,
                patterns,
            )

            (
                filtered_digit,
                filtered_confidence,
            ) = decode_one_digit(
                filtered_levels,
                patterns,
            )

            if np.all(
                np.isfinite(
                    filtered_levels
                )
            ):

                (
                    margin_digit,
                    margin,
                ) = margin_candidate(
                    filtered_levels,
                    patterns,
                )

            else:

                margin_digit = -1

                margin = float(
                    "nan"
                )

            print()

            print(
                (
                    f"  digit {digit_position + 1}: "
                    f"RAW decoder={raw_digit}, "
                    f"conf={raw_confidence:.3f} | "
                    f"FILTERED decoder={filtered_digit}, "
                    f"conf={filtered_confidence:.3f} | "
                    f"margin_best={margin_digit}, "
                    f"margin={margin:.2f}"
                )
            )

            print(
                "    RAW     : "
                + format_segments(
                    raw_levels
                )
            )

            print(
                "    FILTERED: "
                + format_segments(
                    filtered_levels
                )
            )

    print()

    print(
        "======================================================================"
    )


# ======================================================================
# Patch helpers
# ======================================================================


def install_decode_hook():

    patched = []

    # Patch every module-level alias which may actually be used
    # by the layered ROI/fixedgrid/flow pipeline.
    modules = (
        core,
        roi_app,
        fixed_app,
        flow,
    )

    for module in modules:

        if hasattr(
            module,
            "decode_samples",
        ):

            old_function = getattr(
                module,
                "decode_samples",
            )

            setattr(
                module,
                "decode_samples",
                capturing_decode_samples,
            )

            patched.append(
                (
                    module,
                    old_function,
                )
            )

    return patched


def restore_decode_hooks(
    patched,
) -> None:

    for module, old_function in (
        patched
    ):

        setattr(
            module,
            "decode_samples",
            old_function,
        )


# ======================================================================
# Main
# ======================================================================


def main(
    profile_override: core.Profile | None = None,
) -> int:

    CAPTURES.clear()

    profile_name = (
        profile_override.name
        if profile_override is not None
        else get_argument(
            "--profile",
            "rds200",
        )
    )

    sample_fps_text = (
        get_argument(
            "--sample-fps",
            None,
        )
    )

    if (
        sample_fps_text
        is None
    ):

        sample_fps = float(
            (
                core.PROFILES[
                    profile_name
                ]
                if profile_override is None
                else profile_override
            ).default_sample_fps
        )

    else:

        sample_fps = float(
            sample_fps_text
        )

    patched = (
        install_decode_hook()
    )

    try:

        result = (
            flow.main(
                profile_override=profile_override
            )
        )

    finally:

        restore_decode_hooks(
            patched
        )

    print()

    print(
        "Decode hook diagnostics:"
    )

    print(
        (
            f"  patched aliases : "
            f"{len(patched)}"
        )
    )

    print(
        (
            f"  captured calls  : "
            f"{len(CAPTURES)}"
        )
    )

    if not CAPTURES:

        print(
            (
                "ERROR: decode_samples() was not captured. "
                "No OCR data were modified."
            ),
            file=sys.stderr,
        )

        return (
            result
            if result != 0
            else 1
        )

    # RDS200 should normally make one relevant decode_samples call.
    # Prefer the capture with the largest amount of segment data.
    capture = max(
        CAPTURES,
        key=lambda item: (
            np.asarray(
                item[
                    "filtered"
                ]
            ).size
        ),
    )

    raw = np.asarray(
        capture[
            "raw"
        ],
        dtype=float,
    )

    filtered = np.asarray(
        capture[
            "filtered"
        ],
        dtype=float,
    )

    filter_window = int(
        capture[
            "filter_window"
        ]
    )

    filter_method = str(
        capture[
            "filter_method"
        ]
    )

    diagnostic_file = (
        "segment_diagnostics.csv"
    )

    write_diagnostics(
        raw,
        filtered,
        sample_fps,
        diagnostic_file,
    )

    print()

    print(
        "Segment diagnostic capture:"
    )

    print(
        (
            f"  profile     : "
            f"{capture['profile']}"
        )
    )

    print(
        (
            f"  shape       : "
            f"{raw.shape}"
        )
    )

    print(
        (
            f"  filter      : "
            f"{filter_method}"
        )
    )

    print(
        (
            f"  window      : "
            f"{filter_window}"
        )
    )

    print(
        (
            f"  sample fps  : "
            f"{sample_fps:g}"
        )
    )

    print(
        (
            f"  CSV         : "
            f"{diagnostic_file}"
        )
    )

    print_selected_times(
        raw,
        filtered,
        sample_fps,
    )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
