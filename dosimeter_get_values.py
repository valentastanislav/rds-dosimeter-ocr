#!/usr/bin/env python3
"""Extract changing seven-segment values from RADOS dosimeter videos.

Supported profiles:
    rds200  - RADOS RDS-200, including automatic moving decimal point
              with adaptive segment alignment, stable decimal tracking, and robust digit discrimination
    rds30   - RADOS RDS-30 with blank-aware maximum-margin segment decoding

Examples:
    python3 dosimeter_get_values.py IMG_0742.MOV out.csv --profile rds200
    python3 dosimeter_get_values.py IMG_0754.MOV out.csv --profile rds30

Dependencies:
    python3 -m pip install numpy opencv-python
    ffmpeg and ffprobe must be available on PATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import cv2
import numpy as np


SEGMENT_ORDER = ("a", "b", "c", "d", "e", "f", "g")

STANDARD_DIGIT_PATTERNS: dict[int, tuple[int, ...]] = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}

# The RDS-30 LCD uses two non-standard glyphs:
# 7 includes the upper-left segment, and 9 omits the bottom segment.
RDS30_DIGIT_PATTERNS = dict(STANDARD_DIGIT_PATTERNS)
RDS30_DIGIT_PATTERNS[7] = (1, 1, 1, 0, 0, 1, 0)
RDS30_DIGIT_PATTERNS[9] = (1, 1, 1, 0, 0, 1, 1)


@dataclass(frozen=True)
class FixedGridMeasurementConfig:
    mode: str
    segment_percentile: float
    background_percentile: float


FixedGridGeometryMode = Literal[
    "manual_grid",
    "profile",
]


@dataclass(frozen=True)
class Profile:
    name: str
    canonical_width: int
    canonical_height: int
    digit_boxes: tuple[tuple[int, int, int, int], ...]
    segment_polygons: dict[str, np.ndarray]
    decimal_places: int
    display_aspect_min: float
    display_aspect_max: float
    flow_feature_exclusion_box: tuple[float, float, float, float]
    fixedgrid_primary_measurement: FixedGridMeasurementConfig
    fixedgrid_geometry_mode: FixedGridGeometryMode
    # Candidate decimal-point boxes: (x1, y1, x2, y2, decimal_places).
    # If empty, decimal_places above is used as a fixed value.
    decimal_candidates: tuple[tuple[int, int, int, int, int], ...] = ()
    display_threshold: int = 80
    default_sample_fps: float = 5.0
    default_filter_window: int = 5
    temporal_filter: str = "median"  # median or max
    digit_patterns: dict[int, tuple[int, ...]] | None = None
    decoder: str = "pattern"  # pattern, prototype, or hybrid
    prototypes: np.ndarray | None = None  # shape (positions, 10, 7)
    default_mode_window: int = 1
    preserve_edge_runs: bool = False
    choose_run_value_at_center: bool = False
    adaptive_y_shift: bool = False
    y_shift_min: int = 0
    y_shift_max: int = 0
    default_min_confidence: float = 0.0
    aux_segment_percentile: float | None = None
    hybrid_pattern_margin: float = 0.10
    leading_blank_threshold: float | None = None


DisplayFinder = Callable[
    [
        np.ndarray,
        Profile,
        tuple[int, int, int, int] | None,
    ],
    tuple[
        np.ndarray | None,
        tuple[int, int, int, int] | None,
    ],
]
DecimalSequenceObserver = Callable[
    [Sequence[int]],
    None,
]


# RDS-200: coordinates in a canonical 493 x 356 crop of the black bezel.
# Each digit patch is normalized to 65 x 130 pixels.
RDS200_SEGMENTS = {
    "a": np.array([[23, 28], [47, 28], [44, 36], [21, 36]], np.int32),
    "b": np.array([[45, 33], [53, 35], [50, 60], [42, 59]], np.int32),
    "c": np.array([[40, 73], [48, 71], [45, 100], [37, 104]], np.int32),
    "d": np.array([[13, 99], [40, 99], [37, 107], [11, 107]], np.int32),
    "e": np.array([[10, 73], [18, 70], [15, 99], [7, 103]], np.int32),
    "f": np.array([[13, 33], [21, 31], [18, 59], [10, 63]], np.int32),
    "g": np.array([[20, 63], [43, 62], [40, 71], [18, 72]], np.int32),
}

RDS200 = Profile(
    name="rds200",
    canonical_width=493,
    canonical_height=356,
    digit_boxes=(
        (139, 149, 204, 279),
        (202, 149, 267, 279),
        (265, 149, 330, 279),
    ),
    segment_polygons=RDS200_SEGMENTS,
    decimal_places=1,
    display_aspect_min=1.20,
    display_aspect_max=1.60,
    flow_feature_exclusion_box=(0.23, 0.40, 0.78, 0.86),
    fixedgrid_primary_measurement=FixedGridMeasurementConfig(
        mode="local",
        segment_percentile=35.0,
        background_percentile=70.0,
    ),
    fixedgrid_geometry_mode="manual_grid",
    # The RDS-200 moves the decimal point with the measurement range:
    # x.xxx is rendered as X.XX (2 decimal places), while xx.x is XX.X.
    decimal_candidates=(
        (191, 241, 201, 256, 2),  # decimal point after the first digit
        (255, 241, 264, 256, 1),  # decimal point after the second digit
    ),
    default_sample_fps=5.0,
    default_filter_window=5,
    temporal_filter="median",
    digit_patterns=STANDARD_DIGIT_PATTERNS,
    decoder="pattern",
    adaptive_y_shift=True,
    y_shift_min=-22,
    y_shift_max=10,
    default_min_confidence=0.20,
)

# RDS-30: the digit geometry is wider and slightly more trapezoidal.
RDS30_SEGMENTS = {
    "a": np.array([[27, 27], [51, 27], [48, 35], [23, 35]], np.int32),
    "b": np.array([[56, 38], [64, 40], [61, 60], [53, 60]], np.int32),
    "c": np.array([[53, 76], [61, 74], [58, 101], [50, 104]], np.int32),
    "d": np.array([[21, 106], [47, 106], [44, 114], [18, 114]], np.int32),
    "e": np.array([[12, 77], [20, 73], [17, 101], [9, 104]], np.int32),
    "f": np.array([[17, 38], [24, 35], [21, 60], [14, 63]], np.int32),
    "g": np.array([[24, 64], [51, 63], [46, 72], [20, 73]], np.int32),
}

# Position-specific seven-segment darkness prototypes for the RDS-30 profile.
# They are used after a five-frame temporal maximum and make the decoder robust
# against the multiplexed LCD and unequal segment brightness.
RDS30_PROTOTYPES = np.asarray(
    [
        [
            [34.0, 25.0, 19.0, 36.0, 11.5, 33.0, 1.0],
            [12.0, 50.0, 36.0, 0.0, 2.0, 12.0, 5.0],
            [48.0, 45.5, 1.0, 48.0, 47.0, 2.0, 42.0],
            [48.0, 50.0, 38.0, 48.0, 1.0, 2.0, 41.0],
            [12.0, 47.0, 36.0, 1.0, 1.0, 36.0, 30.0],
            [41.0, 4.0, 35.0, 35.0, 3.0, 34.0, 31.0],
            [42.0, 6.0, 34.0, 34.0, 8.25, 33.0, 29.0],
            [53.0, 52.0, 35.0, 3.0, 6.0, 47.0, 5.0],
            [47.0, 47.0, 35.0, 47.0, 37.0, 47.0, 36.0],
            [49.0, 51.0, 36.0, 1.0, 2.0, 48.0, 35.0],
        ],
        [
            [44.0, 45.0, 37.0, 49.0, 30.0, 43.0, 2.0],
            [12.0, 50.0, 36.0, 0.0, 2.0, 12.0, 5.0],
            [48.0, 46.0, 1.0, 48.0, 47.0, 2.0, 40.5],
            [50.0, 50.0, 36.0, 47.0, 1.0, 2.0, 40.0],
            [12.0, 47.0, 36.0, 1.0, 1.0, 36.0, 30.0],
            [41.0, 4.0, 35.0, 35.0, 3.0, 34.0, 31.0],
            [42.0, 6.0, 34.0, 34.0, 8.25, 33.0, 29.0],
            [52.0, 52.0, 33.0, 4.0, 10.0, 53.0, 5.0],
            [47.0, 46.0, 35.0, 48.0, 38.0, 47.0, 36.0],
            [49.0, 47.0, 35.0, 2.0, 4.0, 51.0, 36.0],
        ],
        [
            [48.5, 52.0, 34.0, 34.0, 7.0, 27.0, 9.0],
            [12.0, 50.0, 36.0, 0.0, 2.0, 12.0, 5.0],
            [49.0, 48.0, 0.0, 37.0, 14.1, 12.0, 21.0],
            [54.0, 46.0, 35.0, 36.0, 1.0, 11.0, 29.0],
            [12.0, 47.0, 36.0, 1.0, 1.0, 36.0, 30.0],
            [41.0, 4.0, 35.0, 35.0, 3.0, 34.0, 31.0],
            [42.0, 6.0, 34.0, 34.0, 8.25, 33.0, 29.0],
            [53.0, 51.0, 36.0, 1.0, 1.0, 35.0, 3.0],
            [42.0, 51.0, 36.0, 34.0, 11.0, 30.0, 30.0],
            [50.0, 53.5, 36.0, 0.0, 1.0, 37.05, 33.0],
        ],
        [
            [33.0, 23.5, 18.0, 36.0, 11.0, 32.0, 1.0],
            [12.0, 50.0, 36.0, 0.0, 2.0, 12.0, 5.0],
            [48.0, 46.0, 1.0, 48.0, 47.0, 2.0, 40.5],
            [50.0, 50.0, 36.0, 47.0, 1.0, 2.0, 40.0],
            [12.0, 47.0, 36.0, 1.0, 1.0, 36.0, 30.0],
            [41.0, 4.0, 35.0, 35.0, 3.0, 34.0, 31.0],
            [42.0, 6.0, 34.0, 34.0, 8.25, 33.0, 29.0],
            [53.0, 52.0, 35.0, 3.0, 6.0, 47.0, 5.0],
            [47.0, 47.0, 35.0, 47.0, 37.0, 47.0, 36.0],
            [49.0, 51.0, 36.0, 1.0, 2.0, 48.0, 35.0],
        ],
    ],
    dtype=float,
)

RDS30 = Profile(
    name="rds30",
    canonical_width=493,
    canonical_height=356,
    digit_boxes=(
        (5, 6, 116, 350),
        (121, 6, 232, 350),
        (237, 6, 354, 350),
        (355, 6, 488, 350),
    ),
    segment_polygons=RDS30_SEGMENTS,
    decimal_places=2,
    display_aspect_min=1.20,
    display_aspect_max=1.60,
    flow_feature_exclusion_box=(0.33, 0.34, 0.98, 0.80),
    fixedgrid_primary_measurement=FixedGridMeasurementConfig(
        mode="core",
        segment_percentile=50.0,
        background_percentile=90.0,
    ),
    fixedgrid_geometry_mode="profile",
    default_sample_fps=30.0,
    default_filter_window=5,
    temporal_filter="median",
    digit_patterns=RDS30_DIGIT_PATTERNS,
    decoder="rds30_margin",
    prototypes=RDS30_PROTOTYPES,
    aux_segment_percentile=30.0,
    hybrid_pattern_margin=0.10,
    default_mode_window=21,
    preserve_edge_runs=True,
    choose_run_value_at_center=True,
    adaptive_y_shift=True,
    y_shift_min=-5,
    y_shift_max=18,
    # The leftmost field is intentionally blank below 10 uSv/h.  It must be
    # treated as a valid leading blank, not forced to the nearest digit (usually 1).
    leading_blank_threshold=12.0,
    # Confidence is diagnostic for this profile.  Temporal voting removes isolated
    # weak frames; rejecting them first discarded too much valid data.
    default_min_confidence=0.20,
)

PROFILES = {profile.name: profile for profile in (RDS200, RDS30)}


@dataclass
class VideoInfo:
    width: int
    height: int
    rotation: int
    duration: float

    @property
    def oriented_size(self) -> tuple[int, int]:
        if abs(self.rotation) % 180 == 90:
            return self.height, self.width
        return self.width, self.height


@dataclass
class Sample:
    time_s: float
    darkness: np.ndarray | None
    display_found: bool
    aux_darkness: np.ndarray | None = None
    decimal_scores: np.ndarray | None = None
    alignment_shift: int = 0


@dataclass
class DecodedSample:
    time_s: float
    value: float | None
    confidence: float


@dataclass
class Run:
    start_index: int
    end_index: int
    value: float
    confidence: float

    @property
    def sample_count(self) -> int:
        return self.end_index - self.start_index


def run_command(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Program {command[0]!r} was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{message}") from exc
    return result.stdout


def probe_video(path: Path) -> VideoInfo:
    data = json.loads(
        run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:stream_tags=rotate:stream_side_data=rotation:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream was found in the input file.")

    stream = streams[0]
    rotation = 0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
            break
    else:
        rotation = int(round(float(stream.get("tags", {}).get("rotate", 0))))

    duration = float(data.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise RuntimeError("Could not determine the video duration.")

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        rotation=rotation,
        duration=duration,
    )


def even_number(value: float) -> int:
    result = max(2, int(round(value)))
    return result if result % 2 == 0 else result + 1


def iter_ffmpeg_frames(
    video: Path,
    info: VideoInfo,
    sample_fps: float,
    processing_width: int,
) -> Iterable[np.ndarray]:
    oriented_width, oriented_height = info.oriented_size
    processing_height = even_number(processing_width * oriented_height / oriented_width)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-an",
        "-vf",
        f"fps={sample_fps:g},scale={processing_width}:{processing_height}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=64 * 1024 * 1024,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH.") from exc

    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Could not open the ffmpeg output pipe.")

    frame_size = processing_width * processing_height * 3
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg returned an incomplete video frame.")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(
                processing_height, processing_width, 3
            )
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}:\n{stderr}")

def normalize_display_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    profile: Profile,
) -> np.ndarray | None:
    """Crop a display box safely and normalize it to profile coordinates."""

    frame_height, frame_width = gray.shape

    x, y, width, height = box

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(frame_width, int(round(x + width)))
    y2 = min(frame_height, int(round(y + height)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (profile.canonical_width, profile.canonical_height),
        interpolation=cv2.INTER_CUBIC,
    )


def display_crop_quality(
    display: np.ndarray,
    profile: Profile,
) -> float:
    """
    Estimate how much a normalized crop looks like the dosimeter display.

    We use two largely illumination-independent properties:

      1. the black bezel should be darker than the central LCD,
      2. the digit area should contain visible intensity structure.

    Higher score is better.
    """

    height, width = display.shape

    if height < 20 or width < 20:
        return float("-inf")

    # --------------------------------------------------------
    # Black bezel versus central area
    # --------------------------------------------------------

    border_x = max(2, int(round(0.08 * width)))
    border_y = max(2, int(round(0.08 * height)))

    border_pixels = np.concatenate(
        [
            display[:border_y, :].ravel(),
            display[-border_y:, :].ravel(),
            display[:, :border_x].ravel(),
            display[:, -border_x:].ravel(),
        ]
    )

    center = display[
        int(round(0.15 * height)) : int(round(0.85 * height)),
        int(round(0.15 * width)) : int(round(0.85 * width)),
    ]

    if center.size == 0:
        return float("-inf")

    border_level = float(np.median(border_pixels))
    center_level = float(np.median(center))

    # Positive if bezel is darker than the center.
    bezel_contrast = center_level - border_level

    # --------------------------------------------------------
    # Structure in the expected digit region
    # --------------------------------------------------------

    x1 = min(box[0] for box in profile.digit_boxes)
    y1 = min(box[1] for box in profile.digit_boxes)
    x2 = max(box[2] for box in profile.digit_boxes)
    y2 = max(box[3] for box in profile.digit_boxes)

    digit_region = display[y1:y2, x1:x2]

    if digit_region.size == 0:
        return float("-inf")

    p10 = float(np.percentile(digit_region, 10))
    p90 = float(np.percentile(digit_region, 90))

    digit_structure = p90 - p10

    # Both contributions are useful, but digit structure is given
    # slightly more weight because reflections can alter bezel brightness.
    quality = (
        0.7 * max(bezel_contrast, 0.0)
        + 1.0 * digit_structure
    )

    return float(quality)


def find_display_crop(
    frame: np.ndarray,
    profile: Profile,
    previous_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """
    Find and track the dosimeter display.

    Strategy:

      1. Try global contour detection using several darkness thresholds.
      2. If we already know the previous display position, simultaneously
         search small translations/scalings around that position.
      3. If global contour detection temporarily fails because of glare or
         illumination, use the locally tracked position.

    This is much more robust for videos where the geometry is almost fixed
    but illumination changes between frames.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_height, frame_width = gray.shape
    frame_area = float(frame_width * frame_height)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # ========================================================
    # 1. LOCAL TRACKING AROUND PREVIOUS POSITION
    # ========================================================

    best_tracking_quality = float("-inf")
    best_tracking_box: tuple[int, int, int, int] | None = None
    best_tracking_display: np.ndarray | None = None

    if previous_box is not None:

        px, py, pw, ph = previous_box

        # Small scale changes cope with minor camera movement.
        scales = (
            0.94,
            0.97,
            1.00,
            1.03,
            1.06,
        )

        # Search around the last location.
        x_offsets = (
            -0.06,
            -0.03,
            0.00,
            0.03,
            0.06,
        )

        y_offsets = (
            -0.06,
            -0.03,
            0.00,
            0.03,
            0.06,
        )

        previous_center_x = px + pw / 2.0
        previous_center_y = py + ph / 2.0

        for scale in scales:

            candidate_width = pw * scale
            candidate_height = ph * scale

            for dx_fraction in x_offsets:

                for dy_fraction in y_offsets:

                    center_x = (
                        previous_center_x
                        + dx_fraction * pw
                    )

                    center_y = (
                        previous_center_y
                        + dy_fraction * ph
                    )

                    x = center_x - candidate_width / 2.0
                    y = center_y - candidate_height / 2.0

                    candidate_box = (
                        int(round(x)),
                        int(round(y)),
                        int(round(candidate_width)),
                        int(round(candidate_height)),
                    )

                    display = normalize_display_box(
                        gray,
                        candidate_box,
                        profile,
                    )

                    if display is None:
                        continue

                    quality = display_crop_quality(
                        display,
                        profile,
                    )

                    # Small penalty for moving away from the last box.
                    movement_penalty = (
                        8.0 * abs(dx_fraction)
                        + 8.0 * abs(dy_fraction)
                        + 10.0 * abs(scale - 1.0)
                    )

                    quality -= movement_penalty

                    if quality > best_tracking_quality:

                        best_tracking_quality = quality
                        best_tracking_box = candidate_box
                        best_tracking_display = display

    # ========================================================
    # 2. GLOBAL CONTOUR DETECTION
    # ========================================================

    otsu_threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    threshold_candidates = {
        int(profile.display_threshold),
        45,
        55,
        65,
        75,
        85,
        95,
        105,
        115,
        125,
        135,
        int(round(float(otsu_threshold))),
    }

    threshold_candidates = sorted(
        value
        for value in threshold_candidates
        if 30 <= value <= 160
    )

    global_candidates: list[
        tuple[
            float,
            tuple[int, int, int, int],
            np.ndarray,
        ]
    ] = []

    preferred_aspect = 1.40

    for threshold in threshold_candidates:

        binary = (
            blurred < threshold
        ).astype(np.uint8) * 255

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones(
                (5, 5),
                dtype=np.uint8,
            ),
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            x, y, width, height = cv2.boundingRect(
                contour
            )

            if width <= 0 or height <= 0:
                continue

            bounding_area = float(
                width * height
            )

            contour_area = float(
                cv2.contourArea(contour)
            )

            if bounding_area <= 0:
                continue

            aspect = width / height

            area_fraction = (
                bounding_area
                / frame_area
            )

            fill = (
                contour_area
                / bounding_area
            )

            # ------------------------------------------------
            # Broad geometric cuts
            # ------------------------------------------------

            if not (
                0.95 <= aspect <= 2.00
            ):
                continue

            if area_fraction < 0.001:
                continue

            if area_fraction > 0.50:
                continue

            if y > 0.95 * frame_height:
                continue

            if fill < 0.20:
                continue

            box = (
                x,
                y,
                width,
                height,
            )

            display = normalize_display_box(
                gray,
                box,
                profile,
            )

            if display is None:
                continue

            crop_quality = display_crop_quality(
                display,
                profile,
            )

            # Reject completely featureless candidates.
            if crop_quality < 8.0:
                continue

            # ------------------------------------------------
            # Shape score
            # ------------------------------------------------

            aspect_error = abs(
                math.log(
                    max(aspect, 1e-6)
                    / preferred_aspect
                )
            )

            shape_score = math.exp(
                -1.2 * aspect_error
            )

            # ------------------------------------------------
            # Mild size score
            # ------------------------------------------------

            size_score = math.sqrt(
                max(area_fraction, 1e-9)
            )

            # ------------------------------------------------
            # Temporal preference
            # ------------------------------------------------

            tracking_score = 1.0

            if previous_box is not None:

                px, py, pw, ph = previous_box

                previous_center_x = (
                    px + pw / 2.0
                )

                previous_center_y = (
                    py + ph / 2.0
                )

                center_x = (
                    x + width / 2.0
                )

                center_y = (
                    y + height / 2.0
                )

                distance = math.hypot(
                    center_x - previous_center_x,
                    center_y - previous_center_y,
                )

                distance /= max(
                    0.30 * frame_width,
                    1.0,
                )

                scale_difference = abs(
                    math.log(
                        max(width, 1)
                        / max(pw, 1)
                    )
                )

                tracking_score = (
                    math.exp(-distance)
                    * math.exp(
                        -1.2 * scale_difference
                    )
                )

            # ------------------------------------------------
            # Final global score
            # ------------------------------------------------

            score = (
                crop_quality
                * shape_score
                * (0.5 + size_score)
                * (0.5 + 0.5 * tracking_score)
            )

            global_candidates.append(
                (
                    float(score),
                    box,
                    display,
                )
            )

    # ========================================================
    # 3. CHOOSE GLOBAL DETECTION OR LOCAL TRACKER
    # ========================================================

    if global_candidates:

        global_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _score, global_box, global_display = (
            global_candidates[0]
        )

        # If we do not yet have tracking information, global
        # detection is our acquisition step.
        if best_tracking_display is None:

            return (
                global_display,
                global_box,
            )

        global_quality = display_crop_quality(
            global_display,
            profile,
        )

        # Prefer the global detector if it is clearly better.
        if (
            global_quality
            > best_tracking_quality + 8.0
        ):

            return (
                global_display,
                global_box,
            )

    # ========================================================
    # 4. FALL BACK TO TRACKING
    # ========================================================

    if (
        best_tracking_display is not None
        and best_tracking_box is not None
        and best_tracking_quality >= 8.0
    ):

        return (
            best_tracking_display,
            best_tracking_box,
        )

    # No reliable display yet.
    return None, previous_box

def make_segment_masks(profile: Profile) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for name in SEGMENT_ORDER:
        mask = np.zeros((130, 65), dtype=np.uint8)
        cv2.fillPoly(mask, [profile.segment_polygons[name]], 255)
        masks.append(mask.astype(bool))
    return tuple(masks)


def extract_darkness_from_patches(
    patches: Sequence[np.ndarray],
    segment_masks: Sequence[np.ndarray],
    segment_percentile: float = 50.0,
    background_percentile: float = 90.0,
) -> np.ndarray:
    result = np.empty((len(patches), 7), dtype=float)
    for digit_index, patch in enumerate(patches):
        background = float(
            np.percentile(
                patch,
                background_percentile,
            )
        )
        for segment_index, mask in enumerate(segment_masks):
            segment_level = float(
                np.percentile(patch[mask], segment_percentile)
            )
            result[digit_index, segment_index] = background - segment_level
    return result


def digit_patches(display: np.ndarray, profile: Profile) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for x1, y1, x2, y2 in profile.digit_boxes:
        patch = display[y1:y2, x1:x2]
        if patch.shape != (130, 65):
            patch = cv2.resize(patch, (65, 130), interpolation=cv2.INTER_LINEAR)
        patches.append(patch)
    return patches


def extract_darkness(
    display: np.ndarray,
    profile: Profile,
    segment_masks: Sequence[np.ndarray],
    segment_percentile: float = 50.0,
    background_percentile: float = 90.0,
) -> np.ndarray:
    return extract_darkness_from_patches(
        digit_patches(display, profile),
        segment_masks,
        segment_percentile=segment_percentile,
        background_percentile=background_percentile,
    )


def make_shifted_segment_masks(profile: Profile, y_shift: int) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for name in SEGMENT_ORDER:
        points = profile.segment_polygons[name].copy()
        points[:, 1] += int(y_shift)
        mask = np.zeros((130, 65), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        masks.append(mask.astype(bool))
    return tuple(masks)


def make_y_shift_mask_bank(profile: Profile) -> dict[int, tuple[np.ndarray, ...]]:
    if not profile.adaptive_y_shift:
        return {0: make_segment_masks(profile)}
    return {
        shift: make_shifted_segment_masks(profile, shift)
        for shift in range(profile.y_shift_min, profile.y_shift_max + 1)
    }


def pattern_quality(
    darkness: np.ndarray,
    patterns: dict[int, tuple[int, ...]],
) -> float:
    quality = 0.0
    for digit_darkness in darkness:
        best = digit_scores(digit_darkness, patterns)[0]
        quality += best[0] + 0.20 * max(0.0, 8.0 - best[2])
    return float(quality)


def clahe_patches(patches: Sequence[np.ndarray]) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    return [clahe.apply(patch) for patch in patches]


def extract_darkness_adaptive(
    display: np.ndarray,
    profile: Profile,
    mask_bank: dict[int, tuple[np.ndarray, ...]],
    previous_shift: int | None,
    contrast_mode: str,
) -> tuple[np.ndarray, int]:
    """Measure segment darkness while tracking vertical LCD geometry.

    The black-bezel crop can contain a slightly different amount of border in
    different videos.  A fixed seven-segment mask can therefore miss all three
    horizontal bars by 10--15 pixels even though the display looks perfectly
    readable.  We search a small bank of vertically shifted masks and track the
    best shift from frame to frame.

    CLAHE is available as a fallback, but raw grayscale is preferred whenever
    it gives an equally good seven-segment pattern because it separates active
    segments from faint inactive LCD ghosts more reliably.
    """
    patches = digit_patches(display, profile)
    shifts = sorted(mask_bank)
    if previous_shift is None:
        candidates = shifts
    else:
        candidates = [s for s in shifts if abs(s - previous_shift) <= 3]
        if not candidates:
            candidates = shifts

    best_score = float("-inf")
    best_shift = candidates[0]
    best_darkness: np.ndarray | None = None
    for shift in candidates:
        darkness = extract_darkness_from_patches(patches, mask_bank[shift])
        alignment_score = float(np.sum(np.maximum(darkness - 8.0, 0.0)))
        if previous_shift is not None:
            alignment_score -= 3.0 * abs(shift - previous_shift)
        if alignment_score > best_score:
            best_score = alignment_score
            best_shift = shift
            best_darkness = darkness

    # If local tracking lands on an edge with weak evidence, retry globally.
    if previous_shift is not None and (
        best_darkness is None
        or best_score < 90.0
        or abs(best_shift - previous_shift) >= 3
    ):
        for shift in shifts:
            darkness = extract_darkness_from_patches(patches, mask_bank[shift])
            alignment_score = float(np.sum(np.maximum(darkness - 8.0, 0.0)))
            alignment_score -= 1.0 * abs(shift - previous_shift)
            if alignment_score > best_score:
                best_score = alignment_score
                best_shift = shift
                best_darkness = darkness

    if best_darkness is None:
        raise RuntimeError("Could not evaluate adaptive segment masks.")

    if contrast_mode == "none":
        return best_darkness, best_shift

    enhanced = extract_darkness_from_patches(
        clahe_patches(patches), mask_bank[best_shift]
    )
    if contrast_mode == "clahe":
        return enhanced, best_shift

    patterns = profile.digit_patterns or STANDARD_DIGIT_PATTERNS
    raw_quality = pattern_quality(best_darkness, patterns)
    enhanced_quality = pattern_quality(enhanced, patterns)
    if enhanced_quality < 0.80 * raw_quality:
        return enhanced, best_shift
    return best_darkness, best_shift


def extract_decimal_scores(
    display: np.ndarray,
    profile: Profile,
    y_shift: int = 0,
) -> np.ndarray | None:
    """Measure local darkness of each configured decimal-point candidate.

    A tight box is used for the dot itself. Its median intensity is compared
    with a narrow surrounding ring, making the score insensitive to gradual
    illumination changes and reflections on the LCD.
    """
    if not profile.decimal_candidates:
        return None

    scores: list[float] = []
    height, width = display.shape
    margin = 5

    for x1, y1, x2, y2, _decimal_places in profile.decimal_candidates:
        y1 = int(round(y1 + y_shift))
        y2 = int(round(y2 + y_shift))
        patch = display[y1:y2, x1:x2]
        if patch.size == 0:
            scores.append(float("nan"))
            continue

        outer_x1 = max(0, x1 - margin)
        outer_y1 = max(0, y1 - margin)
        outer_x2 = min(width, x2 + margin)
        outer_y2 = min(height, y2 + margin)
        outer = display[outer_y1:outer_y2, outer_x1:outer_x2]

        ring_mask = np.ones(outer.shape, dtype=bool)
        ring_mask[
            y1 - outer_y1 : y2 - outer_y1,
            x1 - outer_x1 : x2 - outer_x1,
        ] = False

        ring = outer[ring_mask]
        background = float(np.percentile(ring, 85)) if ring.size else float(np.percentile(patch, 90))
        dot_level = float(np.median(patch))
        scores.append(background - dot_level)

    return np.asarray(scores, dtype=float)


def temporal_filter(
    data: np.ndarray,
    window: int,
    method: str,
) -> np.ndarray:

    if window <= 1:
        return data.copy()

    radius = window // 2

    filtered = np.full_like(
        data,
        np.nan,
    )

    for index in range(data.shape[0]):

        start = max(
            0,
            index - radius,
        )

        end = min(
            data.shape[0],
            index + radius + 1,
        )

        block = data[start:end]

        # Work column-by-column.  np.nanmedian/nanmax emits a RuntimeWarning
        # when every value in a slice is NaN; for such slices NaN is exactly
        # what we want anyway.
        flat_block = block.reshape(
            block.shape[0],
            -1,
        )

        flat_result = np.full(
            flat_block.shape[1],
            np.nan,
            dtype=float,
        )

        for column in range(
            flat_block.shape[1]
        ):

            values = flat_block[:, column]

            values = values[
                np.isfinite(values)
            ]

            if values.size == 0:
                continue

            if method == "median":

                flat_result[column] = float(
                    np.median(values)
                )

            elif method == "max":

                flat_result[column] = float(
                    np.max(values)
                )

            else:

                raise ValueError(
                    f"Unknown temporal filter: {method}"
                )

        filtered[index] = flat_result.reshape(
            data.shape[1:]
        )

    return filtered


def digit_scores(
    darkness: np.ndarray,
    patterns: dict[int, tuple[int, ...]],
) -> list[tuple[float, int, float]]:
    scores: list[tuple[float, int, float]] = []

    for digit, pattern in patterns.items():
        pattern_array = np.asarray(pattern, dtype=bool)
        active = darkness[pattern_array]
        inactive = darkness[~pattern_array]

        if inactive.size:
            active_mean = float(np.mean(active))
            inactive_mean = float(np.mean(inactive))
            contrast = active_mean - inactive_mean
            residual = float(
                np.sum((active - active_mean) ** 2)
                + np.sum((inactive - inactive_mean) ** 2)
            )
            score = residual / (7.0 * max(contrast, 1.0) ** 2)
        else:
            contrast = float(np.mean(active))
            score = float(np.var(active)) / max(contrast, 1.0) ** 2 + 0.15

        if contrast < 10.0:
            score += 2.0 * (10.0 - contrast) / 10.0

        scores.append((score, digit, contrast))

    return sorted(scores)


def decode_pattern_digit(
    darkness: np.ndarray,
    patterns: dict[int, tuple[int, ...]],
) -> tuple[int | None, float]:
    """Decode one RDS-200 digit.

    The original least-squares classifier is retained as the baseline, but it
    can confuse 7, 8 or 9 with 1 when horizontal/left segments are darker than
    the LCD background yet weaker than the two right-hand segments.  A second,
    conservative binary-pattern check corrects only those clear cases.

    The correction is deliberately asymmetric: it is applied when the baseline
    selected 1 (or rejected the digit), because that is the observed failure
    mode.  This avoids turning uniform glare into spurious 8s.
    """
    scores = digit_scores(darkness, patterns)
    best = scores[0]
    second = scores[1]
    confidence = 1.0 - best[0] / max(second[0], 1e-9)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    baseline_digit: int | None
    if best[2] < 8.0 or best[0] > 1.0:
        baseline_digit = None
    else:
        baseline_digit = int(best[1])

    levels = np.asarray(darkness, dtype=float)
    active_threshold = 7.0
    observed = tuple((levels >= active_threshold).astype(int))
    pattern_to_digit = {tuple(pattern): digit for digit, pattern in patterns.items()}
    binary_digit = pattern_to_digit.get(observed)

    if (
        binary_digit is not None
        and binary_digit != 1
        and binary_digit != baseline_digit
        and baseline_digit in (None, 1)
    ):
        pattern = np.asarray(patterns[binary_digit], dtype=bool)
        minimum_active = float(np.min(levels[pattern]))
        maximum_inactive = (
            float(np.max(levels[~pattern])) if np.any(~pattern) else float("-inf")
        )

        if binary_digit == 8:
            # With no inactive segments, require every segment to be clearly
            # dark.  This prevents reflections/poor crops from creating 8s.
            accepted = minimum_active >= 14.0
            separation = minimum_active - 14.0
        else:
            accepted = minimum_active >= 7.0 and maximum_inactive <= 6.0
            separation = min(minimum_active - 7.0, 6.0 - maximum_inactive)

        if accepted:
            binary_confidence = float(np.clip(0.5 + 0.08 * separation, 0.0, 1.0))
            return int(binary_digit), max(confidence, binary_confidence)

    return baseline_digit, confidence


def decode_prototype_digit(
    darkness: np.ndarray,
    position: int,
    prototypes: np.ndarray,
) -> tuple[int, float]:
    distances = np.mean((prototypes[position] - darkness[None, :]) ** 2, axis=1)
    order = np.argsort(distances)
    best = int(order[0])
    best_distance = float(distances[order[0]])
    second_distance = float(distances[order[1]])
    confidence = 1.0 - best_distance / max(second_distance, 1e-9)
    return best, float(np.clip(confidence, 0.0, 1.0))


def decode_rds30_margin_digit(
    darkness: np.ndarray,
    patterns: dict[int, tuple[int, ...]],
) -> tuple[int | None, float]:
    """Decode an RDS-30 digit from seven local-darkness measurements.

    For each possible glyph, calculate the threshold margin between its weakest
    active segment and its strongest inactive segment.  A positive margin means
    that one threshold separates the two classes exactly.  This directly uses
    the seven-segment geometry and avoids the old nearest-prototype failure modes:
    a blank field becoming 1, 8 becoming 6, and 9 becoming 8.

    Digit 8 has no inactive segment, so it is accepted only when *all* seven
    segments are genuinely dark.  The caller handles the valid leading blank.
    """
    levels = np.asarray(darkness, dtype=float)
    if levels.shape != (7,) or not np.all(np.isfinite(levels)):
        return None, 0.0

    candidates: list[tuple[float, float, int, float]] = []
    for digit, pattern in patterns.items():
        active_mask = np.asarray(pattern, dtype=bool)
        active = levels[active_mask]
        inactive = levels[~active_mask]

        if inactive.size:
            margin = float(np.min(active) - np.max(inactive))
            contrast = float(np.mean(active) - np.mean(inactive))
            score = margin + 0.10 * contrast
        else:
            # All seven bars must be clearly active before calling the glyph 8.
            margin = float(np.min(active) - 10.0)
            contrast = float(np.mean(active) - 10.0)
            score = margin + 0.10 * contrast

        candidates.append((score, margin, int(digit), contrast))

    candidates.sort(reverse=True)
    best = candidates[0]
    second = candidates[1]

    # If no clean threshold exists, retain the older continuous pattern fit as
    # a fallback.  This is useful for a weak segment under uneven illumination.
    if best[1] < -2.0:
        return decode_pattern_digit(levels, patterns)

    # Confidence combines the active/inactive margin with the lead over the
    # second-best glyph.  It is intentionally diagnostic rather than a hard gate.
    margin_confidence = 1.0 / (1.0 + math.exp(-best[1] / 3.0))
    gap_confidence = 1.0 / (1.0 + math.exp(-(best[0] - second[0]) / 2.0))
    confidence = float(np.clip(margin_confidence * gap_confidence, 0.0, 1.0))
    return best[2], confidence


def decode_hybrid_digit(
    prototype_darkness: np.ndarray,
    pattern_darkness: np.ndarray,
    position: int,
    prototypes: np.ndarray,
    patterns: dict[int, tuple[int, ...]],
    pattern_margin: float,
) -> tuple[int | None, float]:
    """Combine the RDS-30 position prototypes with a low-percentile pattern read.

    The prototype decoder handles position-dependent weak segments well, notably
    the weak lower-left segment that distinguishes 6 from 5.  It is, however,
    sensitive to illumination and used to confuse the RDS-30 glyphs 7 and 9
    with 4.  The auxiliary decoder samples the darkest 30 percent of each
    segment region and therefore sees the top segment reliably even when a
    mask only partly overlaps it.

    We select the pattern result when its confidence clearly exceeds the
    prototype result.  The 5/6 pair is deliberately left to the prototype
    decoder because that pair has a position-dependent weak segment on this
    LCD.
    """
    prototype_digit, prototype_confidence = decode_prototype_digit(
        prototype_darkness, position, prototypes
    )
    pattern_digit, pattern_confidence = decode_pattern_digit(
        pattern_darkness, patterns
    )

    if pattern_digit is None or pattern_digit == prototype_digit:
        confidence = max(
            prototype_confidence,
            pattern_confidence if pattern_digit == prototype_digit else 0.0,
        )
        # A nearest-prototype classifier always returns *some* digit.  On an
        # almost blank/misaligned patch the low-activity prototype for digit 1
        # is commonly the nearest one.  Do not promote such a guess to data.
        if confidence < 0.05:
            return None, confidence
        return prototype_digit, confidence

    if {prototype_digit, pattern_digit} == {5, 6}:
        return prototype_digit, max(
            prototype_confidence, 0.60 * pattern_confidence
        )

    if pattern_confidence > prototype_confidence + pattern_margin:
        return int(pattern_digit), pattern_confidence

    return prototype_digit, prototype_confidence


def decode_decimal_places_sequence(
    decimal_scores: np.ndarray,
    integer_values: Sequence[int | None],
    profile: Profile,
    switch_penalty: float,
) -> list[int]:
    """Choose a temporally stable moving-decimal sequence with Viterbi DP.

    The decimal point normally remains in one position for a whole run.  A
    switch is therefore penalized, except near the physical range boundary at
    about 10 uSv/h, where X.XX naturally changes to XX.X.
    """
    candidates = profile.decimal_candidates
    if not candidates:
        return [profile.decimal_places] * len(integer_values)

    n_samples = len(integer_values)
    n_states = len(candidates)
    if n_samples == 0:
        return []

    scores = np.asarray(decimal_scores, dtype=float)
    pair_differences = []
    for row in scores:
        finite = row[np.isfinite(row)]
        if finite.size >= 2:
            pair_differences.append(float(np.max(finite) - np.min(finite)))
    evidence_scale = max(2.0, float(np.median(pair_differences)) if pair_differences else 2.0)

    emission = np.zeros((n_samples, n_states), dtype=float)
    for index, row in enumerate(scores):
        if not np.isfinite(row).any():
            continue
        finite_max = float(np.nanmax(row))
        strength = float(np.clip(finite_max / 8.0, 0.0, 1.0))
        for state in range(n_states):
            if not np.isfinite(row[state]):
                emission[index, state] = 1.0
            else:
                emission[index, state] = (
                    strength * max(0.0, finite_max - float(row[state])) / evidence_scale
                )

    cost = np.full((n_samples, n_states), float("inf"), dtype=float)
    back = np.zeros((n_samples, n_states), dtype=np.int16)
    cost[0] = emission[0]

    for index in range(1, n_samples):
        for current in range(n_states):
            best_value = float("inf")
            best_previous = 0
            for previous in range(n_states):
                transition = 0.0
                if current != previous:
                    transition = switch_penalty
                    previous_integer = integer_values[index - 1]
                    current_integer = integer_values[index]
                    if previous_integer is not None and current_integer is not None:
                        previous_value = previous_integer / (
                            10 ** candidates[previous][4]
                        )
                        current_value = current_integer / (
                            10 ** candidates[current][4]
                        )
                        # The RDS-200 changes decimal position around 10 uSv/h.
                        if 7.0 <= previous_value <= 14.0 and 7.0 <= current_value <= 14.0:
                            transition *= 0.15
                candidate_cost = cost[index - 1, previous] + transition
                if candidate_cost < best_value:
                    best_value = candidate_cost
                    best_previous = previous
            cost[index, current] = best_value + emission[index, current]
            back[index, current] = best_previous

    states = np.zeros(n_samples, dtype=np.int16)
    states[-1] = int(np.argmin(cost[-1]))
    for index in range(n_samples - 1, 0, -1):
        states[index - 1] = back[index, states[index]]

    return [candidates[int(state)][4] for state in states]


def decode_samples(
    samples: Sequence[Sample],
    profile: Profile,
    filter_window: int,
    decimal_places_override: int | None = None,
    minimum_confidence: float = 0.0,
    decimal_switch_penalty: float = 4.0,
    decimal_sequence_observer: DecimalSequenceObserver | None = None,
) -> list[DecodedSample]:
    darkness = np.full(
        (len(samples), len(profile.digit_boxes), 7), np.nan, dtype=float
    )
    for index, sample in enumerate(samples):
        if sample.darkness is not None:
            darkness[index] = sample.darkness

    filtered = temporal_filter(darkness, filter_window, profile.temporal_filter)

    filtered_auxiliary: np.ndarray | None = None
    if profile.decoder in ("hybrid", "rds30_margin"):
        auxiliary = np.full_like(darkness, np.nan)
        for index, sample in enumerate(samples):
            if sample.aux_darkness is not None:
                auxiliary[index] = sample.aux_darkness
        filtered_auxiliary = temporal_filter(
            auxiliary, filter_window, profile.temporal_filter
        )

    filtered_decimal_scores: np.ndarray | None = None
    if profile.decimal_candidates and decimal_places_override is None:
        decimal_scores = np.full(
            (len(samples), len(profile.decimal_candidates)), np.nan, dtype=float
        )
        for index, sample in enumerate(samples):
            if sample.decimal_scores is not None:
                decimal_scores[index] = sample.decimal_scores
        filtered_decimal_scores = temporal_filter(
            decimal_scores, filter_window, "median"
        )

    patterns = profile.digit_patterns or STANDARD_DIGIT_PATTERNS
    integer_values: list[int | None] = []
    sample_confidences: list[float] = []

    for index in range(len(samples)):
        if np.isnan(filtered[index]).all():
            integer_values.append(None)
            sample_confidences.append(0.0)
            continue

        digits: list[int] = []
        confidences: list[float] = []
        for position, digit_darkness in enumerate(filtered[index]):
            if profile.decoder == "prototype":
                if profile.prototypes is None:
                    raise RuntimeError("Prototype decoder selected without prototypes.")
                digit, confidence = decode_prototype_digit(
                    digit_darkness, position, profile.prototypes
                )
            elif profile.decoder == "hybrid":
                if profile.prototypes is None or filtered_auxiliary is None:
                    raise RuntimeError("Hybrid decoder selected without auxiliary data.")
                digit, confidence = decode_hybrid_digit(
                    digit_darkness,
                    filtered_auxiliary[index, position],
                    position,
                    profile.prototypes,
                    patterns,
                    profile.hybrid_pattern_margin,
                )
            elif profile.decoder == "rds30_margin":
                if filtered_auxiliary is None:
                    raise RuntimeError("RDS-30 decoder selected without auxiliary data.")
                levels = filtered_auxiliary[index, position]
                if (
                    position == 0
                    and profile.leading_blank_threshold is not None
                    and float(np.max(levels)) < profile.leading_blank_threshold
                ):
                    # -1 is a valid leading blank, distinct from None (unrecognized).
                    digit = -1
                    confidence = float(
                        np.clip(
                            0.5
                            + (profile.leading_blank_threshold - float(np.max(levels)))
                            / 8.0,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    digit, confidence = decode_rds30_margin_digit(levels, patterns)
            else:
                digit, confidence = decode_pattern_digit(digit_darkness, patterns)
                if digit is None:
                    digits = []
                    break
            if digit is None:
                digits = []
                confidences.append(confidence)
                break
            digits.append(int(digit))
            confidences.append(confidence)

        confidence = min(confidences, default=0.0)
        if not digits or confidence < minimum_confidence:
            integer_values.append(None)
            sample_confidences.append(confidence)
            continue

        value_digits = digits
        if value_digits and value_digits[0] == -1:
            value_digits = value_digits[1:]
        if not value_digits or any(digit < 0 for digit in value_digits):
            integer_values.append(None)
            sample_confidences.append(confidence)
            continue

        integer_value = 0
        for digit in value_digits:
            integer_value = 10 * integer_value + digit
        integer_values.append(integer_value)
        sample_confidences.append(confidence)

    if decimal_places_override is not None:
        decimal_places_sequence = [decimal_places_override] * len(samples)
    elif filtered_decimal_scores is not None:
        decimal_places_sequence = decode_decimal_places_sequence(
            filtered_decimal_scores,
            integer_values,
            profile,
            switch_penalty=decimal_switch_penalty,
        )
    else:
        decimal_places_sequence = [profile.decimal_places] * len(samples)

    if decimal_sequence_observer is not None:
        decimal_sequence_observer(
            list(decimal_places_sequence)
        )

    decoded: list[DecodedSample] = []
    for sample, integer_value, confidence, decimal_places in zip(
        samples, integer_values, sample_confidences, decimal_places_sequence
    ):
        if integer_value is None:
            decoded.append(DecodedSample(sample.time_s, None, confidence))
        else:
            decoded.append(
                DecodedSample(
                    sample.time_s,
                    integer_value / (10 ** decimal_places),
                    confidence,
                )
            )
    return decoded


def fill_unrecognized(decoded: Sequence[DecodedSample]) -> list[DecodedSample]:
    result = [DecodedSample(item.time_s, item.value, item.confidence) for item in decoded]
    recognized_indices = [i for i, item in enumerate(result) if item.value is not None]
    if not recognized_indices:
        return result

    first = recognized_indices[0]
    for index in range(first):
        result[index].value = result[first].value
        result[index].confidence = 0.0

    last = recognized_indices[-1]
    for index in range(last + 1, len(result)):
        result[index].value = result[last].value
        result[index].confidence = 0.0

    index = first
    while index <= last:
        if result[index].value is not None:
            index += 1
            continue
        gap_start = index
        while index <= last and result[index].value is None:
            index += 1
        gap_end = index
        left = result[gap_start - 1]
        right = result[gap_end]
        split = (gap_start + gap_end) // 2
        for gap_index in range(gap_start, gap_end):
            source = left if gap_index < split else right
            result[gap_index].value = source.value
            result[gap_index].confidence = 0.0

    return result


def centered_mode(decoded: Sequence[DecodedSample], window: int) -> list[DecodedSample]:
    if window <= 1:
        return [DecodedSample(x.time_s, x.value, x.confidence) for x in decoded]
    radius = window // 2
    result: list[DecodedSample] = []

    for index, item in enumerate(decoded):
        start = max(0, index - radius)
        end = min(len(decoded), index + radius + 1)
        values = [x.value for x in decoded[start:end] if x.value is not None]
        if not values:
            result.append(DecodedSample(item.time_s, None, 0.0))
            continue

        counts = Counter(values)
        maximum = max(counts.values())
        candidates = {value for value, count in counts.items() if count == maximum}
        chosen = item.value if item.value in candidates else min(candidates)
        confidences = [
            x.confidence
            for x in decoded[start:end]
            if x.value == chosen
        ]
        result.append(
            DecodedSample(
                item.time_s,
                chosen,
                float(np.mean(confidences)) if confidences else 0.0,
            )
        )

    return result


def make_runs(decoded: Sequence[DecodedSample]) -> list[Run]:
    if not decoded:
        return []
    if decoded[0].value is None:
        raise RuntimeError("The video contains no recognized values.")

    runs: list[Run] = []
    start = 0
    for index in range(1, len(decoded) + 1):
        changed = index == len(decoded) or decoded[index].value != decoded[start].value
        if not changed:
            continue
        confidence = float(np.mean([item.confidence for item in decoded[start:index]]))
        runs.append(
            Run(
                start_index=start,
                end_index=index,
                value=float(decoded[start].value),
                confidence=confidence,
            )
        )
        start = index
    return runs


def merge_short_runs(
    runs: list[Run],
    minimum_samples: int,
    preserve_edges: bool,
) -> list[Run]:
    if minimum_samples <= 1:
        return runs

    runs = [Run(r.start_index, r.end_index, r.value, r.confidence) for r in runs]

    while len(runs) > 1:
        short_index = next(
            (
                i
                for i, run in enumerate(runs)
                if run.sample_count < minimum_samples
                and not (preserve_edges and (i == 0 or i == len(runs) - 1))
            ),
            None,
        )
        if short_index is None:
            break

        run = runs[short_index]
        if short_index == 0:
            target_index = 1
        elif short_index == len(runs) - 1:
            target_index = short_index - 1
        else:
            previous = runs[short_index - 1]
            following = runs[short_index + 1]
            previous_distance = abs(run.value - previous.value)
            following_distance = abs(run.value - following.value)
            if previous_distance < following_distance:
                target_index = short_index - 1
            elif following_distance < previous_distance:
                target_index = short_index + 1
            else:
                target_index = (
                    short_index - 1
                    if previous.confidence >= following.confidence
                    else short_index + 1
                )

        target = runs[target_index]
        combined_samples = run.sample_count + target.sample_count
        combined_confidence = (
            run.confidence * run.sample_count
            + target.confidence * target.sample_count
        ) / combined_samples

        if target_index < short_index:
            target.end_index = run.end_index
            target.confidence = combined_confidence
            del runs[short_index]
        else:
            target.start_index = run.start_index
            target.confidence = combined_confidence
            del runs[short_index]

        compacted: list[Run] = []
        for item in runs:
            if compacted and compacted[-1].value == item.value:
                previous = compacted[-1]
                total = previous.sample_count + item.sample_count
                previous.confidence = (
                    previous.confidence * previous.sample_count
                    + item.confidence * item.sample_count
                ) / total
                previous.end_index = item.end_index
            else:
                compacted.append(item)
        runs = compacted

    return runs


def choose_run_values_from_centers(
    runs: Sequence[Run],
    decoded: Sequence[DecodedSample],
) -> list[Run]:
    result: list[Run] = []
    for run in runs:
        center = (run.start_index + run.end_index) // 2
        center_value = decoded[center].value
        value = run.value if center_value is None else float(center_value)
        confidence = float(
            np.mean([x.confidence for x in decoded[run.start_index : run.end_index]])
        )
        result.append(
            Run(run.start_index, run.end_index, value, confidence)
        )
    return result


def run_boundaries(
    runs: Sequence[Run], sample_fps: float, duration: float
) -> list[tuple[float, float, Run]]:
    result: list[tuple[float, float, Run]] = []
    for index, run in enumerate(runs):
        start = 0.0 if index == 0 else (run.start_index - 0.5) / sample_fps
        end = duration if index == len(runs) - 1 else (run.end_index - 0.5) / sample_fps
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        result.append((start, end, run))
    return result


def write_interval_csv(
    path: Path,
    intervals: Sequence[tuple[float, float, Run]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["start_s", "end_s", "duration_s", "value", "confidence", "samples"]
        )
        for start, end, run in intervals:
            writer.writerow(
                [
                    f"{start:.3f}",
                    f"{end:.3f}",
                    f"{end - start:.3f}",
                    f"{run.value:.6g}",
                    f"{run.confidence:.4f}",
                    run.sample_count,
                ]
            )


def write_raw_csv(path: Path, decoded: Sequence[DecodedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "value", "confidence"])
        for sample in decoded:
            writer.writerow(
                [
                    f"{sample.time_s:.3f}",
                    "" if sample.value is None else f"{sample.value:.6g}",
                    f"{sample.confidence:.4f}",
                ]
            )


def save_debug_screenshots(
    video: Path,
    info: VideoInfo,
    profile: Profile,
    sample_fps: float,
    processing_width: int,
    intervals: Sequence[tuple[float, float, Run]],
    output_dir: Path,
    contrast_mode: str = "auto",
    display_finder: DisplayFinder | None = None,
) -> None:
    """Save one normalized display image from the center of every final interval."""
    selected_display_finder = (
        find_display_crop
        if display_finder is None
        else display_finder
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[int, tuple[int, float, float, Run]] = {}
    for interval_number, (start, end, run) in enumerate(intervals, start=1):
        center_time = 0.5 * (start + end)
        frame_index = max(0, int(round(center_time * sample_fps)))
        targets[frame_index] = (interval_number, start, end, run)

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "interval",
                "start_s",
                "end_s",
                "center_s",
                "value",
                "confidence",
                "image",
            ]
        )

        previous_box: tuple[int, int, int, int] | None = None
        previous_shift: int | None = None
        shift_mask_bank = make_y_shift_mask_bank(profile)
        pending = set(targets)

        for frame_index, frame in enumerate(
            iter_ffmpeg_frames(video, info, sample_fps, processing_width)
        ):
            display, previous_box = selected_display_finder(
                frame,
                profile,
                previous_box,
            )
            if display is not None and profile.adaptive_y_shift:
                effective_contrast = (
                    contrast_mode if profile.name == "rds200" else "none"
                )
                _darkness, previous_shift = extract_darkness_adaptive(
                    display,
                    profile,
                    shift_mask_bank,
                    previous_shift,
                    effective_contrast,
                )
            else:
                previous_shift = 0
            if frame_index not in pending:
                continue

            interval_number, start, end, run = targets[frame_index]
            center_time = frame_index / sample_fps
            value_text = f"{run.value:.6g}"
            safe_value = value_text.replace("-", "m").replace(".", "p")
            filename = (
                f"interval_{interval_number:03d}_"
                f"t{center_time:08.3f}_value_{safe_value}.jpg"
            )

            if display is None:
                canvas = np.full((120, 640, 3), 255, dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f"t={center_time:.3f} s  value={value_text}  DISPLAY NOT FOUND",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                shift = previous_shift or 0
                for x1, y1, x2, y2 in profile.digit_boxes:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), 1)
                    box_width = max(1, x2 - x1)
                    box_height = max(1, y2 - y1)
                    scale_x = box_width / 65.0
                    scale_y = box_height / 130.0
                    for name in SEGMENT_ORDER:
                        # Extraction first resizes each digit box to 65 x 130.
                        # Draw the inverse-transformed polygon here so the debug
                        # image shows the pixels that were actually measured.
                        local = profile.segment_polygons[name].astype(float)
                        points = np.empty_like(local, dtype=np.int32)
                        points[:, 0] = np.rint(x1 + local[:, 0] * scale_x).astype(np.int32)
                        points[:, 1] = np.rint(
                            y1 + (local[:, 1] + shift) * scale_y
                        ).astype(np.int32)
                        cv2.polylines(overlay, [points], True, (0, 180, 0), 1)
                for x1, y1, x2, y2, decimal_places in profile.decimal_candidates:
                    y1 += shift
                    y2 += shift
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), 2)
                    cv2.putText(
                        overlay,
                        f".{decimal_places}",
                        (x1 - 4, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 220),
                        1,
                        cv2.LINE_AA,
                    )

                header_height = 48
                canvas = np.full(
                    (overlay.shape[0] + header_height, overlay.shape[1], 3),
                    255,
                    dtype=np.uint8,
                )
                canvas[header_height:] = overlay
                cv2.putText(
                    canvas,
                    (
                        f"interval {interval_number}: t={center_time:.3f} s, "
                        f"value={value_text}, confidence={run.confidence:.3f}, "
                        f"y_shift={previous_shift or 0}"
                    ),
                    (10, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            image_path = output_dir / filename
            if not cv2.imwrite(str(image_path), canvas):
                raise RuntimeError(f"Could not write debug image: {image_path}")

            writer.writerow(
                [
                    interval_number,
                    f"{start:.3f}",
                    f"{end:.3f}",
                    f"{center_time:.3f}",
                    value_text,
                    f"{run.confidence:.4f}",
                    filename,
                ]
            )
            pending.remove(frame_index)
            if not pending:
                break


def calculate_summary(
    intervals: Sequence[tuple[float, float, Run]],
    video: Path,
    profile: Profile,
    duration: float,
    display_found_fraction: float,
    recognized_fraction: float,
) -> dict[str, object]:
    values = np.asarray([run.value for _, _, run in intervals], dtype=float)
    weights = np.asarray([end - start for start, end, _ in intervals], dtype=float)
    total_weight = float(np.sum(weights))

    mean = float(np.sum(values * weights) / total_weight)
    variance = float(np.sum(weights * (values - mean) ** 2) / total_weight)

    return {
        "input_video": str(video),
        "profile": profile.name,
        "video_duration_s": duration,
        "display_found_fraction": display_found_fraction,
        "recognized_fraction": recognized_fraction,
        "interval_count": len(intervals),
        "time_weighted_mean": mean,
        "time_weighted_variance": variance,
        "time_weighted_std_dev": math.sqrt(variance),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def print_summary(summary: dict[str, object]) -> None:
    print(f"Video duration:          {summary['video_duration_s']:10.3f} s")
    print(f"Display found:           {100 * float(summary['display_found_fraction']):10.2f} %")
    print(f"Recognized samples:      {100 * float(summary['recognized_fraction']):10.2f} %")
    print(f"Number of intervals:     {int(summary['interval_count']):10d}")
    print()
    print(f"Time-weighted mean:      {float(summary['time_weighted_mean']):10.5g}")
    print(f"Time-weighted variance:  {float(summary['time_weighted_variance']):10.5g}")
    print(f"Time-weighted std. dev.: {float(summary['time_weighted_std_dev']):10.5g}")
    print(f"Minimum:                 {float(summary['minimum']):10.5g}")
    print(f"Maximum:                 {float(summary['maximum']):10.5g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract changing seven-segment values from a dosimeter video."
    )
    parser.add_argument("video", type=Path, help="input video, for example video.MOV")
    parser.add_argument("output", type=Path, help="interval CSV output")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="rds200",
        help="dosimeter profile (default: rds200)",
    )
    parser.add_argument(
        "--decimal-places",
        choices=("auto", "0", "1", "2", "3"),
        default="auto",
        help=(
            "decimal places: auto detects the moving RDS-200 decimal point; "
            "a number forces a fixed position (default: auto)"
        ),
    )
    parser.add_argument(
        "--contrast",
        choices=("auto", "none", "clahe"),
        default="auto",
        help=(
            "RDS-200 digit contrast handling: auto prefers raw grayscale and "
            "uses CLAHE only when it clearly improves pattern quality "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="reject samples below this digit confidence; profile default if omitted",
    )
    parser.add_argument(
        "--decimal-switch-penalty",
        type=float,
        default=4.0,
        help=(
            "penalty for moving the decimal point between adjacent samples; "
            "automatically relaxed near 10 uSv/h (default: 4.0)"
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="samples per second; profile default if omitted",
    )
    parser.add_argument(
        "--filter-window",
        "--median-window",
        dest="filter_window",
        type=int,
        default=None,
        help="odd temporal segment-filter window; profile default if omitted",
    )
    parser.add_argument(
        "--mode-window",
        type=int,
        default=None,
        help="odd temporal value-mode window; profile default if omitted",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.6,
        help="merge recognized intervals shorter than this many seconds (default: 0.6)",
    )
    parser.add_argument(
        "--processing-width",
        type=int,
        default=540,
        help="working video width used by ffmpeg (default: 540)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary JSON path; default is OUTPUT.summary.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="optional CSV containing every decoded sample before interval merging",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for one normalized display screenshot from the "
            "center of every final interval"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise RuntimeError(f"Input video does not exist: {args.video}")
    if args.sample_fps is not None and args.sample_fps <= 0:
        raise RuntimeError("--sample-fps must be positive.")
    if args.processing_width < 200:
        raise RuntimeError("--processing-width must be at least 200 pixels.")
    for name, value in (
        ("--filter-window", args.filter_window),
        ("--mode-window", args.mode_window),
    ):
        if value is not None and (value < 1 or value % 2 == 0):
            raise RuntimeError(f"{name} must be a positive odd integer.")
    if args.min_interval < 0:
        raise RuntimeError("--min-interval cannot be negative.")
    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        raise RuntimeError("--min-confidence must be between 0 and 1.")
    if args.decimal_switch_penalty < 0:
        raise RuntimeError("--decimal-switch-penalty cannot be negative.")
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(
                f"{program} was not found. On macOS it can be installed with: "
                "brew install ffmpeg"
            )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        profile = PROFILES[args.profile]
        sample_fps = (
            args.sample_fps
            if args.sample_fps is not None
            else profile.default_sample_fps
        )
        filter_window = (
            args.filter_window
            if args.filter_window is not None
            else profile.default_filter_window
        )
        mode_window = (
            args.mode_window
            if args.mode_window is not None
            else profile.default_mode_window
        )

        info = probe_video(args.video)
        segment_masks = make_segment_masks(profile)
        shift_mask_bank = make_y_shift_mask_bank(profile)
        minimum_confidence = (
            args.min_confidence
            if args.min_confidence is not None
            else profile.default_min_confidence
        )
        samples: list[Sample] = []
        previous_box: tuple[int, int, int, int] | None = None
        previous_shift: int | None = None

        print(
            f"Processing {args.video} with profile {profile.name} "
            f"at {sample_fps:g} samples/s "
            f"(contrast={args.contrast}, min_confidence={minimum_confidence:g})...",
            file=sys.stderr,
        )

        for frame_index, frame in enumerate(
            iter_ffmpeg_frames(args.video, info, sample_fps, args.processing_width)
        ):
            display, previous_box = find_display_crop(frame, profile, previous_box)
            if display is None:
                samples.append(
                    Sample(
                        frame_index / sample_fps,
                        None,
                        display_found=False,
                        aux_darkness=None,
                        decimal_scores=None,
                        alignment_shift=previous_shift or 0,
                    )
                )
            else:
                if profile.adaptive_y_shift:
                    # CLAHE was introduced for the RDS-200.  On the RDS-30 the
                    # raw LCD grayscale plus temporal maximum is more stable and
                    # substantially faster, so only its geometry is adapted.
                    effective_contrast = (
                        args.contrast if profile.name == "rds200" else "none"
                    )
                    measured_darkness, previous_shift = extract_darkness_adaptive(
                        display,
                        profile,
                        shift_mask_bank,
                        previous_shift,
                        effective_contrast,
                    )
                else:
                    measured_darkness = extract_darkness(
                        display, profile, segment_masks
                    )
                    previous_shift = 0

                auxiliary_darkness = None
                if profile.aux_segment_percentile is not None:
                    # The auxiliary RDS-30 pattern decoder must use the same
                    # shifted masks as the primary decoder.  Previously it used
                    # the unshifted grid, which missed the top bars and changed
                    # 7/3/0 into 1-like glyphs in videos with a slightly different
                    # bezel crop.
                    active_masks = shift_mask_bank.get(
                        previous_shift or 0, segment_masks
                    )
                    auxiliary_darkness = extract_darkness_from_patches(
                        digit_patches(display, profile),
                        active_masks,
                        segment_percentile=profile.aux_segment_percentile,
                    )

                samples.append(
                    Sample(
                        frame_index / sample_fps,
                        measured_darkness,
                        display_found=True,
                        aux_darkness=auxiliary_darkness,
                        decimal_scores=extract_decimal_scores(
                            display, profile, y_shift=previous_shift or 0
                        ),
                        alignment_shift=previous_shift or 0,
                    )
                )

        if not samples:
            raise RuntimeError("No frames were decoded from the video.")

        decimal_places_override = (
            None if args.decimal_places == "auto" else int(args.decimal_places)
        )
        decoded_raw = decode_samples(
            samples,
            profile,
            filter_window,
            decimal_places_override=decimal_places_override,
            minimum_confidence=minimum_confidence,
            decimal_switch_penalty=args.decimal_switch_penalty,
        )
        display_found_fraction = sum(s.display_found for s in samples) / len(samples)
        recognized_fraction = sum(s.value is not None for s in decoded_raw) / len(decoded_raw)
        print(
            f"Display detection: "
            f"{100.0 * display_found_fraction:.1f}% "
            f"({sum(s.display_found for s in samples)}/{len(samples)} frames)",
            file=sys.stderr,
        )

        print(
            f"Digit recognition: "
            f"{100.0 * recognized_fraction:.1f}% "
            f"({sum(s.value is not None for s in decoded_raw)}/{len(decoded_raw)} frames)",
            file=sys.stderr,
        )

        if args.raw_output is not None:
            write_raw_csv(args.raw_output, decoded_raw)

        filled = fill_unrecognized(decoded_raw)
        smoothed = centered_mode(filled, mode_window)
        runs = make_runs(smoothed)
        minimum_samples = max(1, int(math.ceil(args.min_interval * sample_fps)))
        runs = merge_short_runs(
            runs,
            minimum_samples,
            preserve_edges=profile.preserve_edge_runs,
        )
        if profile.choose_run_value_at_center:
            runs = choose_run_values_from_centers(runs, filled)
        intervals = run_boundaries(runs, sample_fps, info.duration)

        write_interval_csv(args.output, intervals)

        if args.debug_dir is not None:
            save_debug_screenshots(
                args.video,
                info,
                profile,
                sample_fps,
                args.processing_width,
                intervals,
                args.debug_dir,
                contrast_mode=args.contrast,
            )

        summary_path = args.summary
        if summary_path is None:
            summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")

        summary = calculate_summary(
            intervals,
            args.video,
            profile,
            info.duration,
            display_found_fraction,
            recognized_fraction,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print_summary(summary)
        print()
        print(f"Intervals written to:    {args.output}")
        print(f"Summary written to:      {summary_path}")
        if args.raw_output is not None:
            print(f"Raw samples written to:  {args.raw_output}")
        if args.debug_dir is not None:
            print(f"Debug screenshots:       {args.debug_dir}")
        return 0

    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
