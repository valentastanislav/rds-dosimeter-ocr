#!/usr/bin/env python3
"""Opt-in coupled-geometry decoder experiment for the RDS-30 LCD.

The normal OCR pipeline owns stabilization, rectification, decimal handling,
and interval construction.  This module only consumes the exact rectified
displays cached during that main pass and returns ordinary ``DecodedSample``
objects.  It deliberately has no access to ground-truth values.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import dosimeter_get_values as core


HORIZONTAL_SEGMENTS = frozenset(("a", "d", "g"))


@dataclass(frozen=True, order=True)
class CoupledGeometryState:
    """Residual mask field in normalized 65 x 130 digit coordinates."""

    tx: int
    ty: int
    kx: int
    ky: int


@dataclass(frozen=True)
class SegmentEvidence:
    global_contrast: float
    local_contrast: float
    oriented_edge: float
    cross_edge: float
    continuity: float


@dataclass(frozen=True)
class SegmentEvidenceModel:
    intercept: float
    global_weight: float
    local_weight: float
    oriented_edge_weight: float
    cross_edge_weight: float
    continuity_weight: float
    disagreement_weight: float


@dataclass(frozen=True)
class JointSpatialConfig:
    tx_values: tuple[int, ...] = (-10, -5, 0, 5, 10)
    ty_values: tuple[int, ...] = (-8, -4, 0, 4, 8)
    kx_values: tuple[int, ...] = (-6, 0, 6)
    ky_values: tuple[int, ...] = (-4, 0, 4)
    temporal_window: int = 5
    coarse_candidate_count: int = 24
    temporal_translation_weight: float = 0.16
    temporal_differential_weight: float = 0.08
    geometry_boundary_penalty: float = 0.20
    rejection_threshold: float = 0.40
    horizontal_model: SegmentEvidenceModel = SegmentEvidenceModel(
        intercept=-2.0,
        global_weight=0.045,
        local_weight=0.055,
        oriented_edge_weight=0.025,
        cross_edge_weight=-0.010,
        continuity_weight=0.025,
        disagreement_weight=-0.018,
    )
    vertical_model: SegmentEvidenceModel = SegmentEvidenceModel(
        intercept=-2.0,
        global_weight=0.045,
        local_weight=0.055,
        oriented_edge_weight=0.025,
        cross_edge_weight=-0.010,
        continuity_weight=0.025,
        disagreement_weight=-0.018,
    )


@dataclass
class JointSpatialFrameDiagnostic:
    frame_index: int
    time_s: float
    state: CoupledGeometryState
    offsets: tuple[tuple[int, int], ...]
    digits: tuple[int, ...]
    digit_margins: tuple[float, ...]
    segment_entropy: float
    evidence_agreement: float
    spatial_quality: float
    geometry_gap: float
    temporal_change: float
    boundary: bool
    confidence: float
    accepted: bool


@dataclass
class JointSpatialDiagnostics:
    frames: list[JointSpatialFrameDiagnostic] = field(default_factory=list)


def normalized_digit_x(position: int, digit_count: int) -> float:
    if digit_count <= 1:
        return 0.0
    return -1.0 + 2.0 * position / (digit_count - 1)


def geometry_offsets(
    state: CoupledGeometryState,
    digit_count: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(round(state.tx + state.kx * normalized_digit_x(position, digit_count))),
            int(round(state.ty + state.ky * normalized_digit_x(position, digit_count))),
        )
        for position in range(digit_count)
    )


def temporal_geometry_penalty(
    previous: CoupledGeometryState,
    current: CoupledGeometryState,
    config: JointSpatialConfig,
) -> float:
    translation = abs(current.tx - previous.tx) + abs(current.ty - previous.ty)
    differential = abs(current.kx - previous.kx) + abs(current.ky - previous.ky)
    return (
        config.temporal_translation_weight * translation
        + config.temporal_differential_weight * differential
    )


def combine_segment_evidence(
    evidence: SegmentEvidence,
    model: SegmentEvidenceModel,
) -> float:
    disagreement = abs(evidence.global_contrast - evidence.local_contrast)
    logit = (
        model.intercept
        + model.global_weight * evidence.global_contrast
        + model.local_weight * evidence.local_contrast
        + model.oriented_edge_weight * evidence.oriented_edge
        + model.cross_edge_weight * evidence.cross_edge
        + model.continuity_weight * evidence.continuity
        + model.disagreement_weight * disagreement
    )
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def glyph_log_likelihood(
    probabilities: np.ndarray,
    pattern: Sequence[int],
) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    expected = np.asarray(pattern, dtype=bool)
    return float(np.sum(np.where(expected, np.log(values), np.log1p(-values))))


def confidence_accepts(confidence: float, threshold: float) -> bool:
    return math.isfinite(confidence) and confidence >= threshold


def translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    source_x1 = max(0, -dx)
    source_y1 = max(0, -dy)
    source_x2 = min(width, width - dx)
    source_y2 = min(height, height - dy)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return result
    target_x1 = source_x1 + dx
    target_y1 = source_y1 + dy
    target_x2 = source_x2 + dx
    target_y2 = source_y2 + dy
    result[target_y1:target_y2, target_x1:target_x2] = mask[
        source_y1:source_y2,
        source_x1:source_x2,
    ]
    return result


def _outside_band(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = translate_mask(mask, dx, dy)
    return shifted & ~mask


def _masked_percentile(image: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    pixels = image[mask]
    if pixels.size == 0:
        return float("nan")
    return float(np.percentile(pixels, percentile))


def _continuity(
    patch: np.ndarray,
    mask: np.ndarray,
    background: float,
    horizontal: bool,
) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan")
    coordinates = xs if horizontal else ys
    low = int(np.min(coordinates))
    high = int(np.max(coordinates)) + 1
    edges = np.linspace(low, high, 5)
    values: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if horizontal:
            part = mask & (np.indices(mask.shape)[1] >= start) & (np.indices(mask.shape)[1] < end)
        else:
            part = mask & (np.indices(mask.shape)[0] >= start) & (np.indices(mask.shape)[0] < end)
        level = _masked_percentile(patch, part, 30.0)
        if math.isfinite(level):
            values.append(background - level)
    return min(values, default=float("nan"))


def measure_segment_evidence(
    patch: np.ndarray,
    mask: np.ndarray,
    ring: np.ndarray,
    segment_name: str,
) -> SegmentEvidence:
    horizontal = segment_name in HORIZONTAL_SEGMENTS
    global_background = float(np.percentile(patch, 90.0))
    segment_p30 = _masked_percentile(patch, mask, 30.0)
    segment_p10 = _masked_percentile(patch, mask, 10.0)
    local_background = _masked_percentile(patch, ring, 70.0)

    if horizontal:
        normal_bands = (_outside_band(mask, 0, -4), _outside_band(mask, 0, 4))
        cross_bands = (_outside_band(mask, -4, 0), _outside_band(mask, 4, 0))
    else:
        normal_bands = (_outside_band(mask, -4, 0), _outside_band(mask, 4, 0))
        cross_bands = (_outside_band(mask, 0, -4), _outside_band(mask, 0, 4))

    inside_level = _masked_percentile(patch, mask, 30.0)
    normal_levels = [_masked_percentile(patch, band, 60.0) for band in normal_bands]
    cross_levels = [_masked_percentile(patch, band, 60.0) for band in cross_bands]
    oriented_edge = float(np.mean(normal_levels) - inside_level)
    cross_edge = float(np.mean(cross_levels) - inside_level)
    continuity = _continuity(patch, mask, local_background, horizontal)

    return SegmentEvidence(
        global_contrast=global_background - segment_p30,
        local_contrast=local_background - segment_p10,
        oriented_edge=oriented_edge,
        cross_edge=cross_edge,
        continuity=continuity,
    )


def _percentile_series(
    patches: np.ndarray,
    mask: np.ndarray,
    percentile: float,
) -> np.ndarray:
    pixels = patches[:, mask]
    if pixels.shape[1] == 0:
        return np.full(len(patches), np.nan, dtype=float)
    return np.percentile(pixels, percentile, axis=1)


def _segment_evidence_series(
    patches: np.ndarray,
    mask: np.ndarray,
    ring: np.ndarray,
    segment_name: str,
    global_background: np.ndarray,
) -> np.ndarray:
    horizontal = segment_name in HORIZONTAL_SEGMENTS
    if horizontal:
        normal_bands = (_outside_band(mask, 0, -4), _outside_band(mask, 0, 4))
        cross_bands = (_outside_band(mask, -4, 0), _outside_band(mask, 4, 0))
    else:
        normal_bands = (_outside_band(mask, -4, 0), _outside_band(mask, 4, 0))
        cross_bands = (_outside_band(mask, 0, -4), _outside_band(mask, 0, 4))

    segment_p30 = _percentile_series(patches, mask, 30.0)
    segment_p10 = _percentile_series(patches, mask, 10.0)
    local_background = _percentile_series(patches, ring, 70.0)
    normal_values = [
        _percentile_series(patches, band, 60.0)
        for band in normal_bands
        if np.any(band)
    ]
    cross_values = [
        _percentile_series(patches, band, 60.0)
        for band in cross_bands
        if np.any(band)
    ]
    normal = (
        np.mean(normal_values, axis=0) - segment_p30
        if normal_values
        else np.zeros(len(patches), dtype=float)
    )
    cross = (
        np.mean(cross_values, axis=0) - segment_p30
        if cross_values
        else np.zeros(len(patches), dtype=float)
    )

    ys, xs = np.nonzero(mask)
    coordinates = xs if horizontal else ys
    continuity_parts: list[np.ndarray] = []
    if coordinates.size:
        low = int(np.min(coordinates))
        high = int(np.max(coordinates)) + 1
        coordinate_grid = np.indices(mask.shape)[1 if horizontal else 0]
        edges = np.linspace(low, high, 5)
        for start, end in zip(edges[:-1], edges[1:]):
            part = mask & (coordinate_grid >= start) & (coordinate_grid < end)
            if np.any(part):
                continuity_parts.append(
                    local_background - _percentile_series(patches, part, 30.0)
                )
    continuity = (
        np.min(continuity_parts, axis=0)
        if continuity_parts
        else np.full(len(patches), np.nan, dtype=float)
    )
    return np.column_stack(
        (
            global_background - segment_p30,
            local_background - segment_p10,
            normal,
            cross,
            continuity,
        )
    )


def _probability_series(
    evidence: np.ndarray,
    model: SegmentEvidenceModel,
) -> np.ndarray:
    evidence = np.nan_to_num(evidence, nan=-20.0, posinf=80.0, neginf=-20.0)
    disagreement = np.abs(evidence[:, 0] - evidence[:, 1])
    logits = (
        model.intercept
        + model.global_weight * evidence[:, 0]
        + model.local_weight * evidence[:, 1]
        + model.oriented_edge_weight * evidence[:, 2]
        + model.cross_edge_weight * evidence[:, 3]
        + model.continuity_weight * evidence[:, 4]
        + model.disagreement_weight * disagreement
    )
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _temporal_median_evidence(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(data) == 0:
        return data.copy()
    radius = window // 2
    result = np.empty_like(data, dtype=float)
    for index in range(min(radius, len(data))):
        result[index] = np.median(data[: index + radius + 1], axis=0)
    if len(data) > 2 * radius:
        windows = np.lib.stride_tricks.sliding_window_view(
            data,
            window_shape=window,
            axis=0,
        )
        result[radius : len(data) - radius] = np.median(windows, axis=-1)
    for index in range(max(radius, len(data) - radius), len(data)):
        result[index] = np.median(data[index - radius :], axis=0)
    return result


def make_local_masks(
    profile: core.Profile,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks = core.make_segment_masks(profile)
    kernel = np.ones((9, 9), np.uint8)
    rings: list[np.ndarray] = []
    for mask in masks:
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        rings.append(dilated & ~mask)
    return masks, rings


def _digit_patches_from_cache(
    display_cache: dict[int, np.ndarray],
    profile: core.Profile,
    sample_count: int,
) -> list[np.ndarray]:
    by_position: list[list[np.ndarray]] = [[] for _ in profile.digit_boxes]
    fallback = np.full((130, 65), 255, dtype=np.uint8)
    for frame_index in range(sample_count):
        display = display_cache.get(frame_index)
        if display is None:
            for target in by_position:
                target.append(fallback)
            continue
        patches = core.digit_patches(display, profile)
        for target, patch in zip(by_position, patches):
            if patch.ndim == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            target.append(patch.astype(np.uint8, copy=False))
    return [np.asarray(position_patches) for position_patches in by_position]


def _all_states(config: JointSpatialConfig) -> list[CoupledGeometryState]:
    return [
        CoupledGeometryState(tx, ty, kx, ky)
        for tx in config.tx_values
        for ty in config.ty_values
        for kx in config.kx_values
        for ky in config.ky_values
    ]


def _boundary(state: CoupledGeometryState, config: JointSpatialConfig) -> bool:
    return (
        state.tx in (config.tx_values[0], config.tx_values[-1])
        or state.ty in (config.ty_values[0], config.ty_values[-1])
        or state.kx in (config.kx_values[0], config.kx_values[-1])
        or state.ky in (config.ky_values[0], config.ky_values[-1])
    )


def _state_distance(first: CoupledGeometryState, second: CoupledGeometryState) -> float:
    return float(
        abs(first.tx - second.tx)
        + abs(first.ty - second.ty)
        + 0.5 * abs(first.kx - second.kx)
        + 0.5 * abs(first.ky - second.ky)
    )


class JointSpatialDecoder:
    def __init__(
        self,
        display_cache: dict[int, np.ndarray],
        config: JointSpatialConfig | None = None,
        diagnostics_dir: Path | None = None,
    ) -> None:
        self.display_cache = display_cache
        self.config = JointSpatialConfig() if config is None else config
        self.diagnostics_dir = diagnostics_dir
        self.diagnostics = JointSpatialDiagnostics()

    def __call__(
        self,
        samples: Sequence[core.Sample],
        profile: core.Profile,
        filter_window: int,
        decimal_places_override: int | None = None,
        minimum_confidence: float = 0.0,
        decimal_switch_penalty: float = 4.0,
        decimal_sequence_observer: core.DecimalSequenceObserver | None = None,
    ) -> list[core.DecodedSample]:
        del filter_window
        del minimum_confidence
        del decimal_switch_penalty
        if profile.decoder != "rds30_margin" or len(profile.digit_boxes) != 4:
            raise RuntimeError("The joint-spatial experiment requires a four-digit margin-decoder profile.")
        decimal_places = profile.decimal_places if decimal_places_override is None else decimal_places_override
        if decimal_sequence_observer is not None:
            decimal_sequence_observer([decimal_places] * len(samples))

        patches = _digit_patches_from_cache(
            self.display_cache,
            profile,
            len(samples),
        )
        masks, rings = make_local_masks(profile)
        states = _all_states(self.config)
        state_offsets = [geometry_offsets(state, len(profile.digit_boxes)) for state in states]

        offsets_by_position: list[list[tuple[int, int]]] = []
        offset_indices: list[dict[tuple[int, int], int]] = []
        for position in range(len(profile.digit_boxes)):
            values = sorted({offsets[position] for offsets in state_offsets})
            offsets_by_position.append(values)
            offset_indices.append({value: index for index, value in enumerate(values)})

        patterns = profile.digit_patterns or core.RDS30_DIGIT_PATTERNS
        glyph_patterns = [(digit, pattern) for digit, pattern in sorted(patterns.items())]
        blank_pattern = (0,) * 7
        frame_count = len(samples)
        position_scores: list[np.ndarray] = []
        position_digits: list[np.ndarray] = []
        position_margins: list[np.ndarray] = []
        position_entropy: list[np.ndarray] = []
        position_agreement: list[np.ndarray] = []
        position_spatial: list[np.ndarray] = []

        for position, position_offsets in enumerate(offsets_by_position):
            score_table = np.empty((frame_count, len(position_offsets)), dtype=np.float32)
            digit_table = np.empty((frame_count, len(position_offsets)), dtype=np.int8)
            margin_table = np.empty((frame_count, len(position_offsets)), dtype=np.float32)
            entropy_table = np.empty((frame_count, len(position_offsets)), dtype=np.float32)
            agreement_table = np.empty((frame_count, len(position_offsets)), dtype=np.float32)
            spatial_table = np.empty((frame_count, len(position_offsets)), dtype=np.float32)
            legal_patterns = ([(-1, blank_pattern)] + glyph_patterns) if position == 0 else glyph_patterns
            global_background = np.percentile(patches[position], 90.0, axis=(1, 2))
            for offset_index, (dx, dy) in enumerate(position_offsets):
                shifted_masks = [translate_mask(mask, dx, dy) for mask in masks]
                shifted_rings = [translate_mask(ring, dx, dy) for ring in rings]
                evidence = np.stack(
                    [
                        _segment_evidence_series(
                            patches[position], mask, ring, name, global_background
                        )
                        for name, mask, ring in zip(
                            core.SEGMENT_ORDER, shifted_masks, shifted_rings
                        )
                    ],
                    axis=1,
                )
                evidence = _temporal_median_evidence(
                    evidence,
                    self.config.temporal_window,
                )
                probabilities = np.column_stack(
                    [
                        _probability_series(
                            evidence[:, segment_index],
                            self.config.horizontal_model
                            if name in HORIZONTAL_SEGMENTS
                            else self.config.vertical_model,
                        )
                        for segment_index, name in enumerate(core.SEGMENT_ORDER)
                    ]
                )
                clipped = np.clip(probabilities, 1.0e-6, 1.0 - 1.0e-6)
                glyph_score_matrix = np.column_stack(
                    [
                        np.sum(
                            np.where(
                                np.asarray(pattern, dtype=bool)[None, :],
                                np.log(clipped),
                                np.log1p(-clipped),
                            ),
                            axis=1,
                        )
                        for _, pattern in legal_patterns
                    ]
                )
                order = np.argsort(glyph_score_matrix, axis=1)
                row = np.arange(frame_count)
                best_index = order[:, -1]
                second_index = order[:, -2]
                best_score = glyph_score_matrix[row, best_index]
                score_table[:, offset_index] = best_score
                digit_table[:, offset_index] = np.asarray(
                    [legal_patterns[index][0] for index in best_index], dtype=np.int8
                )
                margin_table[:, offset_index] = (
                    best_score - glyph_score_matrix[row, second_index]
                )
                entropy_table[:, offset_index] = np.mean(
                    -clipped * np.log(clipped) - (1.0 - clipped) * np.log1p(-clipped),
                    axis=1,
                )
                agreement_table[:, offset_index] = np.mean(
                    np.exp(-np.abs(evidence[:, :, 0] - evidence[:, :, 1]) / 20.0),
                    axis=1,
                )
                spatial_table[:, offset_index] = np.mean(
                    np.maximum(
                        0.0,
                        evidence[:, :, 2]
                        - 0.25 * np.maximum(0.0, evidence[:, :, 3]),
                    ),
                    axis=1,
                )
            position_scores.append(score_table)
            position_digits.append(digit_table)
            position_margins.append(margin_table)
            position_entropy.append(entropy_table)
            position_agreement.append(agreement_table)
            position_spatial.append(spatial_table)

        state_offset_index = np.asarray(
            [
                [offset_indices[position][offsets[position]] for position in range(4)]
                for offsets in state_offsets
            ],
            dtype=np.int16,
        )
        emissions = np.zeros((frame_count, len(states)), dtype=np.float32)
        for position in range(4):
            emissions += position_scores[position][:, state_offset_index[:, position]]
        emissions -= np.asarray(
            [self.config.geometry_boundary_penalty if _boundary(state, self.config) else 0.0 for state in states],
            dtype=np.float32,
        )[None, :]

        keep = min(self.config.coarse_candidate_count, len(states))
        top = np.argpartition(emissions, -keep, axis=1)[:, -keep:]
        candidates: list[np.ndarray] = []
        for frame_index in range(frame_count):
            nearby = [top[frame_index]]
            if frame_index:
                nearby.append(top[frame_index - 1])
            if frame_index + 1 < frame_count:
                nearby.append(top[frame_index + 1])
            candidates.append(np.unique(np.concatenate(nearby)))

        costs: list[np.ndarray] = []
        back: list[np.ndarray] = []
        costs.append(-emissions[0, candidates[0]].astype(float))
        back.append(np.full(len(candidates[0]), -1, dtype=np.int16))
        for frame_index in range(1, frame_count):
            current_indices = candidates[frame_index]
            previous_indices = candidates[frame_index - 1]
            current_cost = np.empty(len(current_indices), dtype=float)
            current_back = np.empty(len(current_indices), dtype=np.int16)
            for local_index, state_index in enumerate(current_indices):
                transition = np.asarray(
                    [
                        temporal_geometry_penalty(states[previous], states[state_index], self.config)
                        for previous in previous_indices
                    ]
                )
                alternatives = costs[-1] + transition
                best_previous = int(np.argmin(alternatives))
                current_cost[local_index] = alternatives[best_previous] - float(emissions[frame_index, state_index])
                current_back[local_index] = best_previous
            costs.append(current_cost)
            back.append(current_back)

        path = np.empty(frame_count, dtype=np.int32)
        local = int(np.argmin(costs[-1]))
        for frame_index in range(frame_count - 1, -1, -1):
            path[frame_index] = int(candidates[frame_index][local])
            if frame_index:
                local = int(back[frame_index][local])

        decoded: list[core.DecodedSample] = []
        self.diagnostics.frames.clear()
        for frame_index, state_index in enumerate(path):
            state = states[int(state_index)]
            offsets = state_offsets[int(state_index)]
            digits: list[int] = []
            margins: list[float] = []
            entropies: list[float] = []
            agreements: list[float] = []
            spatial_values: list[float] = []
            for position in range(4):
                offset_index = state_offset_index[state_index, position]
                digits.append(int(position_digits[position][frame_index, offset_index]))
                margins.append(float(position_margins[position][frame_index, offset_index]))
                entropies.append(float(position_entropy[position][frame_index, offset_index]))
                agreements.append(float(position_agreement[position][frame_index, offset_index]))
                spatial_values.append(float(position_spatial[position][frame_index, offset_index]))

            order = np.argsort(emissions[frame_index])[::-1]
            geometry_second = next(
                (
                    int(candidate)
                    for candidate in order[1:]
                    if _state_distance(state, states[int(candidate)]) >= 3.0
                ),
                int(order[1]),
            )
            geometry_gap = float(emissions[frame_index, state_index] - emissions[frame_index, geometry_second])
            previous_state = state if frame_index == 0 else states[int(path[frame_index - 1])]
            temporal_change = _state_distance(previous_state, state)
            boundary = _boundary(state, self.config)
            worst_margin = min(margins)
            mean_entropy = float(np.mean(entropies))
            mean_agreement = float(np.mean(agreements))
            mean_spatial = float(np.mean(spatial_values))
            confidence_logit = (
                -1.2
                + 0.55 * worst_margin
                - 1.8 * mean_entropy
                + 1.2 * mean_agreement
                + 0.025 * mean_spatial
                + 0.35 * geometry_gap
                - 0.12 * temporal_change
                - (0.75 if boundary else 0.0)
            )
            confidence = 1.0 / (1.0 + math.exp(-float(np.clip(confidence_logit, -40.0, 40.0))))
            accepted = (
                samples[frame_index].display_found
                and confidence_accepts(confidence, self.config.rejection_threshold)
            )

            value_digits = digits[1:] if digits and digits[0] == -1 else digits
            value: float | None = None
            if accepted and value_digits and all(digit >= 0 for digit in value_digits):
                integer = 0
                for digit in value_digits:
                    integer = 10 * integer + digit
                value = integer / (10 ** decimal_places)
            decoded.append(core.DecodedSample(samples[frame_index].time_s, value, confidence))
            self.diagnostics.frames.append(
                JointSpatialFrameDiagnostic(
                    frame_index=frame_index,
                    time_s=samples[frame_index].time_s,
                    state=state,
                    offsets=offsets,
                    digits=tuple(digits),
                    digit_margins=tuple(margins),
                    segment_entropy=mean_entropy,
                    evidence_agreement=mean_agreement,
                    spatial_quality=mean_spatial,
                    geometry_gap=geometry_gap,
                    temporal_change=temporal_change,
                    boundary=boundary,
                    confidence=confidence,
                    accepted=accepted,
                )
            )

        if self.diagnostics_dir is not None:
            self._write_diagnostics(profile, patches)
        return decoded

    def _write_diagnostics(self, profile: core.Profile, patches: list[np.ndarray]) -> None:
        output_dir = self.diagnostics_dir
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        geometry_path = output_dir / "geometry.csv"
        with geometry_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "frame_index", "time_s", "tx", "ty", "kx", "ky",
                    "dx1", "dy1", "dx2", "dy2", "dx3", "dy3", "dx4", "dy4",
                    "digit1", "digit2", "digit3", "digit4",
                    "margin1", "margin2", "margin3", "margin4",
                    "segment_entropy", "evidence_agreement", "spatial_quality",
                    "geometry_gap", "temporal_change", "boundary", "confidence", "accepted",
                )
            )
            for item in self.diagnostics.frames:
                writer.writerow(
                    (
                        item.frame_index, f"{item.time_s:.6f}",
                        item.state.tx, item.state.ty, item.state.kx, item.state.ky,
                        *(coordinate for offset in item.offsets for coordinate in offset),
                        *item.digits,
                        *(f"{value:.8f}" for value in item.digit_margins),
                        f"{item.segment_entropy:.8f}", f"{item.evidence_agreement:.8f}",
                        f"{item.spatial_quality:.8f}", f"{item.geometry_gap:.8f}",
                        f"{item.temporal_change:.8f}", int(item.boundary),
                        f"{item.confidence:.8f}", int(item.accepted),
                    )
                )

        preview_dir = output_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        selected = {
            0,
            max(0, len(self.diagnostics.frames) // 2),
            max(0, len(self.diagnostics.frames) - 1),
        }
        ranked_confidence = sorted(
            self.diagnostics.frames,
            key=lambda item: item.confidence,
        )
        selected.update(item.frame_index for item in ranked_confidence[:8])
        selected.update(item.frame_index for item in ranked_confidence[-3:])
        for position in (2, 3):
            ranked_margin = sorted(
                self.diagnostics.frames,
                key=lambda item: item.digit_margins[position],
            )
            selected.update(item.frame_index for item in ranked_margin[:5])
        for position in range(4):
            observed_digits = {
                item.digits[position]
                for item in self.diagnostics.frames
                if item.accepted
            }
            for digit in observed_digits:
                examples = [
                    item
                    for item in self.diagnostics.frames
                    if item.accepted and item.digits[position] == digit
                ]
                selected.add(
                    min(
                        examples,
                        key=lambda item: item.digit_margins[position],
                    ).frame_index
                )
        masks, _ = make_local_masks(profile)
        for frame_index in sorted(selected):
            item = self.diagnostics.frames[frame_index]
            display = self.display_cache.get(frame_index)
            if display is None:
                continue
            canvas = display.copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            for position, ((x1, y1, x2, y2), (dx, dy), digit, margin) in enumerate(
                zip(profile.digit_boxes, item.offsets, item.digits, item.digit_margins), start=1
            ):
                scale_x = (x2 - x1) / 65.0
                scale_y = (y2 - y1) / 130.0
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (170, 170, 170), 1)
                for mask in masks:
                    for drawn_mask, color in (
                        (mask, (150, 150, 150)),
                        (translate_mask(mask, dx, dy), (0, 190, 0)),
                    ):
                        contours, _ = cv2.findContours(
                            drawn_mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        for contour in contours:
                            contour = contour.astype(np.float32)
                            contour[:, 0, 0] = x1 + contour[:, 0, 0] * scale_x
                            contour[:, 0, 1] = y1 + contour[:, 0, 1] * scale_y
                            cv2.polylines(
                                canvas,
                                [np.rint(contour).astype(np.int32)],
                                True,
                                color,
                                1,
                            )
                cv2.putText(
                    canvas, f"p{position}={digit} m={margin:.2f}", (x1, min(canvas.shape[0]-5, y1+18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 220), 1, cv2.LINE_AA,
                )
            status = "ACCEPT" if item.accepted else "REJECT"
            cv2.putText(
                canvas,
                f"{status} conf={item.confidence:.3f} geom=({item.state.tx},{item.state.ty},{item.state.kx},{item.state.ky})",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 160, 0) if item.accepted else (0, 0, 220), 1, cv2.LINE_AA,
            )
            enlarged = cv2.resize(canvas, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(preview_dir / f"frame_{frame_index:05d}.png"), enlarged)


def make_joint_spatial_decoder(
    display_cache: dict[int, np.ndarray],
    rejection_threshold: float | None = None,
    diagnostics_dir: Path | None = None,
) -> JointSpatialDecoder:
    config = JointSpatialConfig()
    if rejection_threshold is not None:
        from dataclasses import replace

        config = replace(config, rejection_threshold=rejection_threshold)
    return JointSpatialDecoder(display_cache, config=config, diagnostics_dir=diagnostics_dir)
