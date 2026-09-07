#!/usr/bin/env python3
"""
Four-corner tracked rectification for hand-held dosimeter videos.

Required files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

The four physical bezel corners are tracked frame by frame.  The
perspective transform is then determined directly from these points.

This version pads the tracking image around all edges, so corner
templates may safely lie very close to the original image boundary.

Example:

    python3 dosimeter_get_values_quadtrack.py IMG_1151edited.mov out_quad.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad 0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493 \
        --grid 0.302231,0.438202,0.726166,0.786517 \
        --contrast auto \
        --raw-output raw_quad.csv \
        --debug-dir debug_quad
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


try:
    import dosimeter_get_values as core
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values.py."
    ) from exc


try:
    import dosimeter_get_values_roi as roi_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_roi.py."
    ) from exc


try:
    import dosimeter_get_values_rectified as rect_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_rectified.py."
    ) from exc


try:
    import dosimeter_get_values_fixedgrid as fixed_app
except ImportError as exc:
    raise SystemExit(
        "Could not import dosimeter_get_values_fixedgrid.py."
    ) from exc


BASE_ROI_FINDER = (
    roi_app.find_display_crop_roi
)


# ======================================================================
# Wrapper arguments
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--track-time",
        type=float,
        default=None,
        help=(
            "reference frame used for corner templates; "
            "default is middle of video"
        ),
    )

    parser.add_argument(
        "--corner-patch",
        type=int,
        default=21,
        help=(
            "half-size of each corner template in pixels "
            "(default: 21 -> 43x43 template)"
        ),
    )

    parser.add_argument(
        "--corner-search",
        type=int,
        default=55,
        help=(
            "corner search radius in normalized-display pixels "
            "(default: 55)"
        ),
    )

    parser.add_argument(
        "--corner-min-score",
        type=float,
        default=0.28,
        help=(
            "minimum normalized template correlation "
            "(default: 0.28)"
        ),
    )

    parser.add_argument(
        "--corner-max-step",
        type=float,
        default=55.0,
        help=(
            "maximum allowed single-frame corner movement "
            "(default: 55 pixels)"
        ),
    )

    parser.add_argument(
        "--quad-max-hold",
        type=int,
        default=3,
        help=(
            "number of failed frames for which previous quad is reused "
            "(default: 3)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Tracking image
# ======================================================================


_TRACK_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def tracking_image(
    image: np.ndarray,
) -> np.ndarray:

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image

    gray = _TRACK_CLAHE.apply(
        gray
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    maximum = float(
        np.max(
            magnitude
        )
    )

    if maximum > 0:

        magnitude = (
            magnitude
            / maximum
        )

    return magnitude.astype(
        np.float32
    )


# ======================================================================
# Padding
# ======================================================================


def padded_tracking_image(
    image: np.ndarray,
    padding: int,
) -> np.ndarray:
    """
    Pad tracking image so corner templates may extend outside the
    original crop.

    BORDER_REPLICATE is intentionally used rather than zeros, because
    an artificial black border would itself become a strong trackable edge.
    """

    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REPLICATE,
    )


# ======================================================================
# Patch extraction from padded coordinates
# ======================================================================


def extract_patch_padded(
    padded_image: np.ndarray,
    original_center: np.ndarray,
    radius: int,
    padding: int,
) -> np.ndarray | None:

    cx = int(
        round(
            float(
                original_center[0]
            )
        )
    ) + padding

    cy = int(
        round(
            float(
                original_center[1]
            )
        )
    ) + padding

    x1 = (
        cx - radius
    )

    y1 = (
        cy - radius
    )

    x2 = (
        cx + radius + 1
    )

    y2 = (
        cy + radius + 1
    )

    patch = padded_image[
        y1:y2,
        x1:x2,
    ]

    expected = (
        2 * radius + 1
    )

    if patch.shape != (
        expected,
        expected,
    ):

        return None

    return patch.copy()


# ======================================================================
# Quad geometry
# ======================================================================


def polygon_area(
    points: np.ndarray,
) -> float:

    return abs(
        float(
            cv2.contourArea(
                points.astype(
                    np.float32
                )
            )
        )
    )


def quad_is_convex(
    points: np.ndarray,
) -> bool:

    integer_points = np.rint(
        points
    ).astype(
        np.int32
    )

    return bool(
        cv2.isContourConvex(
            integer_points
        )
    )


def edge_lengths(
    points: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        [
            np.linalg.norm(
                points[1]
                - points[0]
            ),
            np.linalg.norm(
                points[2]
                - points[1]
            ),
            np.linalg.norm(
                points[3]
                - points[2]
            ),
            np.linalg.norm(
                points[0]
                - points[3]
            ),
        ],
        dtype=float,
    )


def quad_is_sane(
    candidate: np.ndarray,
    reference: np.ndarray,
    previous: np.ndarray | None,
    max_step: float,
) -> bool:

    if candidate.shape != (
        4,
        2,
    ):

        return False

    if not np.all(
        np.isfinite(
            candidate
        )
    ):

        return False

    if not quad_is_convex(
        candidate
    ):

        return False

    reference_area = polygon_area(
        reference
    )

    candidate_area = polygon_area(
        candidate
    )

    if reference_area <= 0:

        return False

    area_ratio = (
        candidate_area
        / reference_area
    )

    if not (
        0.65
        <= area_ratio
        <= 1.40
    ):

        return False

    reference_edges = edge_lengths(
        reference
    )

    candidate_edges = edge_lengths(
        candidate
    )

    ratios = (
        candidate_edges
        / reference_edges
    )

    if np.any(
        ratios < 0.65
    ):

        return False

    if np.any(
        ratios > 1.40
    ):

        return False

    if not (
        0.55
        <= (
            candidate_edges[0]
            / candidate_edges[2]
        )
        <= 1.80
    ):

        return False

    if not (
        0.55
        <= (
            candidate_edges[1]
            / candidate_edges[3]
        )
        <= 1.80
    ):

        return False

    if previous is not None:

        movements = np.linalg.norm(
            candidate
            - previous,
            axis=1,
        )

        if float(
            np.max(
                movements
            )
        ) > max_step:

            return False

    return True


# ======================================================================
# Locate one corner in padded image
# ======================================================================


def locate_corner(
    padded_image: np.ndarray,
    template: np.ndarray,
    predicted_center: np.ndarray,
    patch_radius: int,
    search_radius: int,
    padding: int,
) -> tuple[
    np.ndarray | None,
    float,
]:

    # Convert original-image coordinates to padded coordinates.
    cx = int(
        round(
            float(
                predicted_center[0]
            )
        )
    ) + padding

    cy = int(
        round(
            float(
                predicted_center[1]
            )
        )
    ) + padding

    template_height, template_width = (
        template.shape[:2]
    )

    x1 = (
        cx
        - search_radius
        - patch_radius
    )

    y1 = (
        cy
        - search_radius
        - patch_radius
    )

    x2 = (
        cx
        + search_radius
        + patch_radius
        + 1
    )

    y2 = (
        cy
        + search_radius
        + patch_radius
        + 1
    )

    # Padding is deliberately chosen large enough, but keep this safe.
    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        padded_image.shape[1],
        x2,
    )

    y2 = min(
        padded_image.shape[0],
        y2,
    )

    search = padded_image[
        y1:y2,
        x1:x2,
    ]

    if (
        search.shape[0]
        < template_height
        or search.shape[1]
        < template_width
    ):

        return (
            None,
            float("-inf"),
        )

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED,
    )

    (
        _minimum_value,
        maximum_value,
        _minimum_location,
        maximum_location,
    ) = cv2.minMaxLoc(
        result
    )

    # Center in PADDED coordinates.
    padded_match_x = (
        x1
        + maximum_location[0]
        + patch_radius
    )

    padded_match_y = (
        y1
        + maximum_location[1]
        + patch_radius
    )

    # Back to ORIGINAL crop coordinates.
    match_x = (
        padded_match_x
        - padding
    )

    match_y = (
        padded_match_y
        - padding
    )

    point = np.asarray(
        [
            float(
                match_x
            ),
            float(
                match_y
            ),
        ],
        dtype=np.float32,
    )

    return (
        point,
        float(
            maximum_value
        ),
    )


# ======================================================================
# Four-corner tracker
# ======================================================================


class QuadTracker:

    def __init__(
        self,
        reference_display: np.ndarray,
        normalized_quad: np.ndarray,
        profile: core.Profile,
        patch_radius: int,
        search_radius: int,
        minimum_score: float,
        maximum_step: float,
        maximum_hold: int,
    ) -> None:

        self.reference_display = (
            reference_display.copy()
        )

        self.profile = (
            profile
        )

        self.patch_radius = (
            patch_radius
        )

        self.search_radius = (
            search_radius
        )

        self.minimum_score = (
            minimum_score
        )

        self.maximum_step = (
            maximum_step
        )

        self.maximum_hold = (
            maximum_hold
        )

        # Enough padding for both template and search area.
        self.padding = (
            patch_radius
            + search_radius
            + 8
        )

        height, width = (
            reference_display.shape[:2]
        )

        self.reference_points = (
            np.asarray(
                normalized_quad,
                dtype=np.float32,
            ).copy()
        )

        self.reference_points[
            :,
            0
        ] *= float(
            width - 1
        )

        self.reference_points[
            :,
            1
        ] *= float(
            height - 1
        )

        prepared_reference = (
            tracking_image(
                reference_display
            )
        )

        padded_reference = (
            padded_tracking_image(
                prepared_reference,
                self.padding,
            )
        )

        templates: list[
            np.ndarray
        ] = []

        for point in (
            self.reference_points
        ):

            patch = (
                extract_patch_padded(
                    padded_reference,
                    point,
                    patch_radius,
                    self.padding,
                )
            )

            if patch is None:

                raise RuntimeError(
                    "Could not create a corner template."
                )

            templates.append(
                patch
            )

        self.reference_templates = (
            templates
        )

        self.last_points = (
            self.reference_points.copy()
        )

        self.hold_count = 0

        self.total = 0
        self.accepted = 0
        self.held = 0
        self.reference_fallback = 0
        self.rejected_geometry = 0
        self.rejected_score = 0

        self.score_sum = 0.0
        self.score_count = 0


    def reset(
        self,
    ) -> None:

        self.last_points = (
            self.reference_points.copy()
        )

        self.hold_count = 0


    def track(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        str,
        list[float],
    ]:

        self.total += 1

        prepared = tracking_image(
            display
        )

        padded = padded_tracking_image(
            prepared,
            self.padding,
        )

        candidate_points = []
        scores = []

        # ------------------------------------------------------
        # Locate all four corners.
        # ------------------------------------------------------

        for index in range(4):

            point, score = (
                locate_corner(
                    padded,
                    self.reference_templates[
                        index
                    ],
                    self.last_points[
                        index
                    ],
                    self.patch_radius,
                    self.search_radius,
                    self.padding,
                )
            )

            candidate_points.append(
                point
            )

            scores.append(
                score
            )

        # ------------------------------------------------------
        # Correlation check
        # ------------------------------------------------------

        if (
            any(
                point is None
                for point in candidate_points
            )
            or min(
                scores
            ) < self.minimum_score
        ):

            self.rejected_score += 1

            return self._fallback(
                scores
            )

        candidate = np.asarray(
            candidate_points,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Geometry check
        # ------------------------------------------------------

        if not quad_is_sane(
            candidate,
            self.reference_points,
            self.last_points,
            self.maximum_step,
        ):

            self.rejected_geometry += 1

            return self._fallback(
                scores
            )

        # ------------------------------------------------------
        # Mild temporal smoothing
        # ------------------------------------------------------

        smoothed = (
            0.82 * candidate
            + 0.18 * self.last_points
        )

        if not quad_is_sane(
            smoothed,
            self.reference_points,
            self.last_points,
            self.maximum_step,
        ):

            self.rejected_geometry += 1

            return self._fallback(
                scores
            )

        self.last_points = (
            smoothed.astype(
                np.float32
            )
        )

        self.hold_count = 0

        self.accepted += 1

        self.score_sum += float(
            np.mean(
                scores
            )
        )

        self.score_count += 1

        return (
            self.last_points.copy(),
            "tracked",
            scores,
        )


    def _fallback(
        self,
        scores: list[float],
    ) -> tuple[
        np.ndarray,
        str,
        list[float],
    ]:

        if (
            self.hold_count
            < self.maximum_hold
        ):

            self.hold_count += 1

            self.held += 1

            return (
                self.last_points.copy(),
                "held",
                scores,
            )

        self.reference_fallback += 1

        self.last_points = (
            self.reference_points.copy()
        )

        self.hold_count = 0

        return (
            self.last_points.copy(),
            "reference",
            scores,
        )


    def rectify(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        str,
        list[float],
    ]:

        points, method, scores = (
            self.track(
                display
            )
        )

        destination = np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    self.profile.canonical_width
                    - 1.0,
                    0.0,
                ],
                [
                    self.profile.canonical_width
                    - 1.0,
                    self.profile.canonical_height
                    - 1.0,
                ],
                [
                    0.0,
                    self.profile.canonical_height
                    - 1.0,
                ],
            ],
            dtype=np.float32,
        )

        matrix = (
            cv2.getPerspectiveTransform(
                points.astype(
                    np.float32
                ),
                destination,
            )
        )

        rectified = (
            cv2.warpPerspective(
                display,
                matrix,
                (
                    self.profile.canonical_width,
                    self.profile.canonical_height,
                ),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )

        return (
            rectified,
            method,
            scores,
        )


# ======================================================================
# Main
# ======================================================================


def main() -> int:

    wrapper_args, remaining = (
        parse_wrapper_args(
            sys.argv[1:]
        )
    )

    if (
        wrapper_args.corner_patch < 8
    ):

        print(
            "Error: --corner-patch is too small.",
            file=sys.stderr,
        )

        return 1

    if (
        wrapper_args.corner_search
        <= wrapper_args.corner_patch
    ):

        print(
            "Error: --corner-search must be larger than --corner-patch.",
            file=sys.stderr,
        )

        return 1

    saved_argv = (
        sys.argv
    )

    # ----------------------------------------------------------
    # Parse fixed-grid arguments
    # ----------------------------------------------------------

    try:

        sys.argv = [
            saved_argv[0],
            *remaining,
        ]

        (
            fixed_extra,
            roi_remaining,
        ) = fixed_app.parse_extra_args()

        args = (
            fixed_app.parse_roi_args(
                roi_remaining
            )
        )

    finally:

        sys.argv = (
            saved_argv
        )

    if args.roi is None:

        print(
            "Error: --roi is required.",
            file=sys.stderr,
        )

        return 1

    try:

        profile = (
            core.PROFILES[
                args.profile
            ]
        )

        info = (
            core.probe_video(
                args.video
            )
        )

        sample_fps = (
            args.sample_fps
            if args.sample_fps is not None
            else profile.default_sample_fps
        )

        track_time = (
            wrapper_args.track_time
            if wrapper_args.track_time
            is not None
            else 0.5 * info.duration
        )

        track_time = max(
            0.0,
            min(
                track_time,
                info.duration,
            ),
        )

        # ------------------------------------------------------
        # Reference display
        # ----------------------------------------------------------

        print(
            (
                "Acquiring corner-tracking reference "
                f"near t={track_time:.3f} s..."
            ),
            file=sys.stderr,
        )

        (
            reference_display,
            actual_time,
        ) = (
            rect_app.find_reference_display(
                args.video,
                info,
                profile,
                args.roi,
                args.processing_width,
                sample_fps,
                track_time,
            )
        )

        print(
            (
                "Corner reference acquired at "
                f"t={actual_time:.3f} s."
            ),
            file=sys.stderr,
        )

        tracker = QuadTracker(
            reference_display=(
                reference_display
            ),
            normalized_quad=(
                fixed_extra.quad
            ),
            profile=(
                profile
            ),
            patch_radius=(
                wrapper_args.corner_patch
            ),
            search_radius=(
                wrapper_args.corner_search
            ),
            minimum_score=(
                wrapper_args.corner_min_score
            ),
            maximum_step=(
                wrapper_args.corner_max_step
            ),
            maximum_hold=(
                wrapper_args.quad_max_hold
            ),
        )

        print(
            (
                "Corner templates created successfully "
                f"(patch={2 * wrapper_args.corner_patch + 1}x"
                f"{2 * wrapper_args.corner_patch + 1}, "
                f"padding={tracker.padding})."
            ),
            file=sys.stderr,
        )

        # ------------------------------------------------------
        # Replace fixedgrid rectification
        # ----------------------------------------------------------

        original_factory = (
            fixed_app.make_rectified_finder
        )

        def make_quadtracked_finder(
            quad: np.ndarray,
        ):

            del quad

            def finder(
                frame,
                in_profile,
                previous_box,
                roi,
            ):

                # New sequential pass, including debug pass.
                if previous_box is None:

                    tracker.reset()

                display, box = (
                    BASE_ROI_FINDER(
                        frame,
                        in_profile,
                        previous_box,
                        roi,
                    )
                )

                if display is None:

                    return (
                        None,
                        box,
                    )

                (
                    rectified,
                    _method,
                    _scores,
                ) = tracker.rectify(
                    display
                )

                return (
                    rectified,
                    box,
                )

            return finder

        fixed_app.make_rectified_finder = (
            make_quadtracked_finder
        )

        # ------------------------------------------------------
        # Run fixed-grid decoder
        # ----------------------------------------------------------

        try:

            sys.argv = [
                saved_argv[0],
                *remaining,
            ]

            result = (
                fixed_app.main()
            )

        finally:

            sys.argv = (
                saved_argv
            )

            fixed_app.make_rectified_finder = (
                original_factory
            )

        # ------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        print()

        print(
            "Four-corner tracking diagnostics:"
        )

        print(
            f"  processed         : {tracker.total}"
        )

        print(
            f"  accepted          : {tracker.accepted}"
        )

        print(
            f"  held previous quad: {tracker.held}"
        )

        print(
            f"  reference fallback: {tracker.reference_fallback}"
        )

        print(
            f"  rejected score    : {tracker.rejected_score}"
        )

        print(
            f"  rejected geometry : {tracker.rejected_geometry}"
        )

        if (
            tracker.score_count > 0
        ):

            print(
                (
                    "  mean corner score : "
                    f"{tracker.score_sum / tracker.score_count:.4f}"
                )
            )

        return result

    except (
        RuntimeError,
        ValueError,
        OSError,
        cv2.error,
    ) as exc:

        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )