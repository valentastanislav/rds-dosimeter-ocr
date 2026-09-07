#!/usr/bin/env python3
"""
Stateful stabilization front-end for hand-held RDS-200 videos.

Required files in the same directory:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

Processing:

    video
      -> ROI tracking
      -> perspective rectification
      -> stateful ECC stabilization
      -> fixed digit grid
      -> seven-segment decoding

Important difference from the previous version:

    ECC does NOT start from identity for every frame.

The previous successful transformation is used as the initial estimate
for the following frame.  If registration temporarily fails, the last
good transformation is retained instead of returning an unregistered
frame.

Example:

    python3 dosimeter_get_values_stabilized.py IMG_1151edited.mov out.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad 0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493 \
        --grid 0.237323,0.485955,0.687627,0.797753 \
        --contrast auto \
        --raw-output raw.csv \
        --debug-dir debug
"""

from __future__ import annotations

import argparse
import math
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


# Keep the normal rectified finder factory.
NORMAL_RECTIFIED_FINDER_FACTORY = (
    fixed_app.make_rectified_finder
)


# ======================================================================
# Command-line option specific to this wrapper
# ======================================================================


def parse_wrapper_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--stabilize-time",
        type=float,
        default=None,
        help=(
            "time used as geometrical reference; "
            "default is middle of video"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Image preparation for ECC only
#
# This does not modify the image sent to the digit decoder.
# ======================================================================


_ECC_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def prepare_ecc_image(
    image: np.ndarray,
) -> np.ndarray:

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    enhanced = _ECC_CLAHE.apply(
        gray
    )

    enhanced = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    return (
        enhanced.astype(np.float32)
        / 255.0
    )


# ======================================================================
# Registration mask
# ======================================================================


def make_registration_mask(
    profile: core.Profile,
) -> np.ndarray:
    """
    Use static graphics for registration and avoid changing digits.

    White = used for ECC
    Black = ignored
    """

    height = (
        profile.canonical_height
    )

    width = (
        profile.canonical_width
    )

    mask = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    # ----------------------------------------------------------
    # Ignore numeric digits + surrounding area
    # ----------------------------------------------------------

    digit_x1 = min(
        box[0]
        for box in profile.digit_boxes
    )

    digit_y1 = min(
        box[1]
        for box in profile.digit_boxes
    )

    digit_x2 = max(
        box[2]
        for box in profile.digit_boxes
    )

    digit_y2 = max(
        box[3]
        for box in profile.digit_boxes
    )

    margin_x = 30
    margin_y = 28

    cv2.rectangle(
        mask,
        (
            max(
                0,
                digit_x1 - margin_x,
            ),
            max(
                0,
                digit_y1 - margin_y,
            ),
        ),
        (
            min(
                width - 1,
                digit_x2 + margin_x,
            ),
            min(
                height - 1,
                digit_y2 + margin_y,
            ),
        ),
        0,
        -1,
    )

    # ----------------------------------------------------------
    # Ignore changing bar graph on left
    # ----------------------------------------------------------

    cv2.rectangle(
        mask,
        (
            0,
            int(
                round(
                    0.34 * height
                )
            ),
        ),
        (
            int(
                round(
                    0.24 * width
                )
            ),
            height - 1,
        ),
        0,
        -1,
    )

    # ----------------------------------------------------------
    # Slight erosion prevents border pixels from dominating.
    # ----------------------------------------------------------

    mask = cv2.erode(
        mask,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    return mask


# ======================================================================
# Transformation checks
# ======================================================================


def identity_homography() -> np.ndarray:

    return np.eye(
        3,
        dtype=np.float32,
    )


def affine_to_homography(
    affine: np.ndarray,
) -> np.ndarray:

    matrix = identity_homography()

    matrix[
        0:2,
        :
    ] = affine

    return matrix


def homography_to_affine(
    matrix: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        matrix[0:2, :],
        dtype=np.float32,
    ).copy()


def transformation_is_sane(
    matrix: np.ndarray,
    width: int,
    height: int,
) -> bool:

    if matrix.shape != (
        3,
        3,
    ):
        return False

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        return False

    corners = np.asarray(
        [
            [0.0, 0.0],
            [
                width - 1.0,
                0.0,
            ],
            [
                width - 1.0,
                height - 1.0,
            ],
            [
                0.0,
                height - 1.0,
            ],
        ],
        dtype=np.float32,
    ).reshape(
        1,
        4,
        2,
    )

    try:
        transformed = (
            cv2.perspectiveTransform(
                corners,
                matrix.astype(
                    np.float32
                ),
            )[0]
        )

    except cv2.error:
        return False

    if not np.all(
        np.isfinite(
            transformed
        )
    ):
        return False

    top_width = float(
        np.linalg.norm(
            transformed[1]
            - transformed[0]
        )
    )

    bottom_width = float(
        np.linalg.norm(
            transformed[2]
            - transformed[3]
        )
    )

    left_height = float(
        np.linalg.norm(
            transformed[3]
            - transformed[0]
        )
    )

    right_height = float(
        np.linalg.norm(
            transformed[2]
            - transformed[1]
        )
    )

    for scale in (
        top_width / width,
        bottom_width / width,
        left_height / height,
        right_height / height,
    ):
        if not (
            0.75
            <= scale
            <= 1.30
        ):
            return False

    center = np.asarray(
        [
            width / 2.0,
            height / 2.0,
        ],
        dtype=float,
    )

    transformed_center = (
        np.mean(
            transformed,
            axis=0,
        )
    )

    center_shift = float(
        np.linalg.norm(
            transformed_center
            - center
        )
    )

    diagonal = math.hypot(
        width,
        height,
    )

    if (
        center_shift
        > 0.22 * diagonal
    ):
        return False

    return True


# ======================================================================
# Stateful ECC tracker
# ======================================================================


class ECCTracker:

    def __init__(
        self,
        reference: np.ndarray,
    ) -> None:

        self.reference = (
            reference.copy()
        )

        self.template = (
            prepare_ecc_image(
                reference
            )
        )

        self.height, self.width = (
            self.template.shape
        )

        self.mask: (
            np.ndarray | None
        ) = None

        self.last_warp = (
            identity_homography()
        )

        self.have_warp = False

        self.consecutive_holds = 0

        self.total = 0

        self.tracked = 0

        self.reacquired = 0

        self.affine = 0

        self.homography = 0

        self.hold = 0

        self.raw = 0

        self.cc_sum = 0.0

        self.cc_count = 0


    def reset(
        self,
    ) -> None:
        """
        Called at the beginning of every sequential video pass.

        This matters because --debug-dir causes a second pass through
        the video from t=0.
        """

        self.last_warp = (
            identity_homography()
        )

        self.have_warp = False

        self.consecutive_holds = 0


    def set_profile(
        self,
        profile: core.Profile,
    ) -> None:

        self.mask = (
            make_registration_mask(
                profile
            )
        )


    def record_cc(
        self,
        cc: float,
    ) -> None:

        if np.isfinite(
            cc
        ):
            self.cc_sum += float(
                cc
            )

            self.cc_count += 1


    def try_affine(
        self,
        current: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[
        bool,
        float,
        np.ndarray,
    ]:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            80,
            1e-6,
        )

        try:
            cc, affine = (
                cv2.findTransformECC(
                    self.template,
                    current,
                    initial.copy(),
                    cv2.MOTION_AFFINE,
                    criteria,
                    self.mask,
                    5,
                )
            )

        except cv2.error:
            return (
                False,
                float("-inf"),
                initial,
            )

        matrix = (
            affine_to_homography(
                affine
            )
        )

        good = (
            np.isfinite(cc)
            and cc >= 0.28
            and transformation_is_sane(
                matrix,
                self.width,
                self.height,
            )
        )

        return (
            bool(good),
            float(cc),
            affine,
        )


    def try_homography(
        self,
        current: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[
        bool,
        float,
        np.ndarray,
    ]:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6,
        )

        try:
            cc, matrix = (
                cv2.findTransformECC(
                    self.template,
                    current,
                    initial.copy(),
                    cv2.MOTION_HOMOGRAPHY,
                    criteria,
                    self.mask,
                    5,
                )
            )

        except cv2.error:
            return (
                False,
                float("-inf"),
                initial,
            )

        good = (
            np.isfinite(cc)
            and cc >= 0.30
            and transformation_is_sane(
                matrix,
                self.width,
                self.height,
            )
        )

        return (
            bool(good),
            float(cc),
            matrix,
        )


    def find_transform(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        str,
    ]:

        current = (
            prepare_ecc_image(
                display
            )
        )

        # ------------------------------------------------------
        # First attempt:
        # use PREVIOUS transform as initial estimate.
        # ------------------------------------------------------

        if self.have_warp:

            affine_initial = (
                homography_to_affine(
                    self.last_warp
                )
            )

        else:

            affine_initial = np.asarray(
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                ],
                dtype=np.float32,
            )

        (
            affine_ok,
            affine_cc,
            affine,
        ) = self.try_affine(
            current,
            affine_initial,
        )

        if affine_ok:

            affine_h = (
                affine_to_homography(
                    affine
                )
            )

            # --------------------------------------------------
            # Homography refinement starts from the good affine
            # solution rather than identity.
            # --------------------------------------------------

            (
                homography_ok,
                homography_cc,
                homography,
            ) = self.try_homography(
                current,
                affine_h,
            )

            # Homography is accepted only if it actually improves
            # the match enough to justify extra degrees of freedom.
            if (
                homography_ok
                and homography_cc
                >= affine_cc + 0.008
            ):

                method = (
                    "homography"
                )

                chosen = (
                    homography
                )

                cc = (
                    homography_cc
                )

            else:

                method = (
                    "affine"
                )

                chosen = (
                    affine_h
                )

                cc = (
                    affine_cc
                )

            if self.have_warp:
                self.tracked += 1
            else:
                self.reacquired += 1

            if method == "affine":
                self.affine += 1
            else:
                self.homography += 1

            self.last_warp = (
                chosen.copy()
            )

            self.have_warp = True

            self.consecutive_holds = 0

            self.record_cc(
                cc
            )

            return (
                chosen,
                float(cc),
                method,
            )

        # ------------------------------------------------------
        # Previous estimate failed.
        #
        # Try a clean acquisition from identity.
        # ------------------------------------------------------

        identity_affine = np.asarray(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                ],
            ],
            dtype=np.float32,
        )

        (
            reacquire_ok,
            reacquire_cc,
            reacquire_affine,
        ) = self.try_affine(
            current,
            identity_affine,
        )

        if reacquire_ok:

            reacquire_h = (
                affine_to_homography(
                    reacquire_affine
                )
            )

            (
                homography_ok,
                homography_cc,
                homography,
            ) = self.try_homography(
                current,
                reacquire_h,
            )

            if (
                homography_ok
                and homography_cc
                >= reacquire_cc + 0.008
            ):

                chosen = (
                    homography
                )

                cc = (
                    homography_cc
                )

                method = (
                    "homography"
                )

                self.homography += 1

            else:

                chosen = (
                    reacquire_h
                )

                cc = (
                    reacquire_cc
                )

                method = (
                    "affine"
                )

                self.affine += 1

            self.last_warp = (
                chosen.copy()
            )

            self.have_warp = True

            self.consecutive_holds = 0

            self.reacquired += 1

            self.record_cc(
                cc
            )

            return (
                chosen,
                float(cc),
                method,
            )

        # ------------------------------------------------------
        # ECC completely failed on this frame.
        #
        # KEEP THE LAST GOOD TRANSFORMATION.
        #
        # This is much safer for a hand-held video than suddenly
        # returning an unstabilized crop.
        # ------------------------------------------------------

        if self.have_warp:

            self.consecutive_holds += 1

            self.hold += 1

            return (
                self.last_warp.copy(),
                float("nan"),
                "hold",
            )

        # Nothing has ever been acquired.
        self.raw += 1

        return (
            identity_homography(),
            float("nan"),
            "raw",
        )


    def align(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        str,
    ]:

        self.total += 1

        (
            matrix,
            cc,
            method,
        ) = self.find_transform(
            display
        )

        stabilized = (
            cv2.warpPerspective(
                display,
                matrix,
                (
                    self.width,
                    self.height,
                ),
                flags=(
                    cv2.INTER_CUBIC
                    | cv2.WARP_INVERSE_MAP
                ),
                borderMode=cv2.BORDER_REPLICATE,
            )
        )

        return (
            stabilized,
            cc,
            method,
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

    saved_argv = (
        sys.argv
    )

    # ----------------------------------------------------------
    # Parse arguments expected by fixedgrid.
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

        base_profile = (
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
            else base_profile.default_sample_fps
        )

        if (
            wrapper_args.stabilize_time
            is not None
        ):

            reference_time = (
                wrapper_args.stabilize_time
            )

        else:

            reference_time = (
                0.5
                * info.duration
            )

        reference_time = max(
            0.0,
            min(
                reference_time,
                info.duration,
            ),
        )

        # ------------------------------------------------------
        # Get reference display.
        # ------------------------------------------------------

        print(
            (
                "Acquiring stabilization reference "
                f"near t={reference_time:.3f} s..."
            ),
            file=sys.stderr,
        )

        (
            reference_display,
            actual_reference_time,
        ) = rect_app.find_reference_display(
            args.video,
            info,
            base_profile,
            args.roi,
            args.processing_width,
            sample_fps,
            reference_time,
        )

        reference_rectified = (
            rect_app.rectify_display(
                reference_display,
                base_profile,
                fixed_extra.quad,
            )
        )

        print(
            (
                "Stabilization reference acquired at "
                f"t={actual_reference_time:.3f} s."
            ),
            file=sys.stderr,
        )

        # ------------------------------------------------------
        # Tracker
        # ------------------------------------------------------

        tracker = ECCTracker(
            reference_rectified
        )

        original_factory = (
            fixed_app.make_rectified_finder
        )

        def make_stateful_finder(
            quad: np.ndarray,
        ):

            normal_finder = (
                NORMAL_RECTIFIED_FINDER_FACTORY(
                    quad
                )
            )

            profile_id = None

            def finder(
                frame,
                in_profile,
                previous_box,
                roi,
            ):

                nonlocal profile_id

                # Start of a new sequential pass.
                #
                # The debug screenshot generator starts again with
                # previous_box=None.
                if previous_box is None:

                    tracker.reset()

                current_profile_id = (
                    id(
                        in_profile
                    )
                )

                if (
                    profile_id
                    != current_profile_id
                ):

                    tracker.set_profile(
                        in_profile
                    )

                    profile_id = (
                        current_profile_id
                    )

                (
                    rectified,
                    box,
                ) = normal_finder(
                    frame,
                    in_profile,
                    previous_box,
                    roi,
                )

                if rectified is None:

                    return (
                        None,
                        box,
                    )

                (
                    stabilized,
                    _cc,
                    _method,
                ) = tracker.align(
                    rectified
                )

                return (
                    stabilized,
                    box,
                )

            return finder

        fixed_app.make_rectified_finder = (
            make_stateful_finder
        )

        # ------------------------------------------------------
        # Run normal fixed-grid application.
        # ------------------------------------------------------

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
        # Statistics
        # ------------------------------------------------------

        print()

        print(
            "Stateful stabilization diagnostics:"
        )

        print(
            f"  processed       : {tracker.total}"
        )

        print(
            f"  tracked         : {tracker.tracked}"
        )

        print(
            f"  reacquired      : {tracker.reacquired}"
        )

        print(
            f"  affine          : {tracker.affine}"
        )

        print(
            f"  homography      : {tracker.homography}"
        )

        print(
            f"  held last warp  : {tracker.hold}"
        )

        print(
            f"  raw/unregistered: {tracker.raw}"
        )

        if (
            tracker.cc_count > 0
        ):

            mean_cc = (
                tracker.cc_sum
                / tracker.cc_count
            )

            print(
                f"  mean ECC        : {mean_cc:.4f}"
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