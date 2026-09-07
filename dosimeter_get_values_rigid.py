#!/usr/bin/env python3
"""
Safe rigid stabilization for hand-held RDS-200 videos.

Required files in the same directory:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

Pipeline:

    video
      -> ROI display tracking
      -> fixed perspective rectification (--quad)
      -> RIGID ECC registration
            translation + rotation ONLY
      -> fixed digit grid (--grid)
      -> seven-segment decoder

Unlike the previous experimental stabilizer:

    - NO affine shear
    - NO homography
    - NO perspective deformation
    - transform must vary smoothly from frame to frame
    - failed frames keep the previous transform only briefly

Example:

    python3 dosimeter_get_values_rigid.py IMG_1151edited.mov out_rigid.csv \
        --profile rds200 \
        --roi 0.277704,0.221833,0.735259,0.388583 \
        --quad 0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493 \
        --grid 0.237323,0.485955,0.687627,0.797753 \
        --contrast auto \
        --raw-output raw_rigid.csv \
        --debug-dir debug_rigid
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


NORMAL_RECTIFIED_FINDER_FACTORY = (
    fixed_app.make_rectified_finder
)


# ======================================================================
# Extra CLI arguments
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
            "reference time for rigid stabilization; "
            "default is the middle of the video"
        ),
    )

    parser.add_argument(
        "--max-rotation",
        type=float,
        default=12.0,
        help=(
            "maximum absolute residual rotation in degrees "
            "(default: 12)"
        ),
    )

    parser.add_argument(
        "--max-frame-rotation",
        type=float,
        default=4.0,
        help=(
            "maximum rotation change between adjacent frames "
            "(default: 4 degrees)"
        ),
    )

    parser.add_argument(
        "--max-frame-shift",
        type=float,
        default=25.0,
        help=(
            "maximum transform translation change between adjacent "
            "frames in canonical pixels (default: 25)"
        ),
    )

    parser.add_argument(
        "--max-hold",
        type=int,
        default=2,
        help=(
            "maximum number of failed frames for which the last good "
            "transform is reused (default: 2)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# ECC preprocessing
# ======================================================================


_ECC_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def prepare_ecc_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Image used ONLY for finding geometrical registration.

    The decoder itself still receives the normal grayscale image.
    """

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
    ECC should follow STATIC parts of the LCD, not the changing digits.

    White = used
    Black = ignored
    """

    height = profile.canonical_height
    width = profile.canonical_width

    mask = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    # ----------------------------------------------------------
    # Mask the changing numerical field
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

    cv2.rectangle(
        mask,
        (
            max(
                0,
                digit_x1 - 25,
            ),
            max(
                0,
                digit_y1 - 25,
            ),
        ),
        (
            min(
                width - 1,
                digit_x2 + 25,
            ),
            min(
                height - 1,
                digit_y2 + 25,
            ),
        ),
        0,
        -1,
    )

    # ----------------------------------------------------------
    # Mask left changing bar graph
    # ----------------------------------------------------------

    cv2.rectangle(
        mask,
        (
            0,
            int(
                round(
                    0.33 * height
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

    return mask


# ======================================================================
# Transform helpers
# ======================================================================


def identity_warp() -> np.ndarray:

    return np.asarray(
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


def warp_angle_deg(
    warp: np.ndarray,
) -> float:

    return math.degrees(
        math.atan2(
            float(warp[1, 0]),
            float(warp[0, 0]),
        )
    )


def warp_translation(
    warp: np.ndarray,
) -> tuple[float, float]:

    return (
        float(warp[0, 2]),
        float(warp[1, 2]),
    )


def angle_difference_deg(
    a: float,
    b: float,
) -> float:

    difference = (
        a - b
    )

    while difference > 180.0:
        difference -= 360.0

    while difference < -180.0:
        difference += 360.0

    return abs(
        difference
    )


# ======================================================================
# Rigid transformation sanity check
# ======================================================================


def rigid_warp_is_sane(
    warp: np.ndarray,
    previous: np.ndarray | None,
    max_rotation: float,
    max_frame_rotation: float,
    max_frame_shift: float,
) -> bool:

    if warp.shape != (
        2,
        3,
    ):

        return False

    if not np.all(
        np.isfinite(
            warp
        )
    ):

        return False

    angle = warp_angle_deg(
        warp
    )

    tx, ty = warp_translation(
        warp
    )

    # Residual geometry after bezel crop + perspective correction
    # should never require an enormous rotation.
    if abs(angle) > max_rotation:

        return False

    # Likewise very large absolute translations indicate ECC got lost.
    if (
        abs(tx) > 90.0
        or abs(ty) > 90.0
    ):

        return False

    # ECC MOTION_EUCLIDEAN should produce an orthogonal rotation matrix.
    # Check this explicitly anyway.
    a = float(
        warp[0, 0]
    )

    b = float(
        warp[0, 1]
    )

    c = float(
        warp[1, 0]
    )

    d = float(
        warp[1, 1]
    )

    row0_length = math.hypot(
        a,
        b,
    )

    row1_length = math.hypot(
        c,
        d,
    )

    row_dot = (
        a * c
        + b * d
    )

    if not (
        0.97 <= row0_length <= 1.03
        and 0.97 <= row1_length <= 1.03
        and abs(row_dot) <= 0.03
    ):

        return False

    if previous is None:

        return True

    previous_angle = (
        warp_angle_deg(
            previous
        )
    )

    previous_tx, previous_ty = (
        warp_translation(
            previous
        )
    )

    if (
        angle_difference_deg(
            angle,
            previous_angle,
        )
        > max_frame_rotation
    ):

        return False

    frame_shift = math.hypot(
        tx - previous_tx,
        ty - previous_ty,
    )

    if (
        frame_shift
        > max_frame_shift
    ):

        return False

    return True


# ======================================================================
# Rigid ECC tracker
# ======================================================================


class RigidTracker:

    def __init__(
        self,
        reference: np.ndarray,
        max_rotation: float,
        max_frame_rotation: float,
        max_frame_shift: float,
        max_hold: int,
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

        self.max_rotation = (
            max_rotation
        )

        self.max_frame_rotation = (
            max_frame_rotation
        )

        self.max_frame_shift = (
            max_frame_shift
        )

        self.max_hold = (
            max_hold
        )

        self.last_good: (
            np.ndarray | None
        ) = None

        self.failed_frames = 0

        self.total = 0
        self.tracked = 0
        self.reacquired = 0
        self.held = 0
        self.identity = 0
        self.rejected = 0

        self.cc_sum = 0.0
        self.cc_count = 0


    def reset(
        self,
    ) -> None:

        self.last_good = None
        self.failed_frames = 0


    def set_profile(
        self,
        profile: core.Profile,
    ) -> None:

        self.mask = (
            make_registration_mask(
                profile
            )
        )


    def run_ecc(
        self,
        current: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[
        bool,
        float,
        np.ndarray,
    ]:

        warp = (
            initial.copy()
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6,
        )

        try:

            cc, warp = (
                cv2.findTransformECC(
                    self.template,
                    current,
                    warp,
                    cv2.MOTION_EUCLIDEAN,
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

        if (
            not np.isfinite(
                cc
            )
            or cc < 0.30
        ):

            return (
                False,
                float(cc),
                warp,
            )

        return (
            True,
            float(cc),
            warp,
        )


    def choose_transform(
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

        previous = (
            self.last_good
        )

        # ------------------------------------------------------
        # First try: continue tracking from previous transform
        # ------------------------------------------------------

        initial = (
            previous
            if previous is not None
            else identity_warp()
        )

        ok, cc, candidate = (
            self.run_ecc(
                current,
                initial,
            )
        )

        if (
            ok
            and rigid_warp_is_sane(
                candidate,
                previous,
                self.max_rotation,
                self.max_frame_rotation,
                self.max_frame_shift,
            )
        ):

            self.last_good = (
                candidate.copy()
            )

            self.failed_frames = 0

            self.tracked += 1

            self.cc_sum += cc
            self.cc_count += 1

            return (
                candidate,
                cc,
                "tracked",
            )

        # ------------------------------------------------------
        # Try reacquisition independently from identity
        # ------------------------------------------------------

        ok2, cc2, candidate2 = (
            self.run_ecc(
                current,
                identity_warp(),
            )
        )

        if (
            ok2
            and rigid_warp_is_sane(
                candidate2,
                None,
                self.max_rotation,
                self.max_frame_rotation,
                self.max_frame_shift,
            )
        ):

            # If we had previous tracking, don't accept a huge sudden
            # jump merely because the isolated ECC fit looks good.
            if previous is not None:

                old_angle = (
                    warp_angle_deg(
                        previous
                    )
                )

                new_angle = (
                    warp_angle_deg(
                        candidate2
                    )
                )

                old_tx, old_ty = (
                    warp_translation(
                        previous
                    )
                )

                new_tx, new_ty = (
                    warp_translation(
                        candidate2
                    )
                )

                rotation_jump = (
                    angle_difference_deg(
                        new_angle,
                        old_angle,
                    )
                )

                translation_jump = (
                    math.hypot(
                        new_tx - old_tx,
                        new_ty - old_ty,
                    )
                )

                # Reacquisition is allowed somewhat more freedom than
                # ordinary frame-to-frame tracking.
                if (
                    rotation_jump
                    > 2.0
                    * self.max_frame_rotation
                    or translation_jump
                    > 2.0
                    * self.max_frame_shift
                ):

                    ok2 = False

            if ok2:

                self.last_good = (
                    candidate2.copy()
                )

                self.failed_frames = 0

                self.reacquired += 1

                self.cc_sum += cc2
                self.cc_count += 1

                return (
                    candidate2,
                    cc2,
                    "reacquired",
                )

        # ------------------------------------------------------
        # Failed.
        #
        # Briefly keep last good transformation.
        # ------------------------------------------------------

        self.rejected += 1

        if (
            self.last_good is not None
            and self.failed_frames
            < self.max_hold
        ):

            self.failed_frames += 1

            self.held += 1

            return (
                self.last_good.copy(),
                float("nan"),
                "held",
            )

        # ------------------------------------------------------
        # If tracking has been lost for too long, return identity.
        #
        # This may be imperfect but it CANNOT geometrically deform
        # the display.
        # ------------------------------------------------------

        self.failed_frames += 1

        self.identity += 1

        return (
            identity_warp(),
            float("nan"),
            "identity",
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

        warp, cc, method = (
            self.choose_transform(
                display
            )
        )

        # OpenCV findTransformECC returns the warp in the convention
        # used together with WARP_INVERSE_MAP.
        aligned = cv2.warpAffine(
            display,
            warp,
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

        return (
            aligned,
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

    # Parse fixed-grid arguments.
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

    if wrapper_args.max_hold < 0:

        print(
            "Error: --max-hold cannot be negative.",
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

        reference_time = (
            wrapper_args.stabilize_time
            if wrapper_args.stabilize_time
            is not None
            else 0.5 * info.duration
        )

        reference_time = max(
            0.0,
            min(
                reference_time,
                info.duration,
            ),
        )

        # ------------------------------------------------------
        # Reference frame
        # ------------------------------------------------------

        print(
            (
                "Acquiring rigid stabilization reference "
                f"near t={reference_time:.3f} s..."
            ),
            file=sys.stderr,
        )

        (
            reference_display,
            actual_reference_time,
        ) = (
            rect_app.find_reference_display(
                args.video,
                info,
                profile,
                args.roi,
                args.processing_width,
                sample_fps,
                reference_time,
            )
        )

        reference_rectified = (
            rect_app.rectify_display(
                reference_display,
                profile,
                fixed_extra.quad,
            )
        )

        print(
            (
                "Rigid stabilization reference acquired at "
                f"t={actual_reference_time:.3f} s."
            ),
            file=sys.stderr,
        )

        tracker = (
            RigidTracker(
                reference_rectified,
                max_rotation=(
                    wrapper_args.max_rotation
                ),
                max_frame_rotation=(
                    wrapper_args.max_frame_rotation
                ),
                max_frame_shift=(
                    wrapper_args.max_frame_shift
                ),
                max_hold=(
                    wrapper_args.max_hold
                ),
            )
        )

        original_factory = (
            fixed_app.make_rectified_finder
        )

        # ------------------------------------------------------
        # Finder factory used by fixed-grid program
        # ------------------------------------------------------

        def make_rigid_finder(
            quad: np.ndarray,
        ):

            normal_finder = (
                NORMAL_RECTIFIED_FINDER_FACTORY(
                    quad
                )
            )

            current_profile_id = None

            def finder(
                frame,
                in_profile,
                previous_box,
                roi,
            ):

                nonlocal current_profile_id

                # debug-dir causes a second sequential pass from frame 0.
                if previous_box is None:

                    tracker.reset()

                new_profile_id = (
                    id(
                        in_profile
                    )
                )

                if (
                    current_profile_id
                    != new_profile_id
                ):

                    tracker.set_profile(
                        in_profile
                    )

                    current_profile_id = (
                        new_profile_id
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
                    aligned,
                    _cc,
                    _method,
                ) = tracker.align(
                    rectified
                )

                return (
                    aligned,
                    box,
                )

            return finder

        fixed_app.make_rectified_finder = (
            make_rigid_finder
        )

        # ------------------------------------------------------
        # Run fixed-grid decoder
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
        # Diagnostics
        # ------------------------------------------------------

        print()

        print(
            "Rigid stabilization diagnostics:"
        )

        print(
            f"  processed  : {tracker.total}"
        )

        print(
            f"  tracked    : {tracker.tracked}"
        )

        print(
            f"  reacquired : {tracker.reacquired}"
        )

        print(
            f"  held       : {tracker.held}"
        )

        print(
            f"  identity   : {tracker.identity}"
        )

        print(
            f"  rejected   : {tracker.rejected}"
        )

        if (
            tracker.cc_count > 0
        ):

            mean_cc = (
                tracker.cc_sum
                / tracker.cc_count
            )

            print(
                f"  mean ECC   : {mean_cc:.4f}"
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