#!/usr/bin/env python3
"""
Conservative translation-only stabilization for RDS-200 videos.

Required files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py

Pipeline:

    video
      -> ROI display detection
      -> fixed perspective correction (--quad)
      -> translation-only stabilization (X/Y)
      -> fixed digit grid (--grid)
      -> decoder

There is deliberately:

    NO rotation
    NO scaling
    NO shear
    NO homography

Therefore this stabilizer cannot geometrically deform the LCD.

IMPORTANT:
The stabilization reference should be the same frame on which the
fixed digit grid was selected.

For the present IMG_1151 video the grid was selected at t=6.8 s,
therefore use:

    --stabilize-time 6.8
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


# Normal fixed-grid rectification:
# ROI crop -> fixed --quad perspective warp.
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
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--stabilize-time",
        type=float,
        required=True,
        help=(
            "reference time for translation stabilization; "
            "use the same time at which --grid was selected"
        ),
    )

    parser.add_argument(
        "--translation-min-cc",
        type=float,
        default=0.42,
        help=(
            "minimum ECC correlation for accepting a translation "
            "(default: 0.42)"
        ),
    )

    parser.add_argument(
        "--translation-max-x",
        type=float,
        default=18.0,
        help=(
            "maximum absolute horizontal correction in canonical pixels "
            "(default: 18)"
        ),
    )

    parser.add_argument(
        "--translation-max-y",
        type=float,
        default=18.0,
        help=(
            "maximum absolute vertical correction in canonical pixels "
            "(default: 18)"
        ),
    )

    parser.add_argument(
        "--translation-max-step",
        type=float,
        default=5.0,
        help=(
            "maximum accepted translation change between adjacent frames "
            "(default: 5 pixels)"
        ),
    )

    parser.add_argument(
        "--translation-smoothing",
        type=float,
        default=0.75,
        help=(
            "weight of new translation, 0..1 "
            "(default: 0.75)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Image used only for geometric matching
# ======================================================================


_ECC_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def prepare_registration_image(
    image: np.ndarray,
) -> np.ndarray:

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image

    # Contrast normalization used ONLY for registration.
    gray = _ECC_CLAHE.apply(
        gray
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return (
        gray.astype(np.float32)
        / 255.0
    )


# ======================================================================
# Mask: use only more-or-less static display features
# ======================================================================


def make_registration_mask(
    profile: core.Profile,
) -> np.ndarray:

    height = (
        profile.canonical_height
    )

    width = (
        profile.canonical_width
    )

    mask = np.full(
        (
            height,
            width,
        ),
        255,
        dtype=np.uint8,
    )

    # ----------------------------------------------------------
    # Exclude changing large digits
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
                digit_x1 - 20,
            ),
            max(
                0,
                digit_y1 - 20,
            ),
        ),
        (
            min(
                width - 1,
                digit_x2 + 20,
            ),
            min(
                height - 1,
                digit_y2 + 20,
            ),
        ),
        0,
        -1,
    )

    # ----------------------------------------------------------
    # Exclude changing left bar graph
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
                    0.23 * width
                )
            ),
            height - 1,
        ),
        0,
        -1,
    )

    return mask


# ======================================================================
# Translation helpers
# ======================================================================


def identity_translation() -> np.ndarray:

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


def translation_xy(
    warp: np.ndarray,
) -> tuple[float, float]:

    return (
        float(
            warp[0, 2]
        ),
        float(
            warp[1, 2]
        ),
    )


# ======================================================================
# Conservative translation tracker
# ======================================================================


class TranslationTracker:

    def __init__(
        self,
        reference: np.ndarray,
        minimum_cc: float,
        maximum_x: float,
        maximum_y: float,
        maximum_step: float,
        smoothing: float,
    ) -> None:

        self.reference = (
            reference.copy()
        )

        self.template = (
            prepare_registration_image(
                reference
            )
        )

        self.height, self.width = (
            self.template.shape
        )

        self.minimum_cc = (
            minimum_cc
        )

        self.maximum_x = (
            maximum_x
        )

        self.maximum_y = (
            maximum_y
        )

        self.maximum_step = (
            maximum_step
        )

        self.smoothing = (
            smoothing
        )

        self.mask: (
            np.ndarray | None
        ) = None

        self.last_warp = (
            identity_translation()
        )

        self.have_good_warp = False

        # Diagnostics
        self.total = 0
        self.accepted = 0
        self.held = 0
        self.rejected_cc = 0
        self.rejected_absolute = 0
        self.rejected_step = 0

        self.cc_sum = 0.0
        self.cc_count = 0

        self.max_seen_abs_x = 0.0
        self.max_seen_abs_y = 0.0


    def reset(
        self,
    ) -> None:

        self.last_warp = (
            identity_translation()
        )

        self.have_good_warp = False


    def set_profile(
        self,
        profile: core.Profile,
    ) -> None:

        self.mask = (
            make_registration_mask(
                profile
            )
        )


    def find_translation(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        str,
    ]:

        self.total += 1

        current = (
            prepare_registration_image(
                display
            )
        )

        # Start ECC at previous shift.
        # If no good shift exists yet, start at zero.
        initial = (
            self.last_warp.copy()
            if self.have_good_warp
            else identity_translation()
        )

        candidate = (
            initial.copy()
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            80,
            1e-6,
        )

        try:

            cc, candidate = (
                cv2.findTransformECC(
                    self.template,
                    current,
                    candidate,
                    cv2.MOTION_TRANSLATION,
                    criteria,
                    self.mask,
                    5,
                )
            )

        except cv2.error:

            self.held += 1

            return (
                self.last_warp.copy(),
                float("nan"),
                "held",
            )

        cc = float(
            cc
        )

        tx, ty = (
            translation_xy(
                candidate
            )
        )

        self.max_seen_abs_x = max(
            self.max_seen_abs_x,
            abs(tx),
        )

        self.max_seen_abs_y = max(
            self.max_seen_abs_y,
            abs(ty),
        )

        # ------------------------------------------------------
        # Correlation threshold
        # ------------------------------------------------------

        if (
            not np.isfinite(
                cc
            )
            or cc < self.minimum_cc
        ):

            self.rejected_cc += 1
            self.held += 1

            return (
                self.last_warp.copy(),
                cc,
                "held",
            )

        # ------------------------------------------------------
        # Absolute shift must remain small
        # ------------------------------------------------------

        if (
            abs(tx) > self.maximum_x
            or abs(ty) > self.maximum_y
        ):

            self.rejected_absolute += 1
            self.held += 1

            return (
                self.last_warp.copy(),
                cc,
                "held",
            )

        # ------------------------------------------------------
        # Frame-to-frame motion must also remain small
        # ------------------------------------------------------

        if self.have_good_warp:

            previous_tx, previous_ty = (
                translation_xy(
                    self.last_warp
                )
            )

            step = math.hypot(
                tx - previous_tx,
                ty - previous_ty,
            )

            if (
                step > self.maximum_step
            ):

                self.rejected_step += 1
                self.held += 1

                return (
                    self.last_warp.copy(),
                    cc,
                    "held",
                )

            # --------------------------------------------------
            # Temporal smoothing
            # --------------------------------------------------

            alpha = (
                self.smoothing
            )

            tx = (
                alpha * tx
                + (1.0 - alpha)
                * previous_tx
            )

            ty = (
                alpha * ty
                + (1.0 - alpha)
                * previous_ty
            )

        accepted = (
            identity_translation()
        )

        accepted[
            0,
            2,
        ] = tx

        accepted[
            1,
            2,
        ] = ty

        self.last_warp = (
            accepted
        )

        self.have_good_warp = True

        self.accepted += 1

        self.cc_sum += cc
        self.cc_count += 1

        return (
            accepted.copy(),
            cc,
            "accepted",
        )


    def align(
        self,
        display: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        str,
    ]:

        warp, cc, method = (
            self.find_translation(
                display
            )
        )

        # This can ONLY translate the image.
        aligned = (
            cv2.warpAffine(
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

    if not (
        0.0
        <= wrapper_args.translation_smoothing
        <= 1.0
    ):

        print(
            "Error: --translation-smoothing must be between 0 and 1.",
            file=sys.stderr,
        )

        return 1

    saved_argv = (
        sys.argv
    )

    # ----------------------------------------------------------
    # Parse fixedgrid arguments
    # ----------------------------------------------------------

    try:

        sys.argv = [
            saved_argv[0],
            *remaining,
        ]

        (
            fixed_extra,
            roi_remaining,
        ) = (
            fixed_app.parse_extra_args()
        )

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

        reference_time = max(
            0.0,
            min(
                wrapper_args.stabilize_time,
                info.duration,
            ),
        )

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # This is the same geometrical state on which the fixed
        # grid was manually selected.
        # ------------------------------------------------------

        print(
            (
                "Acquiring translation reference "
                f"near t={reference_time:.3f} s..."
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
                "Translation reference acquired at "
                f"t={actual_time:.3f} s."
            ),
            file=sys.stderr,
        )

        tracker = TranslationTracker(
            reference=(
                reference_rectified
            ),
            minimum_cc=(
                wrapper_args.translation_min_cc
            ),
            maximum_x=(
                wrapper_args.translation_max_x
            ),
            maximum_y=(
                wrapper_args.translation_max_y
            ),
            maximum_step=(
                wrapper_args.translation_max_step
            ),
            smoothing=(
                wrapper_args.translation_smoothing
            ),
        )

        original_factory = (
            fixed_app.make_rectified_finder
        )

        # ------------------------------------------------------
        # Finder:
        #
        # fixed quad first
        # then translation ONLY
        # ------------------------------------------------------

        def make_translation_finder(
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

                # debug-dir performs another pass from frame zero.
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
            make_translation_finder
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
            "Translation-only stabilization diagnostics:"
        )

        print(
            f"  processed         : {tracker.total}"
        )

        print(
            f"  accepted          : {tracker.accepted}"
        )

        print(
            f"  held previous     : {tracker.held}"
        )

        print(
            f"  rejected CC       : {tracker.rejected_cc}"
        )

        print(
            f"  rejected absolute : {tracker.rejected_absolute}"
        )

        print(
            f"  rejected step     : {tracker.rejected_step}"
        )

        if (
            tracker.cc_count > 0
        ):

            mean_cc = (
                tracker.cc_sum
                / tracker.cc_count
            )

            print(
                f"  mean ECC          : {mean_cc:.4f}"
            )

        print(
            (
                "  max candidate |dx|: "
                f"{tracker.max_seen_abs_x:.2f} px"
            )
        )

        print(
            (
                "  max candidate |dy|: "
                f"{tracker.max_seen_abs_y:.2f} px"
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