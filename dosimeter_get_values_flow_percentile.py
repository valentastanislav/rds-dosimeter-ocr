#!/usr/bin/env python3
"""
RDS-200 optical-flow pipeline with a genuinely configurable
segment-pixel percentile.

This is an experimental wrapper around the existing working pipeline.

It changes ONLY the pixel statistic used inside every seven-segment
polygon.

The previous version patched only:

    dosimeter_get_values.extract_darkness_from_patches

but some layers of the flow/fixed-grid/debug wrapper chain may already
hold imported aliases of that function.  In that case changing the
attribute in the core module does not change the active reference.

This version patches every loaded alias in the complete dosimeter
pipeline and reports exactly which functions were replaced and how many
times the replacement extractor was called.

No expected values are used.
No digit-specific corrections are used.

Required existing files:

    dosimeter_get_values.py
    dosimeter_get_values_roi.py
    dosimeter_get_values_rectified.py
    dosimeter_get_values_fixedgrid.py
    dosimeter_get_values_flow.py
    dosimeter_get_values_flow_diag.py
    dosimeter_get_values_flow_tightsegments.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import dosimeter_get_values as core
import dosimeter_get_values_roi as roi_app
import dosimeter_get_values_rectified as rectified_app
import dosimeter_get_values_fixedgrid as fixed_app
import dosimeter_get_values_flow as flow_app
import dosimeter_get_values_flow_diag as diag_app
import dosimeter_get_values_flow_tightsegments as tight_app


# ======================================================================
# Original function objects
# ======================================================================


ORIGINAL_EXTRACT_FROM_PATCHES = (
    core.extract_darkness_from_patches
)

ORIGINAL_EXTRACT_DARKNESS = (
    core.extract_darkness
)


# ======================================================================
# Diagnostics
# ======================================================================


STATS = {
    "extract_darkness_calls": 0,
    "extract_from_patches_calls": 0,
}


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
        "--segment-percentile",
        type=float,
        default=50.0,
        help=(
            "pixel percentile used inside every segment polygon; "
            "lower values use the darker part of the mask "
            "(default: 50)"
        ),
    )

    return parser.parse_known_args(
        argv
    )


# ======================================================================
# Replacement primitive extractor
# ======================================================================


def make_extract_from_patches(
    percentile: float,
):

    def extract_darkness_from_patches_percentile(
        patches,
        segment_masks,
        segment_percentile=50.0,
    ):
        """
        Same calculation as the original function except that the
        segment pixel percentile is forced to the value supplied on
        this wrapper's command line.

        Background definition remains unchanged:

            90th percentile of the complete normalized digit patch.
        """

        # The argument is intentionally ignored.
        #
        # Some callers pass the original default 50 explicitly or
        # implicitly.  For this experiment we want one controlled
        # percentile everywhere in the RDS-200 primary extraction.
        del segment_percentile

        STATS[
            "extract_from_patches_calls"
        ] += 1

        result = np.empty(
            (
                len(patches),
                7,
            ),
            dtype=float,
        )

        for digit_index, patch in enumerate(
            patches
        ):

            background = float(
                np.percentile(
                    patch,
                    90.0,
                )
            )

            for segment_index, mask in enumerate(
                segment_masks
            ):

                pixels = patch[
                    mask
                ]

                if pixels.size == 0:

                    result[
                        digit_index,
                        segment_index,
                    ] = np.nan

                    continue

                segment_level = float(
                    np.percentile(
                        pixels,
                        percentile,
                    )
                )

                result[
                    digit_index,
                    segment_index,
                ] = (
                    background
                    - segment_level
                )

        return result

    return extract_darkness_from_patches_percentile


# ======================================================================
# Replacement high-level extractor
# ======================================================================


def make_extract_darkness(
    replacement_from_patches,
    percentile: float,
):

    def extract_darkness_percentile(
        display,
        profile,
        segment_masks,
        segment_percentile=50.0,
    ):
        """
        Replacement for the ordinary extract_darkness() entry point.

        We patch this function as well as extract_darkness_from_patches()
        so that imported aliases at either level cannot bypass the
        requested percentile.
        """

        del segment_percentile

        STATS[
            "extract_darkness_calls"
        ] += 1

        patches = core.digit_patches(
            display,
            profile,
        )

        return replacement_from_patches(
            patches,
            segment_masks,
            segment_percentile=percentile,
        )

    return extract_darkness_percentile


# ======================================================================
# Patch every imported alias in the pipeline
# ======================================================================


PIPELINE_MODULES = (
    core,
    roi_app,
    rectified_app,
    fixed_app,
    flow_app,
    diag_app,
    tight_app,
)


def install_extractor_hooks(
    replacement_from_patches,
    replacement_extract_darkness,
):
    """
    Replace attributes which still refer to either original function
    object.

    This catches constructs such as:

        from dosimeter_get_values import extract_darkness

    where changing core.extract_darkness later would otherwise leave
    the imported alias untouched.
    """

    patched = []

    for module in PIPELINE_MODULES:

        module_dict = vars(
            module
        )

        for attribute_name, value in list(
            module_dict.items()
        ):

            if (
                value
                is ORIGINAL_EXTRACT_FROM_PATCHES
            ):

                setattr(
                    module,
                    attribute_name,
                    replacement_from_patches,
                )

                patched.append(
                    (
                        module,
                        attribute_name,
                        value,
                        "from_patches",
                    )
                )

            elif (
                value
                is ORIGINAL_EXTRACT_DARKNESS
            ):

                setattr(
                    module,
                    attribute_name,
                    replacement_extract_darkness,
                )

                patched.append(
                    (
                        module,
                        attribute_name,
                        value,
                        "extract_darkness",
                    )
                )

    return patched


def restore_extractor_hooks(
    patched,
) -> None:

    for (
        module,
        attribute_name,
        original,
        _kind,
    ) in reversed(
        patched
    ):

        setattr(
            module,
            attribute_name,
            original,
        )


# ======================================================================
# Diagnostics
# ======================================================================


def print_patch_diagnostics(
    patched,
) -> None:

    print(
        "Extractor aliases patched:"
    )

    if not patched:

        print(
            "  NONE"
        )

        return

    for (
        module,
        attribute_name,
        _original,
        kind,
    ) in patched:

        print(
            (
                f"  {module.__name__}."
                f"{attribute_name}"
                f"  [{kind}]"
            )
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

    percentile = float(
        wrapper_args.segment_percentile
    )

    if not (
        0.0
        <= percentile
        <= 100.0
    ):

        print(
            (
                "Error: --segment-percentile "
                "must be between 0 and 100."
            ),
            file=sys.stderr,
        )

        return 1

    STATS[
        "extract_darkness_calls"
    ] = 0

    STATS[
        "extract_from_patches_calls"
    ] = 0

    replacement_from_patches = (
        make_extract_from_patches(
            percentile
        )
    )

    replacement_extract_darkness = (
        make_extract_darkness(
            replacement_from_patches,
            percentile,
        )
    )

    patched = (
        install_extractor_hooks(
            replacement_from_patches,
            replacement_extract_darkness,
        )
    )

    saved_argv = (
        sys.argv
    )

    # Remove our private argument before the ordinary pipeline parses
    # the command line.
    sys.argv = [
        saved_argv[0],
        *remaining,
    ]

    print(
        "RDS-200 segment-percentile experiment:"
    )

    print(
        (
            f"  requested percentile : "
            f"{percentile:.1f}"
        )
    )

    print(
        "  background percentile: 90.0"
    )

    print(
        "  digit-specific fixes  : NONE"
    )

    print(
        "  expected values used  : NONE"
    )

    print()

    print_patch_diagnostics(
        patched
    )

    print()

    try:

        result = (
            tight_app.main()
        )

    finally:

        restore_extractor_hooks(
            patched
        )

        sys.argv = (
            saved_argv
        )

    print()

    print(
        "Percentile extractor diagnostics:"
    )

    print(
        (
            f"  requested percentile        : "
            f"{percentile:.1f}"
        )
    )

    print(
        (
            f"  extract_darkness calls      : "
            f"{STATS['extract_darkness_calls']}"
        )
    )

    print(
        (
            f"  extract_from_patches calls  : "
            f"{STATS['extract_from_patches_calls']}"
        )
    )

    if (
        STATS[
            "extract_from_patches_calls"
        ]
        == 0
    ):

        print()

        print(
            "WARNING: replacement extractor was NEVER called."
        )

        print(
            (
                "This means the active optical-flow path still "
                "bypasses this hook."
            )
        )

    return result


if __name__ == "__main__":

    raise SystemExit(
        main()
    )