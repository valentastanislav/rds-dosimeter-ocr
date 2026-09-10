# RDS dosimeter OCR

Experimental Python pipeline for extracting changing seven-segment
readings from RADOS dosimeter videos.

Current development is focused on the RDS-200.

The project started as a sequence of experimental wrappers. The
`cleanup` branch is consolidating the successful pieces into a cleaner
architecture while preserving benchmark behaviour.

## Important files

- `AGENTS.md` — instructions/context for Codex
- `docs/CURRENT_STATE.md` — current OCR state, known failures and deprecated approaches
- `tests/evaluate_dosimeter_ground_truth.py` — raw-sample benchmark
- `tests/IMG_1151_true_values.txt` — current test-video ground truth
- `dosimeter_get_values.py` — original/core implementation
- `dosimeter_get_values_fixedgrid.py` — fixed digit geometry and local segment measurement
- `dosimeter_get_values_flow.py` — optical-flow stabilization and main-pass debug cache

## Dependencies

See `requirements.txt`.

The system also needs `ffmpeg` and `ffprobe`.

## Usage
Manual initialization

--reference-box
    Select a loose rectangular region containing the complete physical
    LCD/display plus a small visible margin on all sides. The reference-box
    edges are only a carrier/crop boundary; they are not the LCD boundary.

--quad
    Inside that reference crop, mark the four actual physical corners of
    the LCD/display in order:
    top-left, top-right, bottom-right, bottom-left.

RDS-30:
    Do not select a digit grid. Digit positions and segment geometry are
    properties of the RDS-30 profile, including the possibly blank leading
    digit.

RDS-200:
    A manual digit grid is still required.