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
