# RDS dosimeter OCR

Python OCR pipeline for extracting changing seven-segment readings from hand-held videos of RADOS dosimeters.

The repository currently contains beta pipelines for **RDS-30** and **RDS-200**. Both use the same primary entry point:

```bash
python3 dosimeter_get_values_flow.py VIDEO.MOV intervals.csv [options]
```

The project is still beta software. Validation results are evidence for the tested videos, not a general accuracy guarantee.

## Installation

```bash
git clone https://github.com/valentastanislav/rds-dosimeter-ocr.git
cd rds-dosimeter-ocr

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Python dependencies are currently `numpy` and `opencv-python`.

The system also needs `ffmpeg` and `ffprobe` on `PATH`.

```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt install ffmpeg
```

## RDS-200 beta

The RDS-200 beta pipeline uses:

- one manually selected reference display box;
- one manually selected perspective quad;
- one manually selected tight grid around the three large digits;
- ring-only optical-flow features outside the changing LCD;
- direct reference-to-frame registration with cumulative tracking as fallback;
- profile-owned per-digit seven-segment sampling geometry;
- automatic decimal-point detection for the two RDS-200 decimal positions;
- RDS-200 pattern refinement enabled by default;
- raw-sample confidence rejection at `0.20`;
- final-summary confidence filtering at `0.20`.

The reference box, quad, and digit grid are **video-specific**. Do not copy them from one recording to another.

### RDS-200 first run

Choose a reference time where the display is clear and reasonably sharp. Then run:

```bash
python3 dosimeter_get_values_flow.py my_video.MOV rds200_intervals.csv \
  --profile rds200 \
  --track-time 20.0 \
  --select-reference-box \
  --select-quad \
  --select-grid \
  --decimal-places auto \
  --contrast auto \
  --raw-output rds200_raw.csv \
  --debug-dir rds200_debug
```

The program prints reusable normalized values for:

```text
--reference-box ...
--quad ...
--grid ...
```

A reproducible rerun can use those values instead of the three interactive selectors.

### RDS-200 tracking defaults

Unless explicitly overridden, the RDS-200 profile uses:

```text
registration            direct reference -> frame
fallback                cumulative frame-to-frame transform
tracking feature region outside ring only
ring padding             0.25 of reference-box size
max translation          150 px
max rotation             10 deg
scale range              0.85 .. 1.15
sample rate              5 Hz
filter window            1
mode window              1
min confidence           0.20
summary min confidence   0.20
```

These are profile defaults. They normally do not need to appear in the command line.

### RDS-200 beta validation state

The current configuration was regression-checked on several development/verification recordings spanning different display values and recording conditions, including `IMG_0742`, `IMG_0744`, `IMG_0747`, and `IMG_1151`.

Examples from the stable-sample evaluator:

- `IMG_0742`: 158/158 stable samples correct;
- `IMG_0744`: 156 correct, 7 wrong, 1 missing out of 164 evaluated stable samples;
- `IMG_0747`: 148 correct, 7 wrong, 4 missing out of 159 evaluated stable samples.

The remaining RDS-200 errors are concentrated in difficult low-contrast/blurred frames. Low-confidence final intervals are retained in the interval CSV for transparency but can be excluded from physical summary statistics as described below.

Do not tune the production geometry or decoder against these videos further; they are now consumed development/regression data.

## RDS-30 beta

The frozen RDS-30 configuration uses:

- optical-flow stabilization;
- manual reference display box;
- manual perspective quad;
- profile-defined digit geometry;
- glyph-independent residual-geometry scoring;
- joint-spatial decoding;
- confidence rejection at `0.358`;
- weak-9 margin rejection at `0.400`.

The frozen algorithm tag is:

```text
rds30-gi-weak9-2026-09-11
```

Independent validation on `IMG_0751.MOV`, with ground truth prepared before OCR, produced:

```text
stable samples:        1088
accepted / recognized: 1087
correct accepted:      1087
wrong accepted:           0
stable coverage:        99.91 %
accepted accuracy:     100.00 %
```

This is one independent validation recording, not a universal performance claim.

### RDS-30 beta run

```bash
python3 dosimeter_get_values_flow.py my_video.MOV rds30_intervals.csv \
  --profile rds30 \
  --track-time 20.0 \
  --select-reference-box \
  --select-quad \
  --decoder-strategy rds30-joint-spatial \
  --joint-spatial-geometry-emission glyph-independent \
  --joint-spatial-confidence 0.358 \
  --joint-spatial-min-nine-margin 0.400 \
  --raw-output rds30_raw.csv \
  --joint-spatial-diagnostics-dir rds30_diagnostics
```

For external beta validation, do not retune `0.358` or `0.400` after inspecting the result.

RDS-30 does **not** use a manual digit grid.

## Manual geometry

### Reference box

`--select-reference-box` asks for a loose rectangle containing the complete physical LCD/display plus a small visible margin.

The box is a tracking/crop carrier. Its edges do not need to coincide with the LCD edges.

```text
+---------------------------+   reference box
|        small margin       |
|   +-------------------+   |
|   |        LCD        |   |
|   +-------------------+   |
|                           |
+---------------------------+
```

Do not select only the digits.

### Perspective quad

`--select-quad` asks for the four actual physical LCD corners inside the reference box.

Click in this order:

```text
1 = top-left
2 = top-right
3 = bottom-right
4 = bottom-left

1 ---------------- 2
|                  |
|       LCD        |
|                  |
4 ---------------- 3
```

Use the physical LCD boundary, not the digit boundary, instrument housing, or reference-box edge.

### RDS-200 digit grid

`--select-grid` is RDS-200 only.

Select a **tight rectangle around all three large digits** in the rectified display. Do not include the scale/unit text or bezel. The decimal dots do not need to define the grid boundary.

The selected grid is used to construct the three digit boxes; the decimal sampling positions are then derived consistently from the final digit geometry.

## Confidence and summary statistics

Two different confidence thresholds intentionally exist.

### `--min-confidence`

This acts on **raw OCR samples** before interval reconstruction.

Profile default:

```text
RDS-200: 0.20
RDS-30:  0.20
```

A raw sample below the threshold is rejected and written with no accepted value in the raw CSV.

### `--summary-min-confidence`

Default:

```text
0.20
```

This acts only on the **final reconstructed intervals when calculating summary statistics**.

Intervals below the threshold:

- remain in the interval CSV;
- remain visible in debug output;
- are excluded from time-weighted mean, variance, standard deviation, minimum, and maximum.

The summary reports both included and excluded duration, so the quality cut is explicit rather than silently deleting OCR output.

Set `--summary-min-confidence 0` if statistics over all final intervals are desired.

## General decoding defaults

```text
option                     RDS-200      RDS-30
------------------------------------------------
--decimal-places           auto         auto
--contrast                 auto         auto
--min-confidence           0.20         0.20
--summary-min-confidence   0.20         0.20
--filter-window            1            5
--mode-window              1            21
--sample-fps               5 Hz         30 Hz
--processing-width         540 px       540 px
```

Run:

```bash
python3 dosimeter_get_values_flow.py --help
```

for the complete current CLI help.

## Outputs

Typical outputs are:

- `*_intervals.csv` — reconstructed value intervals;
- `*_raw.csv` — every sampled OCR result before interval filling/merging;
- `*.summary.json` — time-weighted statistics and quality-cut coverage;
- `--debug-dir` — interval screenshots drawn from the exact cached main-pass displays used by OCR;
- terminal output — geometry, tracking, recognition, and summary diagnostics.

Debug geometry is required to correspond to the actual decoder geometry. It is not recomputed from a separate tracking pass.

## Ground-truth evaluation

If true displayed values are available, prepare them independently before inspecting OCR output when possible.

Use:

```bash
python3 tests/evaluate_dosimeter_ground_truth.py \
  tests/MY_VIDEO_true_values.txt \
  my_raw.csv \
  --guard 0.4
```

Ground truth is evaluation-only. It must not influence production tracking, geometry, decimal selection, confidence, acceptance, or expected value ranges.

## Testing

Run the full unit-test suite with:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

At the current RDS-200 beta checkpoint the suite contains 43 passing tests.

## Production and development files

Primary production files:

- `dosimeter_get_values.py` — profiles, segment measurement, decoding, interval construction, summary statistics;
- `dosimeter_get_values_flow.py` — primary stabilized video pipeline and user CLI;
- `dosimeter_get_values_fixedgrid.py` — fixed-grid geometry and RDS-200 decimal geometry;
- `dosimeter_get_values_roi.py` — ROI/front-end support;
- `dosimeter_get_values_rectified.py` — perspective rectification;
- `dosimeter_rds30_joint_spatial.py` — RDS-30 beta joint-spatial decoder.

Development/evaluation support:

- `dosimeter_get_values_flow_diag.py` — segment-level diagnostics;
- `dosimeter_get_values_flow_tightsegments.py` — historical RDS-200 development geometry helper retained for development tests;
- `tests/rds200_grid_refinement_sweep.py` — historical ground-truth-blind grid experiment;
- `tests/evaluate_dosimeter_ground_truth.py` — stable-sample evaluator;
- `tests/video_manifest.tsv` — recorded development/validation video roles and geometry.

The removed experimental RDS-200 wrapper stack is no longer a supported user entry point. Beta users should run `dosimeter_get_values_flow.py` directly.

## Reproducibility checkpoints

Historical RDS-30 tag:

```text
rds30-gi-weak9-2026-09-11
```

Historical pre-cleanup snapshot:

```text
pre-cleanup-2026-09-07
```

A dedicated RDS-200 beta tag should be created only after the beta branch is merged and the final test/help checks pass.

## Development principle

The intended production chain is:

```text
video
  -> stabilization
  -> fixed reference crop
  -> perspective rectification
  -> digit geometry
  -> segment measurement
  -> temporal filtering
  -> digit decoding
  -> decimal decoding
  -> interval reconstruction
  -> confidence-qualified summary statistics
  -> diagnostics / output
```

Keep geometry, OCR quality, confidence policy, and evaluation concerns separate. Do not compensate for poor geometry with video-specific decoder patches.
