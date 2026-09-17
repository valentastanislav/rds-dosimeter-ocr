# RDS dosimeter OCR

Experimental Python pipeline for extracting changing seven-segment readings from RADOS dosimeter videos.

The repository currently contains working pipelines for the **RDS-30** and **RDS-200**. Current external testing is focused on the RDS-30 beta pipeline.

## Status

### RDS-30 beta

The current frozen RDS-30 decoder configuration uses:

- optical-flow stabilization
- manual reference display box
- manual perspective quad
- profile-defined digit/segment geometry
- glyph-independent residual-geometry scoring
- joint-spatial decoding
- confidence rejection at `0.358`
- weak-9 margin rejection at `0.400`

The frozen algorithm is tagged as:

```text
rds30-gi-weak9-2026-09-11
```

That tag points to the algorithmic freeze before independent validation. The current `main` branch contains the same algorithm plus validation metadata.

Independent validation on `IMG_0751.MOV` used ground truth prepared before OCR and produced:

- 1088 stable samples
- 1087 accepted samples
- 1087 correct accepted samples
- 0 wrong accepted samples
- 99.91% stable-sample coverage
- 100% accepted-sample accuracy

This is still a beta result, not a claim of general performance. New videos from the same instrument type under different conditions are especially useful.

### RDS-200 regression baseline

The older RDS-200 pipeline is retained and regression-tested. Refactoring has preserved exact output parity with the historical 331-sample regression baseline.

RDS-200 still uses a manually selected digit grid. RDS-30 does not.

## Quick start for an RDS-30 beta test

### 1. Clone the repository

```bash
git clone https://github.com/valentastanislav/rds-dosimeter-ocr.git
cd rds-dosimeter-ocr
```

For an exact checkout of the frozen RDS-30 algorithm:

```bash
git checkout rds30-gi-weak9-2026-09-11
```

The current `main` branch contains the same frozen algorithm plus the independent-validation files.

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The Python dependencies are currently `numpy` and `opencv-python`.

The system also needs `ffmpeg` and `ffprobe` available on `PATH`.

Examples:

```bash
# macOS with Homebrew
brew install ffmpeg

# Debian/Ubuntu
sudo apt install ffmpeg
```

### 3. Choose a reference time

Pick a time in the video where the dosimeter display is clearly visible and reasonably sharp. For a first attempt, a value such as `20.0` seconds is usually convenient.

This time is used only to define the reference geometry for tracking.

### 4. Run the frozen RDS-30 beta configuration

Replace `my_video.MOV` with your video filename:

```bash
python3 dosimeter_get_values_flow.py my_video.MOV rds30_intervals.csv \
  --profile rds30 \
  --track-time 20.0 \
  --select-reference-box \
  --select-quad \
  --raw-output rds30_raw.csv \
  --joint-spatial-diagnostics-dir rds30_diagnostics
```

For `--profile rds30`, the validated beta decoder configuration is now selected automatically: joint-spatial decoding, glyph-independent geometry emission, confidence threshold `0.358`, and weak-9 margin threshold `0.400`.

These values can still be overridden explicitly from the command line; `--decoder-strategy default` selects the historical decoder.

For a beta test, **do not tune `0.358` or `0.400` after looking at the result**. The point is to test the frozen configuration on genuinely new data.

## Manual geometry

The RDS-30 run above asks for two manual selections.

### Reference box

`--select-reference-box` opens the reference frame and asks for a rectangular box.

Select a **loose rectangle containing the complete physical LCD/display plus a small visible margin on all sides**.

The box is only a carrier/crop region. Its edges do **not** need to coincide with the LCD edges.

Conceptually:

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

The program prints the normalized result, for example:

```text
--reference-box 0.307407,0.573958,0.696296,0.646875
```

Keep this line if you want to reproduce the run later without clicking again.

### Perspective quad

`--select-quad` asks for the four **actual physical corners of the LCD inside the reference box**.

Click them in this order:

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

Use the LCD boundary, not the digit boundary, the instrument housing, or the reference-box edge.

The program prints the normalized quad, for example:

```text
--quad 0.032520,0.067606,0.981707,0.067606,0.987805,0.932394,0.012195,0.923944
```

Selection can be cancelled with `C` or `Esc`.

### No digit grid for RDS-30

Do **not** select a digit grid for the RDS-30. Digit positions and seven-segment geometry are part of the RDS-30 profile, including the possibly blank leading digit.

RDS-200 is different: it still requires a manual digit grid.

## Main outputs

For the example command above:

- `rds30_intervals.csv` — reconstructed stable value intervals
- `rds30_raw.csv` — sampled OCR values with at least `time_s`, `value`, and `confidence`
- `rds30_diagnostics/` — joint-spatial geometry diagnostics, including `geometry.csv` and any requested preview diagnostics
- terminal output — optical-flow tracking statistics and run diagnostics

The raw output may contain rejected samples with no accepted value. This is intentional: the RDS-30 beta decoder is conservative and prefers rejecting uncertain frames over returning a likely wrong value.

Therefore, recognition/coverage below 100% is not automatically a failure. For beta testing, wrong accepted values are more important than a modest number of rejected samples.

## Optical-flow diagnostics

A successful run prints tracking statistics such as:

```text
steps               : 2001
accepted motion     : 2001
rejected motion     : 0
point redetections  : 210
mean tracked pts    : 28.3
mean RANSAC inliers : 28.3
```

Frequent point redetection is normal. A large number of rejected motion steps, obvious tracking drift, or a badly rectified LCD is worth reporting.

## Beta-test feedback

For a useful external RDS-30 test, please keep the frozen decoder parameters unchanged and send back, if possible:

- the exact command used
- the selected `--reference-box`
- the selected `--quad`
- terminal output
- `rds30_intervals.csv`
- `rds30_raw.csv`
- `rds30_diagnostics/geometry.csv`
- the original video, if it can be shared

If true displayed values are known, the best validation procedure is to write them down independently **before inspecting the OCR output**.

Ground truth is used only for evaluation; it is not used by the production decoder.

## Important files

- `dosimeter_get_values.py` — core profiles and decoding support
- `dosimeter_get_values_flow.py` — optical-flow stabilization and primary video pipeline
- `dosimeter_get_values_fixedgrid.py` — fixed digit geometry and segment measurement
- `dosimeter_rds30_joint_spatial.py` — RDS-30 joint-spatial decoder
- `tests/evaluate_dosimeter_ground_truth.py` — raw-sample benchmark evaluator
- `tests/video_manifest.tsv` — development/validation video roles and recorded geometry
- `tests/IMG_0751_true_values.txt` — independent RDS-30 validation ground truth
- `docs/CURRENT_STATE.md` — historical development notes; some sections predate the current RDS-30 beta state
- `AGENTS.md` — project/development context for coding agents

## Reproducibility checkpoints

Two useful tags are kept in the repository:

```text
pre-cleanup-2026-09-07
```

Historical pre-cleanup snapshot at commit `eb003e7`.

```text
rds30-gi-weak9-2026-09-11
```

Frozen RDS-30 glyph-independent decoder with confidence threshold `0.358` and weak-9 margin threshold `0.400`.

## Development principle

The intended processing chain is:

```text
video
  -> geometry / stabilization
  -> rectified display
  -> digit geometry
  -> segment measurement
  -> temporal filtering
  -> digit decoding
  -> decimal decoding
  -> interval construction
  -> diagnostics / output
```

Ground-truth values must remain independent of production OCR logic.
