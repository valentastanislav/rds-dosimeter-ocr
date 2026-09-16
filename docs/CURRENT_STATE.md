# Current development state

## Goal and scope

Read changing seven-segment LCD values from hand-held videos of RADOS dosimeters.

The shared production architecture is approximately:

```text
flow
  -> fixedgrid
  -> ROI
  -> core measurement / filtering / decoding
```

The current public beta remains focused on RDS-30. RDS-200 remains supported as a regression target and is also undergoing a separate OCR-quality experiment described below.

Ground truth is evaluation-only. It must never influence production geometry, decoding, decimal selection, confidence, acceptance, expected ranges, or transition handling.

## Manual initialization contract

Manual once-per-video initialization is expected.

`reference_box` is a padded axis-aligned carrier containing the complete physical LCD plus visible margin. Its edges are not physical LCD boundaries.

`quad` is the four actual physical LCD corners inside `reference_box`, ordered:

```text
top-left -> top-right -> bottom-right -> bottom-left
```

For RDS-200, `grid` is a tight rectangle around all three large digits in the rectified display. Do not include the scale, `uSv/h`, or bezel. Decimal dots do not need to be included.

For RDS-30, digit geometry is profile-owned and no manual grid is used.

### Hard per-video geometry invariant

`reference_time_s`, `reference_box`, `quad`, and (for RDS-200) `grid` are per-video initialization/calibration.

In particular:

- never copy `reference_box`, `quad`, or `grid` from one video to another;
- select them independently from the image content of each video;
- once selected for a canonical video, freeze and reuse that exact geometry for baseline/refined algorithm comparisons;
- do not change geometry and OCR behavior simultaneously unless the experiment explicitly concerns initialization.

The canonical frozen values belong in `tests/video_manifest.tsv`.

## Geometry / debug invariants

The successful stabilization model remains:

1. choose a reference frame;
2. detect static LK features outside changing LCD content;
3. track sequentially frame-to-frame with pyramidal LK;
4. estimate a RANSAC similarity transform;
5. accumulate transforms relative to the reference frame;
6. stabilize the full frame;
7. crop the fixed `reference_box`;
8. perspective-warp the physical-LCD `quad`;
9. apply fixed digit geometry;
10. cache the exact rectified display used by OCR.

Debug images and diagnostics must use the cached main-pass display. Do not recompute geometry for debug output.

## Canonical video roles

The authoritative list and per-video initialization are in `tests/video_manifest.tsv`.

RDS-200:

```text
IMG_0747.mov   Condition A   dev
IMG_0744.mov   Condition A   verify
IMG_1151.MOV   Condition B   cross_condition_regression
```

`IMG_0744.mov` is verification evidence, not a new tuning set.

`IMG_1151.MOV` has been used extensively in historical development and is not a clean held-out video.

RDS-30 roles remain recorded in the manifest.

---

# RDS-30 public beta

## Frozen configuration

The algorithmic RDS-30 freeze is tagged:

```text
rds30-gi-weak9-2026-09-11
```

at commit `7bc6fde`.

Frozen behavior:

```text
geometry emission:          glyph-independent
confidence threshold:       0.358
weak-9 margin threshold:    0.400
```

Typical beta run:

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

Do not tune the frozen thresholds on an external validation video after inspecting its OCR output.

## Independent RDS-30 validation

`IMG_0751.MOV` is the first independent validation of the complete frozen RDS-30 configuration. Its ground truth was prepared before OCR, and geometry/decoder parameters were not tuned from the video.

Stable-sample result:

```text
stable samples:        1088
accepted / recognized: 1087
correct accepted:      1087
wrong accepted:           0
missing / rejected:       1
accepted accuracy:     100.00%
stable coverage:        99.91%
MAE on accepted:         0
```

This is evidence from one independent video, not a universal performance guarantee.

The next useful RDS-30 test remains a genuinely new external video under different recording conditions with frozen decoder settings.

---

# RDS-200 historical regression invariant

The historical RDS-200 exact regression and the newer OCR experiment below are different things and must not be conflated.

For behavior-preserving refactors, the historical committed baseline remains authoritative:

```text
expected samples: 331
actual samples:   331
required mismatches: 0
```

Use `tests/compare_regression_baseline.py` and the committed baseline under `tests/baseline/`.

Ground-truth accuracy does not replace this exact raw-output regression requirement.

The older historical benchmark is approximately:

```text
stable samples: 283
exact accuracy: 78.45%
MAE:            ~0.10396
```

---

# RDS-200 tight-segment / pattern-refinement experiment

## Source status

As of 2026-09-16, the tested opt-in flag

```text
--rds200-pattern-refinement
```

is not present on repository `main`.

The experiment was run from the local `dosimeter_get_values_flow_tightsegments.py` implementation. Treat the results below as validated local experiment state, not as already merged production behavior.

Before merging or recreating the implementation, preserve default behavior and rerun the exact historical RDS-200 regression.

## Frozen decoder / segment configuration

The development choice was frozen on `IMG_0747.mov` before verification on `IMG_0744.mov` and cross-condition testing on `IMG_1151.MOV`.

Common parameters:

```text
flow model:             similarity
flow redetect every:    1
segment y stretch:      1.45
segment x stretch:      1.0
segment x shift:        +4.0
digit x offsets:        0,0,+3
filter window:          1
mode window:            1
decimal places:         auto
contrast:               auto
OCR sample rate:        5 fps
```

The refined run differs from baseline only by adding:

```text
--rds200-pattern-refinement
```

All baseline/refined comparisons for a given video use identical frozen per-video geometry.

## Frozen per-video initialization

Exact values are stored in `tests/video_manifest.tsv`.

For reference:

```text
IMG_0747.mov
  reference_time_s = 20.0
  reference_box    = 0.396296,0.433333,0.687037,0.542708
  quad             = 0.042683,0.064789,0.951219,0.061972,0.965447,0.943662,0.042683,0.949296
  grid             = 0.239351,0.455056,0.679513,0.792135

IMG_0744.mov
  reference_time_s = 16.0
  reference_box    = 0.350000,0.340625,0.674074,0.463542
  quad             = 0.038618,0.076056,0.957317,0.070423,0.963415,0.946479,0.044715,0.952113
  grid             = 0.255578,0.463483,0.679513,0.789326

IMG_1151.MOV
  reference_time_s = 25.0
  reference_box    = 0.279630,0.240625,0.712963,0.404167
  quad             = 0.048780,0.092958,0.951219,0.081690,0.969512,0.940845,0.056911,0.957747
  grid             = 0.245436,0.471910,0.677485,0.786517
```

These values are video-specific. Do not transfer them between videos.

## Evaluation protocol

Evaluator:

```text
tests/evaluate_dosimeter_ground_truth.py
```

Default transition guard:

```text
+/- 0.400 s
```

Report at least:

- evaluated stable samples;
- recognized stable samples / coverage;
- correct, wrong, and missing stable samples;
- exact accuracy;
- accuracy among recognized samples;
- row-by-row status transitions between candidate algorithms.

MAE is secondary for character OCR. A one-digit positional error can dominate MAE even when the character-level change is informative.

## IMG_0747.mov — development

Stable samples: `159`.

Baseline (`digit3_xplus3`):

```text
recognized:               158 / 159 = 99.37%
correct:                   153
wrong:                       5
missing:                     1
overall exact accuracy:    96.23%
accuracy when recognized:  96.84%
```

With pattern refinement:

```text
recognized:               159 / 159 = 100.00%
correct:                   156
wrong:                       3
missing:                     0
overall exact accuracy:    98.11%
accuracy when recognized:  98.11%
```

Exact stable status transitions:

```text
correct -> correct: 153
wrong   -> correct:   2
wrong   -> wrong:     3
missing -> correct:   1
correct -> wrong:     0
correct -> missing:   0
```

The refinement repaired the second-digit `4 -> 8` failure in the problematic `18.x` family.

Remaining stable errors are a separate first-digit problem:

```text
t=35.2 s: truth 18.0 -> 38.0
t=35.4 s: truth 18.0 -> 38.0
t=36.8 s: truth 18.1 -> 98.1
```

Do not extend the current refinement merely to fix this separate failure mode without a new experiment.

## IMG_0744.mov — same-condition verification

Geometry was selected independently from this video before truth evaluation. No decoder tuning was performed on the verification result.

Stable samples: `164`.

Baseline:

```text
recognized:               145 / 164 = 88.41%
correct:                   140
wrong:                       5
missing:                    19
overall exact accuracy:    85.37%
accuracy when recognized:  96.55%
```

With pattern refinement:

```text
recognized:               146 / 164 = 89.02%
correct:                   141
wrong:                       5
missing:                    18
overall exact accuracy:    85.98%
accuracy when recognized:  96.58%
```

Exact stable status transitions:

```text
correct -> correct: 140
wrong   -> wrong:     5
missing -> missing:  18
missing -> correct:   1
correct -> wrong:     0
correct -> missing:   0
```

Only one stable prediction changed: at `t=14.6 s`, `missing -> 33.8`, which is correct.

Two prediction changes occurred inside the transition guard and do not affect stable accuracy:

```text
t=14.2 s: missing -> 33.8
t=63.2 s: missing -> 67.6
```

The latter shows that refinement is not intrinsically monotonic outside stable evaluation regions.

Known `IMG_0744` failure modes left untouched by this refinement include:

- early `33.x/34.x -> 2.8x` value failures;
- a long missing region for true `31.3`.

Treat these as separate problems.

## IMG_1151.MOV — cross-condition regression

Geometry was reselected independently for this video.

Tracking remained strong:

```text
frames:              331
accepted flow steps: 329 / 330
rejected flow steps:   1
mean tracked points:  ~78
mean RANSAC inliers:  ~77.8
```

### Automatic-decimal failure

With `--decimal-places auto`, the decoder selected one decimal place where the truth uses two.

Typical errors are therefore:

```text
truth 0.42 -> OCR 4.2
truth 0.69 -> OCR 6.9
```

Consequently, the uncorrected evaluator reports `0.00%` exact accuracy for both baseline and refined runs.

This is a decimal-state failure, not evidence that all digit recognition failed.

Do not use ground truth or expected dose range to choose decimal position in production. Fix/test automatic decimal handling as a separate OCR experiment.

### Digit/refinement comparison with decimal state analytically factored out

For diagnostic analysis only, dividing the recognized OCR values by 10 isolates the digit result. This normalization is evaluation analysis, not production logic.

Stable samples: `283`.

Baseline after analytical decimal normalization:

```text
recognized:               257 / 283 = 90.81%
correct:                   255
wrong:                       2
missing:                    26
exact accuracy:            90.11%
accuracy when recognized:  99.22%
```

Refined after analytical decimal normalization:

```text
recognized:               260 / 283 = 91.87%
correct:                   258
wrong:                       2
missing:                    23
exact accuracy:            91.17%
accuracy when recognized:  99.23%
```

Only four predictions changed in the raw 331-frame comparison:

```text
t=19.6 s: missing -> 4.2   transition guard
t=20.6 s: missing -> 4.8   stable; digit value corresponds to truth 0.48 after decimal normalization
t=33.0 s: missing -> 5.4   stable; corresponds to truth 0.54
t=38.4 s: missing -> 5.6   stable; corresponds to truth 0.56
```

The two remaining stable digit errors are unchanged in both runs:

```text
t=24.0 s: truth 0.61 -> OCR 16.1
t=24.8 s: truth 0.61 -> OCR 16.1
```

After the decimal factor is separated, these are first-digit errors (`1.61` vs `0.61`).

`IMG_1151.MOV` is a cross-condition regression case, not a clean held-out test.

## Confidence side effect of pattern refinement

The current refinement affects confidence much more broadly than it changes decoded values.

Observed confidence changes:

```text
IMG_0747: 117 / 332 frames changed; 117 up, 0 down
IMG_0744: 181 / 324 frames changed; 181 up, 0 down
IMG_1151: 159 / 331 frames changed; 159 up, 0 down
```

On stable samples whose prediction itself did not change:

```text
IMG_0747: 55 confidence changes; median +0.0856; mean +0.1063
IMG_0744: 89 confidence changes; median +0.0787; mean +0.0851
IMG_1151: 127 confidence changes; median +0.0685; mean +0.0943
```

No measured confidence change was downward.

This did not create a problem with `--mode-window 1`, but it is a real semantic side effect. Re-evaluate it before using the confidence values for stronger downstream voting, rejection, or weighting.

## RDS-200 experiment conclusion / freeze

Across the three-video experiment, no stable digit regression attributable to pattern refinement was observed:

```text
IMG_0747 dev:      +3 correct, wrong 5 -> 3, missing 1 -> 0
IMG_0744 verify:   +1 correct, wrong unchanged, missing 19 -> 18
IMG_1151 cross:    +3 digit-correct after decimal normalization,
                   wrong unchanged, missing 26 -> 23
```

The development choice is therefore frozen for this experiment.

Do not tune the refinement further on `IMG_0744.mov` or `IMG_1151.MOV`.

Separate future work items are:

1. RDS-200 automatic decimal handling under cross-condition data;
2. remaining first-digit failure modes;
3. confidence semantics if confidence becomes a downstream decision variable;
4. merging/reimplementing the opt-in refinement on top of current `main` while preserving default historical regression behavior.

---

# Testing / development discipline

Keep these concerns separate:

- behavior-preserving refactor;
- geometry/stabilization experiment;
- OCR-quality experiment;
- decimal-handling experiment;
- profile-generalization change.

For behavior-preserving code changes, normally require:

```text
py_compile
git diff --check
exact historical RDS-200 regression
relevant diagnostics / debug comparison
git status
```

Never compensate for poor geometry by tuning decoder rules.

Do not reintroduce without new evidence:

- expected-value or allowed-range constraints;
- video-specific correction tables;
- per-frame manual geometry;
- ECC/homography/template tracking approaches already rejected;
- digit-specific one-off patches;
- debug geometry recomputation.

## Current next steps

RDS-30:

- external beta validation on a genuinely new video with frozen parameters.

RDS-200:

- keep the 2026-09-16 pattern-refinement experiment frozen;
- treat `IMG_0744.mov` as consumed verification evidence, not a tuning set;
- treat `IMG_1151.MOV` as cross-condition regression, not held-out;
- investigate automatic decimal selection separately before interpreting raw cross-condition exact-value accuracy;
- if the refinement is to be merged, first port it cleanly to current `main`, preserve default behavior, and rerun the exact historical regression plus the frozen three-video OCR comparison.
