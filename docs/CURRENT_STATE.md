# Current development state

## Goal

Read changing seven-segment LCD values from hand-held videos of RADOS dosimeters, with current external beta testing focused on the RDS-30.

Ground truth is used only for benchmarking and validation. It must not influence production geometry, decoding, confidence, or acceptance decisions.

## Current public beta status

The repository is public and `main` contains the current RDS-30 beta pipeline plus independent-validation metadata.

The algorithmic RDS-30 freeze is tagged:

```text
rds30-gi-weak9-2026-09-11
```

The tag points to commit `7bc6fde`, with:

```text
geometry emission: glyph-independent
confidence threshold: 0.358
weak-9 margin threshold: 0.400
```

The current `main` branch contains the same frozen algorithm plus later documentation and validation metadata.

A historical pre-cleanup snapshot is preserved as:

```text
pre-cleanup-2026-09-07
```

pointing to commit `eb003e7`.

## RDS-30 processing chain

The current RDS-30 path is:

```text
video
  -> manual reference display box
  -> sequential pyramidal Lucas-Kanade optical flow
  -> frame-to-frame RANSAC similarity transform
  -> accumulated stabilization to the reference frame
  -> fixed reference-box crop
  -> fixed perspective rectification
  -> profile-defined digit geometry
  -> segment evidence measurement
  -> temporal filtering
  -> joint-spatial residual-geometry selection
  -> digit decoding
  -> confidence / weak-9 rejection
  -> interval construction
  -> diagnostics / output
```

The key design rule is that residual geometry must be selected independently of digit identity. The current beta therefore uses `glyph-independent` geometry emission rather than the older glyph-best score.

## Manual geometry contract

### Reference box

The `reference_box` is a loose axis-aligned carrier rectangle containing the complete physical LCD/display plus a small visible margin.

Its edges have no LCD-boundary meaning.

### Quad

The perspective `quad` is the four actual physical LCD corners inside the reference box, clicked in this order:

```text
top-left -> top-right -> bottom-right -> bottom-left
```

The quad follows the LCD boundary, not the digits, housing, or reference-box edge.

### RDS-30 digit geometry

RDS-30 does not require a manually selected digit grid. Digit positions and seven-segment geometry are part of the profile, including the possibly blank leading digit.

## Frozen RDS-30 beta command

Typical interactive run:

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

For external beta tests, do not tune the frozen thresholds after looking at the OCR result.

## RDS-30 validation history

Video roles are recorded in `tests/video_manifest.tsv`.

### Development / consumed videos

`IMG_0754.MOV`
- development video
- corrected padded-carrier / physical-LCD-corner initialization

`IMG_0753.MOV`
- originally verification
- later consumed during robustness and confidence-threshold development
- no longer independent verification

`IMG_1152.MOV`
- originally held out
- later consumed during geometry acceptance, cross-condition validation, and confidence-threshold analysis
- not an independent held-out video

`IMG_0748.MOV`
- development video
- used to diagnose glyph-driven residual-geometry selection and develop glyph-independent geometry emission
- ground-truth interval 34 was corrected from a transcription error: the correct value is `78.50`, not `75.50`

`IMG_0749.MOV`
- initially attempted as frozen validation
- exposed systematic low-margin `4 -> 9` errors
- then consumed to develop the weak-9 margin guard
- therefore not an independent validation video

### Independent frozen validation

`IMG_0751.MOV` is the first independent validation of the complete frozen RDS-30 configuration.

Its ground truth was prepared before OCR. Geometry and decoder parameters were not tuned from this video.

Result on stable samples:

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

This is evidence that the frozen configuration can generalize beyond the development set, but it is one independent video and must not be presented as a universal performance guarantee.

The next useful test is an external video of the same RDS-30 instrument type under different recording conditions.

## Why glyph-independent geometry is important

The older joint-spatial implementation selected residual geometry using the best legal glyph likelihood at each candidate offset. That creates a circular dependency: geometry can move toward a location that makes a different legal digit look more plausible.

The current glyph-independent geometry emission uses only physical image evidence such as contrast, oriented-edge support, cross-edge penalty, and continuity. The geometry path is fixed first; only then are glyph probabilities and digit margins evaluated.

This separation is a core current invariant.

## Weak-9 guard

The RDS-30 LCD uses non-standard glyphs, including a `9` without the bottom segment.

After `IMG_0749.MOV` exposed low-margin `4 -> 9` confusions, a conservative guard was added:

```text
--joint-spatial-min-nine-margin 0.400
```

A sample is rejected when any decoded digit `9` has a glyph margin below this threshold. Other digits are unaffected.

The threshold was frozen before the independent `IMG_0751.MOV` validation.

## Confidence policy

The beta decoder is deliberately conservative. Rejected samples are preferable to accepted wrong values.

Therefore:

- coverage below 100% is not automatically a failure
- accepted accuracy is the primary reliability metric
- a modest rejection fraction can be acceptable
- external tests should report both coverage and wrong accepted values

## Debug invariant

`dosimeter_get_values_flow.py` caches the stabilized and perspective-rectified display used during the main decoding pass.

Debug images and diagnostics must be generated from this exact cached main-pass display.

Do not reintroduce a separate geometry or tracking pass for debug generation.

## RDS-200 regression invariant

RDS-200 remains supported and is treated as a regression invariant while RDS-30 development proceeds.

The historical consensus baseline contains 331 raw samples. Current refactoring has preserved exact output parity:

```text
expected samples: 331
actual samples:   331
mismatching:        0
```

The historical RDS-200 benchmark on stable evaluated samples remains approximately:

```text
stable samples: 283
exact accuracy: 78.45%
MAE:            ~0.10396
```

RDS-200 still uses a manual digit grid and should not be silently changed while working on the RDS-30 path.

## Tests and behavior-preservation requirements

Changes to the current code should preserve:

- default no-option behavior unless a new option is explicitly enabled
- RDS-200 331/331 regression parity
- exact cached-display debug invariant
- RDS-30 profile geometry
- ground-truth independence from production OCR
- frozen RDS-30 beta parameters unless a new development cycle is explicitly started

The default `glyph-best` mode is retained for behavior preservation when the new RDS-30 options are not requested.

## Approaches not to reintroduce without new evidence

Do not revisit these simply because the current beta is imperfect:

- automatic ROI as a replacement for robust stabilization
- manual ROI alone without tracking
- perspective correction plus adaptive-y without tracking
- fixed grid without tracking
- ECC homography / rigid / translation-only registration
- four-corner template tracking
- digit-specific `6 -> 8` patches
- hard-coded expected values or allowed dose ranges
- hard-coded removal of early video time ranges
- debug geometry recomputation
- independent per-segment spatial mask search that chases boundaries
- truth-glyph-margin joint registration as a physical geometry estimator
- banning all `ty = +4`, all `ty = -4`, or all nonzero vertical residual states

In particular, real data contain both correct and incorrect samples at nonzero residual-y states, so residual geometry cannot be filtered by a simple sign or nonzero-state rule.

## Current next step

The immediate development objective is not further tuning on the existing videos.

The useful next step is external beta validation:

1. use a new RDS-30 video under different conditions
2. keep the frozen decoder settings unchanged
3. record `reference_box` and `quad`
4. retain raw output and `geometry.csv`
5. if true values are available, record them independently before inspecting OCR output
6. treat any new failure as validation evidence first, not as an automatic cue to tune on that same video

## Conceptual architecture

The long-term architecture remains:

```text
video input
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

Ground-truth evaluation remains independent of production OCR.