# Current development state

## Scope

The repository contains beta OCR pipelines for two RADOS instruments:

- RDS-30
- RDS-200

Both use the same primary entry point:

```bash
python3 dosimeter_get_values_flow.py VIDEO.MOV intervals.csv [options]
```

Ground truth is evaluation-only. It must never influence production tracking, geometry, decimal selection, confidence, acceptance, expected ranges, or transition handling.

## Shared processing architecture

```text
video
  -> reference-frame geometry
  -> optical-flow stabilization
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

Debug images must come from the cached main-pass display used by OCR. Debug geometry must be the same geometry used by the decoder.

## Manual initialization contract

`reference_box` is a loose carrier around the complete physical LCD/display plus a small visible margin.

`quad` is the four physical LCD corners inside the reference box, ordered:

```text
top-left -> top-right -> bottom-right -> bottom-left
```

For RDS-200, `grid` is a tight rectangle around all three large digits in the rectified display.

For RDS-30, digit geometry is profile-owned and no manual grid is used.

All geometry is per-video. Do not transfer `reference_box`, `quad`, or RDS-200 `grid` between recordings without independently verifying them.

---

# RDS-30 beta

The frozen RDS-30 algorithm is tagged:

```text
rds30-gi-weak9-2026-09-11
```

Frozen decoder behavior:

```text
geometry emission:          glyph-independent
confidence threshold:       0.358
weak-9 margin threshold:    0.400
```

Typical run:

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

Independent validation on `IMG_0751.MOV`, with ground truth prepared before OCR:

```text
stable samples:        1088
accepted / recognized: 1087
correct accepted:      1087
wrong accepted:           0
stable coverage:        99.91 %
accepted accuracy:     100.00 %
```

This is evidence from one independent recording, not a universal performance guarantee.

---

# RDS-200 beta

The RDS-200 experimental tracking/mask stack has been promoted into the normal production path. Beta users should run `dosimeter_get_values_flow.py` directly.

## Production defaults

```text
tracking feature region    static ring outside reference box
ring padding               0.25 of reference-box size
registration               direct reference -> frame
fallback                   cumulative frame-to-frame transform
max translation            150 px
max rotation               10 deg
scale range                0.85 .. 1.15

digit geometry             profile-owned per-digit polygons
decimal geometry           derived from final digit boxes/bottom segment
pattern refinement         enabled
sample rate                5 Hz
filter window              1
mode window                1
raw min confidence         0.20
summary min confidence     0.20
```

Typical first run:

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

The decimal candidates are part of the final digit geometry:

- decimal position 2 is centred between digit 1 and digit 2;
- decimal position 1 is centred between digit 2 and digit 3;
- their lower edge follows the transformed bottom segment.

## Confidence semantics

`--min-confidence` acts on raw OCR samples before interval reconstruction. The RDS-200 default is `0.20`.

`--summary-min-confidence` acts only on final intervals when calculating time-weighted mean, variance, standard deviation, minimum, and maximum. The default is also `0.20`.

Intervals below the summary threshold remain in the interval CSV and debug output. The summary reports how much duration was included and excluded.

## Current RDS-200 regression evidence

Current beta production runs were checked on the following development/verification recordings:

### IMG_0742

```text
stable evaluated: 158
correct:          158
wrong:              0
missing:            0
exact accuracy: 100.00 %
MAE:               0
```

### IMG_0744

```text
stable evaluated: 164
correct:          156
wrong:              7
missing:            1
overall exact:    95.12 %
recognized exact: 95.71 %
MAE:             0.028221
```

The remaining mistakes are concentrated in a few blurred/low-contrast regions. Do not tune the production masks further against this recording.

### IMG_0747

```text
stable evaluated: 159
correct:          148
wrong:              7
missing:            4
overall exact:    93.08 %
recognized exact: 95.48 %
```

The largest OCR outliers have low final interval confidence. With the default summary confidence cut of `0.20`, the tested production run used 97.89% of the video duration for physical statistics and excluded 2.11%.

### IMG_1151

This recording provides cross-condition coverage and exercises the alternative RDS-200 decimal position. The final beta geometry/tracking run was visually stable and the automatic decimal geometry behaved consistently. Treat it as consumed development/regression data, not a clean held-out validation set.

## Development freeze

Do not continue tuning the RDS-200 production geometry or decoder against `IMG_0742`, `IMG_0744`, `IMG_0747`, or `IMG_1151`.

New algorithm changes should be motivated by genuinely new beta recordings or by a clearly isolated reproducible bug.

---

# Historical development helpers

The obsolete user-facing RDS-200 wrapper stack has been removed after promotion of the validated behavior into the production pipeline.

Two development helpers remain because the test suite still uses them:

- `dosimeter_get_values_flow_diag.py`
- `dosimeter_get_values_flow_tightsegments.py`

The historical blind grid sweep remains under:

```text
tests/rds200_grid_refinement_sweep.py
```

These are development/evaluation tools, not beta user entry points.

---

# Testing and release discipline

Full unit test:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

At the current pre-merge beta checkpoint:

```text
Ran 43 tests
OK
```

Before merging the RDS-200 beta branch:

1. rerun the full unit-test suite;
2. inspect `python3 dosimeter_get_values_flow.py --help`;
3. inspect the branch diff for accidental experimental artifacts;
4. merge only after those checks remain clean;
5. create an RDS-200 beta tag only after the merged commit is known.

For future validation, prepare ground truth independently before inspecting OCR output whenever possible.

Never compensate for poor geometry with:

- expected-value or allowed-range constraints;
- video-specific correction tables;
- per-frame manual geometry;
- digit-specific one-off decoder patches.
