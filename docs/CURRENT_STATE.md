# Current development state

## Goal

Read changing RADOS RDS-200 seven-segment LCD values from hand-held video
robustly enough to generalize to other videos and displayed values.

Ground truth is used only for benchmarking.

## Current test video

IMG_1151edited.mov
duration approximately 66.1 s
sampling: 5 samples/s

Reference geometry:

- ROI:
  0.277704,0.221833,0.735259,0.388583
- perspective quad:
  0.018293,0.092958,0.912602,0.090141,0.916667,0.912676,0.014228,0.915493
- fixed grid:
  0.302231,0.438202,0.726166,0.786517
- optical-flow reference time:
  6.8 s
- tight segment y stretch:
  1.45 around y=67
- test decimal mode:
  2 decimal places
- RDS-200 value mode window:
  1

## Ground truth

tests/IMG_1151_true_values.txt

start_s   value

0.0       0.42
3.2       0.39
7.7       0.37
13.7      0.35
16.8      0.42
19.8      0.48
21.2      0.61
28.8      0.54
36.2      0.56
40.7      0.64
46.8      0.66
54.3      0.69
62.1      0.65

Evaluation ignores +/-0.4 s around real LCD transitions because the
physical display itself briefly shows transition values.

Do NOT ignore t=0; the early tracking error is a real pipeline problem.

## Raw benchmark

Stable evaluated samples: 283.

Variant             exact accuracy    notable result
------------------------------------------------------
tight               76.68 %           baseline
tight8              77.39 %           deprecated digit-specific patch
consensus           78.45 %           bad large 8.82 errors
consensus2          78.45 %           MAE improved to 0.10396

Consensus2 used:

--binary-off-sigma 5.0
--binary-off-margin 7.0
--binary-override-margin 2.5

Consensus2 removed the very large 8.82 failures but did not improve the
number of exactly correct samples.

## Main remaining failures

### 0.00--1.20 s

True: 0.42
OCR: 4.11

Cause is believed to be stabilization/geometry, not digit decoding.

Do not hard-code the first 1.4 seconds as invalid.

### True 0.61, roughly 21.2--28.8 s

Intermittent third-digit errors:

0.61 -> 0.62
0.61 -> 0.82

The measured segment pattern for the real digit 1 often contains
spurious middle/top/bottom evidence and a badly measured lower-right
segment.

### True 0.69, roughly 54.3--62.1 s

Consistently decoded as:

0.69 -> 0.65

For the third digit, the upper-right and lower-right segment evidence
collapses late in the video even though the physical glyph is 9.

This points strongly toward extraction/geometry rather than merely
decoder thresholds.

## Important extraction discovery

dosimeter_get_values_fixedgrid.py has its own local segment extractor.

For every segment it currently measures:

segment level:
35th percentile inside the segment mask

background level:
70th percentile in a local ring around the segment

darkness =
background_level - segment_level

These values are currently hard-coded.

fixed_extract_darkness() accepts `segment_percentile` but discards it.

Therefore earlier attempts to sweep the original core
segment_percentile did NOT actually change the active RDS-200
fixed-grid extraction.

This is the next clean experimental target after architecture cleanup.

## Current successful geometry strategy

Sequential optical-flow stabilization:

1. determine one reference display box
2. track static features with pyramidal Lucas-Kanade optical flow
3. estimate frame-to-frame similarity transform with RANSAC
4. accumulate transform to reference
5. stabilize frame
6. use fixed display crop
7. apply fixed perspective quad
8. apply fixed digit grid
9. decode

Typical diagnostics:

steps               330
accepted motion     329
rejected motion       1
point redetections   32
mean tracked pts    ~90
mean RANSAC inliers ~90

## Debug invariant

dosimeter_get_values_flow.py caches the stabilized + perspective-
rectified display during the MAIN decoding pass.

Debug JPGs must always be made from this cache.

Do not reintroduce a separate geometry pass for debug generation.

## Deprecated approaches

Do not reintroduce without new evidence:

- automatic ROI without robust tracking
- manual ROI alone
- perspective + adaptive y shift
- fixed grid without tracking
- ECC homography
- rigid ECC
- four-corner template tracking
- translation-only ECC
- track-time 2.2 with the current quad/grid
- digit-specific 6->8 correction
- independent per-position/per-segment unsupervised ON/OFF clustering
- RDS-200 mode-window 5
- hard-coded expected values
- hard-coded allowed dose ranges
- hard-coded removal of the first 1.4 s
- debug geometry recomputation

## Cleanup objective

Replace the current chain of runtime monkey-patched front-ends with
explicit components.

Desired conceptual pipeline:

video input
    -> geometry/stabilization
    -> rectified display
    -> digit geometry
    -> segment measurement
    -> temporal filtering
    -> digit decoding
    -> decimal decoding
    -> interval construction
    -> diagnostics/output

The ground-truth evaluator should remain independent of production OCR.