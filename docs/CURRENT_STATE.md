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

Evaluation ignores +/-0.4 s aroundEvaluation ignores +/-0.4 s aroundphEvaluation ignores +/-0.iefly shows trEvaluation ignores +/-0.4 s aroundEvaluation ignores +/-0.4 s aroundphEvpiEvaluation ignores +/-0.4 s aroundEvaluation ignores +/-0.4 s arou
VaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVable rVaVaVaVaVaVaVaVaVa-----VaVaVaVaVaVaVaV----VaVaVaVaVaVaVaVaVaVaVaVaVaVa  VaVaVaVaVaVaVaVaVa     VaVaVaVaVaVaVaVaht8         VaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVaVigiVaVaVaVaVaVaVaVh
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

Do not hard-code the first 1.4 seconds asDo not hard-c# Do no0.61, rougDo not hard-c8 s

Intermittent third-digit errors:

0.61 -> 0.62
0.61 -> 0.82

The measured segment pattern for the real digit 1 often contains
spurious middle/top/bottom evidence and a badly measured lower-right
segment.

### True 0.69, roughly 54.3--62.1 s

ConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConerConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiCondiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsiConsi seConsiConsiCotor.


onsiConsiConsiConst currently measures:

segmesegmesegmesegmesegmesegme insisegmesegmesegmesegmesegmesegme insisegmesegmesegmesegmesegmesegme insg asegmesegmesegmesegmesegmesegme ackground_level - segment_level

These values are currently hard-coded.

fixed_extract_darkness() accepts `segment_percentile` but discards it.

Therefore earlier attempts to sweep the original core
segment_percentile did NOT actually change the active RDS-200
fixed-grid extraction.

This is the next clean experimental target after architecture cleanup.


his is the next clean experimental target after architecture cleanabhis ision:his is the next clean experimentalay his is the next clean experimental target after architecture cleanabhis ision:his is the next clean experimentalay his is the next clean experimental target after architecture cleanabhis ision:his is the next clean experimentalay his is the next clean experimental target after architecture cleanabhis ision:his is the next clean experimentalay his is the next clean      his is the next clean experimental tked phis is the next clean experimental target after architecture cleanabhis ision:his is the next clean experimentalay his is the next clean experimental target after archi.


is is the next clean experimental tthis cache.

Do not reintroduce a separate geometry pass for debug generatioDo not reintroduce a separate geometry pass for debthoDo not reintroduce a separate geometry pass for debug gener maDo not reintroduce a pective + adaptive y shift
- fixed grid without tracking
- ECC homography
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ing
    -> interval construction
    -> diagnostics/output

The ground-truth evaluator should remain independent of production OCR.
