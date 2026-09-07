# AGENTS.md

## Project

OCR pipeline for reading changing seven-segment values from RADOS
dosimeter videos, primarily RDS-200 and RDS-30.

The main current development target is robust RDS-200 recognition from
hand-held video.

## General rules

- Prefer a general solution over tuning to one known video.
- Ground-truth values are ONLY for evaluation and diagnostics.
- Never encode expected dose values or allowed value ranges into OCR logic.
- Avoid digit-specific corrections such as special-case 6->8 rules.
- Preserve support for automatic decimal-point detection in the final code.
- `--decimal-places 2` is only a controlled setting for the current test video.
- Preserve the cached-main-pass debug guarantee: debug images must represent
  the exact stabilized/rectified pixels used during decoding.
- RDS-200 currently uses `--mode-window 1`; larger value-mode windows can
  erase real short-lived displayed values.
- Before changing OCR behaviour, run the ground-truth evaluator and compare
  raw-sample accuracy.
- Do not silently reintr- Do not silently reintr- Do not silently reintr- Do not silentlE.- Do not silently reintr- Do not silently r
For IMG_1151edited.mov:

ROI:
0.277704,0.221833,0.735259,0.388583

Perspective quad:
0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.01gr0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.01829ce0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.018293,0.092958,0ect0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.018293,0.0ds0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.018293,0.092958,0.e.0.018293,0.092958,0.912602,0.018293,0.092958,0.912602,0.018293,0.092958,0.9126o turn the successful pieces into explicit,
testable components with normal function arguments/configuration.

Read docs/CURRENT_STATE.md before making architectural changes.
