# Legacy experimental implementations

These files are preserved as development history only.

They are NOT part of the current OCR pipeline and should not be used as
the starting point for new development.

## Deprecated approaches represented here

- automatic/adaptive per-segment calibration
- digit-margin experiments predating the corrected tight-grid geometry
- specific digit corrections such as 6 -> 8
- ECC / rigid stabilization
- translation-only stabilization
- four-corner / quad tracking
- experimental percentile wrapper which did not reach the active
  fixed-grid local segment extractor

Some of these scripts may no longer run from this directory because
they retain imports corresponding to their original location. They are
historical snapshots, not maintained executables.

See ../docs/CURRENT_STATE.md for the current implementation and
benchmark state.
