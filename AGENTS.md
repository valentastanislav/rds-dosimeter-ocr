# AGENTS.md

## Project

OCR pipeline for reading changing seven-segment dose-rate values from
RADOS dosimeter videos.

Supported / target dosimeter profiles:

- RDS-200
- RDS-30

The project goal is a shared video-processing and stabilization pipeline
with profile-specific display geometry, measurement, decimal handling,
and decoding where required.

The code should work on different hand-held videos of the same dosimeter
type under reasonably different framing, perspective, scale, motion, and
lighting conditions.

Fully unattended processing is NOT a requirement.

A normal production workflow may ask the user once per video to choose a
suitable reference frame and manually mark display geometry. After that,
the same stabilization/tracking pipeline should process the complete video
forward and backward from the selected reference.

Read `docs/CURRENT_STATE.md` before making architectural changes.


## Current development phase

The behavior-preserving cleanup/refactor is complete.

Important dependencies are now passed explicitly through the production
pipeline rather than installed by runtime monkey-patching.

The shared active architecture is approximately:

    flow
      -> fixedgrid
      -> ROI
      -> core measurement / filtering / decoding

with explicit dependencies for:

- Profile
- display finder / geometry
- darkness extraction
- decoder
- debug writer
- decimal-sequence observation

Do not reintroduce hidden imported-module mutation, `sys.argv` rewriting
between pipeline layers, or shared mutable run state.

Diagnostic-only tools such as `flow_diag` may temporarily use isolated
instrumentation hooks, but production behavior must not depend on them.

Generalization work has started. The shared fixed-grid/flow path now supports
RDS-30 with profile-owned measurement and digit geometry. Optical-flow
changing-display exclusion geometry is also profile-aware.


## Core design principle

Prefer:

    shared geometry / stabilization
    + explicit profile-specific OCR policy

over:

    separate RDS-200 and RDS-30 pipelines

There must be only one implementation of:

- reference-frame tracking
- sequential pyramidal LK
- RANSAC similarity estimation
- transform accumulation
- full-frame stabilization
- reference display cropping
- perspective rectification
- display caching
- common interval/output plumbing

Device-specific behavior belongs in `Profile` or in an explicit
profile-specific strategy/configuration.

Do not introduce RDS-200-specific assumptions into shared stabilization
code merely because the current historical regression video is RDS-200.


## Product-level initialization policy

Manual once-per-video initialization is acceptable and expected.

It is acceptable for the user to provide or interactively determine:

- reference time / reference frame
- `reference_box`: a padded carrier rectangle containing the complete
  physical LCD plus a visible margin; its edges are not physical LCD
  boundaries
- perspective quad
- digit grid if required by the profile

The perspective `quad` marks the four actual physical LCD corners inside
`reference_box`, ordered top-left, top-right, bottom-right, bottom-left.

RDS-30 uses profile-owned digit geometry and does not use a manual digit
grid. RDS-200 retains its existing manual-grid behavior for now.

These values may be stored and reused for reproducible tests.

Do NOT spend development effort on fully automatic display detection or
automatic reference-frame selection unless explicitly requested later.

The important generalization target is:

> Given a sensible one-time manual initialization, the same tracking and
> stabilization implementation should process the complete video robustly
> for both RDS-200 and RDS-30.

The user should not need to manually correct geometry frame by frame.


## Geometry and stabilization invariants

The successful flow approach is:

1. choose/detect the reference display at a selected reference time;
2. detect static LK features outside changing LCD content;
3. track sequentially frame-to-frame with pyramidal LK;
4. estimate a RANSAC similarity transform for each step;
5. accumulate transforms relative to the reference frame;
6. stabilize the full frame;
7. crop the fixed reference display;
8. apply the fixed perspective quad;
9. use the fixed digit geometry;
10. cache the exact rectified display used by OCR.

Preserve this model unless an experiment explicitly demonstrates a better
general solution.

Do not casually replace it with:

- per-frame digit rediscovery
- ECC
- homography tracking
- template tracking of changing digit segments
- adaptive per-frame digit-grid movement

Moderate translation, rotation, and scale changes are intended to be
handled by the current similarity-transform tracking.

Extreme shake, severe motion blur, complete loss of the dosimeter, or the
display leaving the image need not be supported initially.


## Tracking feature policy

Changing LCD digits must not be used as nominal static LK tracking
features.

The exclusion region is profile geometry:

- RDS-200:
  `flow_feature_exclusion_box = (0.23, 0.40, 0.78, 0.86)`

- RDS-30:
  `flow_feature_exclusion_box = (0.33, 0.34, 0.98, 0.80)`

Coordinates are normalized to the detected/reference display box.

Keep one shared feature-mask implementation.

Do not introduce `profile.name == ...` branches into the optical-flow
algorithm when the difference can be represented declaratively by Profile.


## Profile responsibilities

Profile-level configuration may legitimately include:

- canonical display dimensions
- digit boxes
- segment polygons
- digit count
- decimal candidates or fixed decimal places
- expected display aspect
- flow feature exclusion geometry
- sample rate
- temporal filtering defaults
- digit patterns
- decoder selection
- auxiliary measurement percentile
- leading blank behavior
- mode-window / interval policies
- measurement or decoder strategy identifiers/configuration

Current defaults include:

- RDS-200 OCR sampling: 5 fps
- RDS-30 OCR sampling: 30 fps

These are OCR processing rates associated with the display/profile, not
the native frame rate of the source video.

Be careful with parameters expressed in sample counts. For example,
10 samples means:

- 2.0 s at 5 fps
- 0.333 s at 30 fps

Do not change sample-count-based tracking constants merely to make them
look symmetric between profiles. First determine whether their intended
meaning is frame-based or time-based.


## RDS-200 behavior to preserve

The established RDS-200 production experiment uses:

- three digits
- moving decimal point
- fixed-grid digit geometry
- local segment/background measurement
- segment percentile p35
- background percentile p70
- raw / CLAHE / auto contrast selection
- video-wide binary consensus calibration
- temporal median filtering
- `--mode-window 1`

The successful consensus tight-segment geometry currently corresponds to
the established y-stretch behavior.

Do not make old ignored percentile arguments suddenly effective during
refactoring.

Do not introduce digit-specific patches such as special-case 6 -> 8 rules.

Preserve the existing decoder quirks and automatic-decimal behavior until
they are changed by a separately tested OCR experiment.


## RDS-30 behavior to preserve while generalizing

RDS-30 has genuinely different OCR behavior and should not be forced
through the RDS-200 decoder.

Current RDS-30 characteristics include:

- four digit positions
- custom digit patterns
- fixed two decimal places
- `rds30_margin` decoder
- primary segment measurement currently equivalent to p50 against p90
- auxiliary p30 measurement used by the current decoder
- profile-specific temporal / interval behavior
- leading blank behavior

During generalization, first preserve this existing numerical behavior
explicitly.

Do not tune RDS-30 percentiles, decoder thresholds, prototypes, adaptive
alignment, or glyph logic in the same change that only makes the shared
pipeline profile-neutral.

Architecture changes and OCR-quality experiments should remain separate.


## Decimal handling

Preserve support for automatic decimal-point detection.

`--decimal-places 2` is a controlled RDS-200 regression setting only; it
does not exercise automatic decimal detection.

RDS-200 automatic decimal handling must be separately tested after any
change touching decoding or decimal logic.

RDS-30 currently uses fixed decimal places.

Ground-truth values or expected dose ranges must NEVER be used to choose a
decimal position in production code.


## Debug invariant

This is critical:

> Flow debug images must represent the exact stabilized and rectified
> pixels used during the main OCR pass.

The flow pipeline maintains a `display_cache` populated during actual
decoding.

Flow debug output must read from this cache.

Never re-run or independently reconstruct optical-flow geometry merely to
produce debug screenshots.

For flow debug output, preserve:

    debug_source = cached_main_pass

Whenever a change touches geometry, debug routing, caching, or display
finders, compare pre/post debug trees where practical.


## Ground truth policy

Ground-truth dose values are evaluation-only.

Never use them in production OCR logic.

Specifically, do not encode:

- expected values
- allowed value ranges
- known transition times
- expected first/last interval values
- video-specific correction tables

Ground truth may be used for:

- regression evaluation
- error analysis
- comparing candidate algorithms
- final OCR accuracy metrics

Geometry and tracking should be evaluated from image/tracking evidence
rather than dose truth whenever possible.


## Canonical local video test set

The canonical local-video selection is defined in:

    tests/video_manifest.tsv

Video files themselves are local and intentionally not committed.

Canonical assignments are:

### RDS-200

Development, Condition A:

    IMG_0747.mov

Verification under similar conditions:

    IMG_0744.mov

Cross-condition regression, Condition B:

    IMG_1151.MOV

`IMG_1151.MOV` / the derived historical IMG_1151 test material has already
been used extensively during development. It is therefore useful as a
cross-condition regression case, but it is NOT a statistically clean
held-out video.

### RDS-30

Development, Condition A:

    IMG_0754.MOV

Verification under similar conditions:

    IMG_0753.MOV

Originally held-out different conditions, Condition B:

    IMG_1152.MOV

`IMG_0754.MOV` is the RDS-30 Condition A development video and was used
during tracking development. The validated RDS-30 canonical geometry was
calibrated only on this video, then frozen before successful verification
on the Condition A verification video, `IMG_0753.MOV`.

`IMG_1152.MOV` exposed the old coordinate-contract problem and was consumed
during geometry acceptance. It must not be used for further tuning or
described as a clean held-out test. A future genuinely held-out RDS-30 test
requires a new video.

Do not arbitrarily substitute another local video when a canonical
manifest entry exists.


## Video-test roles

Use the manifest roles consistently.

### `dev`

May be:

- inspected freely
- visualized
- used for diagnostics
- used to choose/tune a general parameter or strategy

Current development videos:

- RDS-200: `IMG_0747.mov`
- RDS-30: `IMG_0754.MOV`

### `verify`

Run after a development choice has been made.

Do not repeatedly tune a parameter specifically to improve this individual
video.

Current verification videos:

- RDS-200: `IMG_0744.mov`
- RDS-30: `IMG_0753.MOV`

### `consumed_heldout`

This role records a video that was originally held out but has since been
used during development or acceptance work.

It must not be used for further tuning or claimed as a clean held-out test.

Current consumed held-out video:

- RDS-30: `IMG_1152.MOV`

There is currently no genuinely held-out RDS-30 video. A new video is
required for future held-out evaluation.

### `cross_condition_regression`

Useful for detecting breakage under different conditions, but not a true
held-out test if already used during prior development.

Current case:

- RDS-200: `IMG_1151.MOV`


## Manual geometry in video tests

`tests/video_manifest.tsv` may store:

- `reference_time_s`
- `reference_box`
- `roi`
- `quad`
- `grid`

Per-video initialization coordinates belong in `tests/video_manifest.tsv`,
not in this document.

The current manual convention is:

- `reference_box` is a padded carrier rectangle containing the complete
  physical LCD plus visible margin. Its edges are not the LCD boundaries.
- `quad` contains the four physical LCD corners inside that carrier in
  top-left, top-right, bottom-right, bottom-left order.
- RDS-30 digit geometry is profile-owned and requires no manual grid.
- RDS-200 keeps its existing manual-grid behavior for now.

Once a manual initialization has been selected for a canonical test video,
reuse it for algorithm regression tests.

Do not change manual geometry and algorithm behavior simultaneously unless
the experiment explicitly concerns initialization.

This separation is important:

    changed OCR/tracking result
        should mean
    changed algorithm

not:

    changed where the user clicked


## Historical RDS-200 regression

The historical exact consensus2 regression must remain available during
generalization work.

For the established IMG_1151-derived regression:

- expected samples: 331
- actual samples: 331
- required mismatches for behavior-preserving changes: 0

Historical consensus diagnostics include approximately:

    OFF level          1.000
    OFF robust sigma   1.483
    high population   43.250
    separation        42.250
    ACTIVE threshold   8.413
    cluster sizes      2374 / 4513
    digit calls        993
    exact patterns     719
    agreement used     683
    generic overrides  18
    ambiguous binary   281
    baseline used      292

The exact baseline comparator is:

    tests/compare_regression_baseline.py

The committed baseline is under:

    tests/baseline/

For behavior-preserving refactors, require exact parity.

Do not replace exact raw regression with ground-truth accuracy alone.


## Testing strategy

Keep these concerns separate.

### 1. Behavior-preserving regression

For refactors:

- run `py_compile`
- run `git diff --check`
- run exact RDS-200 historical regression
- compare debug output when relevant
- require exact parity

### 2. Geometry / stabilization generalization

Dose-value truth is not required.

Inspect:

- reference acquisition
- feature count
- feature spatial distribution
- features inside changing LCD region
- mean tracked points
- mean RANSAC inliers
- accepted / rejected transform steps
- redetections
- transform continuity
- cumulative drift
- stabilized display/grid alignment
- display-found fraction
- cached-main-pass identity

### 3. OCR-quality experiments

Only after geometry is sufficiently stable, evaluate:

- recognized fraction
- exact-value accuracy
- interval accuracy
- MAE where meaningful
- error types by digit/interval
- decimal decisions

Do not compensate for bad geometry by tuning decoder rules.


## Development discipline

Make small, separately testable changes.

Do not combine:

- architectural refactoring
- geometry tuning
- measurement tuning
- decoder tuning
- decimal tuning

into one change.

For each step, state whether it is:

- behavior-preserving refactor
- geometry/stabilization experiment
- profile-generalization change
- OCR-quality experiment

Before changing behavior, establish the relevant pre-change baseline.

Prefer direct function arguments, explicit Profile configuration, and
per-run state over:

- mutable module globals
- monkey-patching imported modules
- hidden alias replacement
- temporary `sys.argv` rewriting

Do not silently reintroduce mechanisms removed during cleanup.


## Failed/deprecated approaches

Do not revisit these without new evidence:

- auto ROI/manual ROI alone as a substitute for stabilization
- perspective rectification plus adaptive y-shift without tracking
- fixed grid without tracking
- ECC homography / rigid / translation approaches
- four-corner template tracking
- RDS-200 `track-time 2.2` with the historical geometry
- digit-specific 6 -> 8 correction
- per-position segment clustering
- RDS-200 `mode-window 5`
- adaptive-y correction as a decoder fix
- debug geometry recomputation
- expected-value/range constraints
- hard-coding the first interval invalid
- arbitrary global polygon tweaking
- monkey-patching obsolete/inactive percentile hooks


## Source-code changes

For substantial code changes, prefer a focused, minimal implementation
rather than unrelated cleanup.

Do not commit automatically unless explicitly requested.

Before presenting a completed change, normally show:

- diff
- `py_compile` result
- `git diff --check`
- relevant regression result
- relevant diagnostic comparison
- `git status`

Do not fix unrelated known quirks while performing a scoped refactor or
generalization step.


## Documentation

Keep these files aligned with the current architecture:

    README.md
    docs/CURRENT_STATE.md
    tests/video_manifest.tsv
    AGENTS.md

Update documentation when a development milestone changes the intended
pipeline or canonical test procedure.

Do not let historical experimental behavior become undocumented production
assumptions.
