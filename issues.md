# ISSUE-001

Status: Open

Area: Action labels (`curvature` metric)

## Observation

The geometric `curvature` progress is speed-independent inside segments of constant curvature, but the
chord window is `curvature_window` × the episode's median per-frame joint step, and that step scales
with the teleop speed. Where curvature changes (a straight joining an arc) the smoothing width, and so
the total progress, differs between recordings of the same path at different speeds.

## Impact

On the constructed test path (≈ 400 frames along the arc) the progress spread between three speed
profiles is 0.72% of the total; with half as many frames the spread between v and 2v is 1.07%. Real
demonstrations have curved segments of tens of frames, where the spread can reach a few percent. The
previous second-difference metric differed by ≈ 20% on the same circle at v and 2v.

## Evidence

`tests/test_arc_core.py` speed-independence test (phase-1 branch, commit 3300af4): spread 0.0072;
the spread scales with window ÷ curved-segment length, ≈ 0.84 · Δwindow · κ^(1/3) per curvature change.

LIBERO `pick up the alphabet soup and place it in the basket`, 50 demonstrations, recorded rows only
(`FTpi05/reports/action-sparsification-src/libero/retime_check.py`): each joint path re-timed (PCHIP over
arc length) at uniform speed, at uniform 2× speed (half the frames) and with a 0.4×–1.6× sinusoidal speed;
largest |s/s_total| difference to the recording along arc length, median / max over episodes:

| metric | uniform | 2× | sine |
|---|---|---|---|
| `curvature`, window relative to the median step | 0.0040 / 0.0111 | 0.0162 / 0.0450 | 0.0035 / 0.0076 |
| `curvature`, absolute window (the recording's) | 0.0036 / 0.0108 | 0.0087 / 0.0332 | 0.0033 / 0.0078 |
| `second_difference` | 0.0443 / 0.0773 | 0.0479 / 0.0873 | 0.0486 / 0.0824 |

The relative window doubles the spread only when the median step itself changes (2×); with half the
frames, the absolute window still leaves 0.9% from the coarser sampling.

## Hypotheses

An absolute arc-length window (radians in joint space, fixed per robot or per dataset) removes the
dependence; it changes the `curvature_window` interface from a multiple of the median step to an
absolute length.

## Next Experiment

Decide between the relative and the absolute window with a training comparison: a 1.6% progress shift
moves a span-40 label point by ≈ 0.6 frames; whether that matters to a trained policy is not measured.

## Resolution

# ISSUE-002

Status: Open

Area: Action labels (`arc_core.progress_curve` step)

## Observation

`step` divides the total progress by every row of the episode, including end-padding rows and still
frames. Under `curvature` and `second_difference` those rows add no progress, so a span of S frames
reaches fewer recorded frames than S, while the server still plays the H points back over S frames: the
executed motion is slower than the recording, and every label taken on a padding row counts as padded.

## Impact

LIBERO run (40 padding rows on 135–195 recorded rows): `curvature` span 40 reaches 26.5–30 recorded
frames (`time` span 40: 34–40); the closed-loop replay of `curvature-s40-d2` executes 11,218–12,240
frames for 7,758 recorded (1.45–1.58×), with speed ratio median 0.70–0.82 of the reference step; the
`pad` column is 22–32% for the geometric metrics against 2–11% for `time`, of which ≈ 21 percentage points are
labels on padding rows. For static tasks the slower execution costs cycle time, not accuracy.

## Evidence

`FTpi05/runs/libero_alphabet_soup/sweep.json` (`reach`, `pad`), `replay_curvature-s40-d2.json`
(`executed_frames` vs `recorded_frames`, `speed_ratio`).

## Hypotheses

Normalizing `step` over recorded, moving rows only (padding rows and still frames excluded) makes a
span of S reach S recorded frames on average under every metric, and leaves padding rows' labels at
the final pose as now.

## Next Experiment

Recompute `reach`, `pad` and the replay's executed/recorded ratio with the step normalized over
recorded rows only, on the LIBERO run and the synthetic chain.

## Resolution
