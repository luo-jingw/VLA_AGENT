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

## Hypotheses

An absolute arc-length window (radians in joint space, fixed per robot or per dataset) removes the
dependence; it changes the `curvature_window` interface from a multiple of the median step to an
absolute length.

## Next Experiment

Measure the spread on the LIBERO task (phase 5): resample each demonstration at its own speed and at
a re-timed copy (same joint path, different speed profile) with the current window definition.

## Resolution
