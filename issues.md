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

Status: Resolved

Area: Action labels (`arc_core.progress_curve` step)

## Observation

The step divided the total progress by every row of the episode, end-padding rows included. Under
`curvature` and `second_difference` padding rows add no progress, so a span of S reached fewer recorded
frames than S while the server still played the H points back over S frames.

## Impact

LIBERO, `curvature` span 40 density 2: reach median 26.5 recorded frames; the closed-loop replay executed
1.45–1.58× the recorded frames.

## Evidence

`FTpi05/runs/libero_alphabet_soup/sweep.json` and `replay_curvature-s40-d2.json` before the change
(git history of `docs/validation.md`).

## Hypotheses

## Next Experiment

## Resolution

`arc_core.chunks` and `progress_curve` take the number of recorded rows and compute the step over the
segments between them (commit e3f93cb); still frames inside the recording still count. One recording
with 10 and 40 padding rows gives the same step, identical progress over the recorded rows and
labels within 5.6e-11 on unpadded frames. LIBERO re-run: `curvature` span 40 reaches 37.6 (density 0)
and 32.8 (density 2) recorded frames; at e = 10 the replay executes 1.03× (density 0) and 1.23×
(density 2) the recorded frames, the remainder being the keyframe windows (`time` density 2: 1.21×). The
`pad` column stays at 25–34% for geometric metrics: labels taken on padding rows hold the final target
and count as padded.

# ISSUE-003

Status: Open

Area: Action labels (geometric metrics with `event_density` 0), `label_replay` re-observation

## Observation

Under `curvature` (and `second_difference`), frames where the arm is exactly still add no progress. A
command edge issued during such a stretch with `event_density` 0 lies on a progress plateau: no label
point maps inside it, padded points clamp to the frame before the plateau, and the command never appears
in any label. In the closed-loop replay the executed command then never switches, the re-observation
window ends at `ceil(last label frame) + 3`, which no longer moves, and each further inference advances
the observation by one frame while executing `e` targets.

## Impact

The sweep rejects such cells (not every edge kept), so selection is unaffected; `label_replay` reports
the edge as missed but its inference and executed-frame counts are inflated at the episode end. LIBERO is
not affected: its demonstrations have no exactly still frames (110/110 edges kept in every cell,
`kf_missed` 0 in both replays).

## Evidence

Deck synthetic recordings (`FTpi05/reports/action-sparsification-src/deck/synthetic.js`, still frames at
the grasp and the release), `curvature` span 40 density 0: the release edge is missed; observed frames at
the end of recording A are 129, 130, 131, 132 (e = 10: 22 inferences, 220 executed frames for 134
recorded; e = 40: 9 inferences, 360 frames). With density 1 or 2 nothing is missed (e = 40: 5
inferences, 200 frames). Python `label_replay.closed_loop` and the JavaScript port give the same
observed frames.

## Hypotheses

The replay could stop once every remaining label is padded and the executed pose equals the final
target; the label side is already handled by `event_density` > 0 around command edges.

## Next Experiment

Count, on the synthetic chain and LIBERO, how many inferences the replay spends after the last label
point reaches the final target, before and after such a stop rule.

## Resolution

# ISSUE-004

Status: Open

Area: `label_replay` re-observation at small execution lengths

## Observation

The re-observation searches recorded frames strictly after the last observation, so every inference
advances the observation by at least one recorded frame. When the executed targets cover less than one
recorded frame of path (small e, rate 1, slow segments or keyframe windows of geometric labels), the
observation runs ahead of the executed pose.

## Impact

LIBERO, `curvature-s40-d2`, rate 1: at e = 1 the replay executes 6980 frames for 7758 recorded, with path
deviation max 2.41 and keyframe position max 5.04 path steps, against max 0.45–0.77 and 1.14–3.07 for
e = 2–40; at rate 2, e = 1 gives max 0.51 and 1.93. The e = 1, rate 1 row overstates the error of the
labels.

## Evidence

`FTpi05/runs/libero_alphabet_soup/replay_curvature-s40-d2.json` and `replay_curvature-s40-d2_r2.json`
(2026-10-13).

## Hypotheses

Allowing the observation to stay on the current frame when the nearest recorded state is the current one
(search from t instead of t + 1, keeping the forward tie-break) removes the forced advance; a stall guard
is then needed so the loop still ends.

## Next Experiment

Replay e = 1 and 2 at rates 1 and 2 with the search starting at t, and compare executed frames and path
deviation with the current rows.

## Resolution
