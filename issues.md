# ISSUE-001

Status: Resolved

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

The chord is a fixed length for the whole dataset: `ResampleConfig.curvature_length` (rad) replaces the
multiple of each episode's median step, `sweep --curvature-length` defaults to 4 path steps (LIBERO
0.1224 rad), and a recorded config with `curvature_window` is refused. Spread of the normalized progress
across speed profiles: constructed path (80 / 40 / 97 frames, `curve_check.py`) 0.0577 → 0.0067; the
test's bent path 0.0072 → 0.0027; LIBERO re-timing median / max uniform 0.0036 / 0.0093, 2× 0.0082 /
0.0310 (was 0.0162 / 0.0450), sine 0.0036 / 0.0077. The remaining 2× figure comes from half as many
frames. Computing the curvature on the densely sampled reconstruction instead of the recorded rows gave
0.010–0.018 on the constructed path and was not adopted.

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

`arc_core.chunks` takes the number of recorded rows; each frame's step is computed from the segments of
its own window that lie between recorded rows, so padding rows never enter a step. One recording with 10
and 40 padding rows gives identical progress over the recorded rows, identical per-frame steps and
identical labels on unpadded frames. LIBERO: `curvature` span 40 reaches exactly 40 recorded frames on
every frame at density 0 and a median of 33.6 at density 2; at e = 10 the replay executes 1.08×
(density 0) and 1.25× (density 2) the recorded frames. The `pad` column is 23–31% for `curvature`: labels
taken on padding rows hold the final target and count as padded.

# ISSUE-003

Status: Resolved

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
the end of recording A are 129, 130, 131, 132 (rate 1; e = 10: 18 inferences, 180 executed frames for
134 recorded; e = 40: 8 inferences, 320 frames). With density 1 or 2 nothing is missed (e = 40: 4
inferences, 160 frames). Python `label_replay.closed_loop` and the JavaScript port give the same
observed frames.

## Hypotheses

The replay could stop once every remaining label is padded and the executed pose equals the final
target; the label side is already handled by `event_density` > 0 around command edges.

## Next Experiment

Count, on the synthetic chain and LIBERO, how many inferences the replay spends after the last label
point reaches the final target, before and after such a stop rule.

## Resolution

A window without free progress is spaced in time (per-window step, `arc_core.chunks`): labels of frames
whose window lies in the final still stretch walk through it at one point per span/H frames and carry
the command edge. Deck synthetic recordings, `curvature` span 40 density 0, rate 1: both edges kept in
the labels and switched in the replay for recordings A and B; A executes 170 frames in 17 inferences
at e = 10 (was 180 / 18 with the release missed) and 200 frames in 5 inferences at e = 40 (was 320 / 8);
the last observations are 118, 123, 127, 129. Frames whose window also moves still skip a still stretch;
their labels switch the command at the first point after it, at the same position.

# ISSUE-004

Status: Open

Area: `label_replay` re-observation at small execution lengths

## Observation

The re-observation searches recorded frames strictly after the last observation, so every inference
advances the observation by at least one recorded frame. When the executed targets cover less than one
recorded frame of path (small e, rate 1, slow segments or keyframe windows of geometric labels), the
observation runs ahead of the executed pose.

## Impact

LIBERO, `curvature-s40-d2`, rate 1: at e = 1 the replay executes 7074 frames for 7758 recorded, with path
deviation max 1.97 and keyframe position max 4.73 path steps, against max 0.42–0.57 and 1.12–3.13 for
e = 2–40; at rate 2, e = 1 gives max 0.53 and 1.97. The e = 1, rate 1 row overstates the error of the
labels. `rate_ceiling` replays the same loop: where an execution length covers less than one recorded
frame of path, the forced advance shortens the wall-clock duration and overstates the speed-up.

## Evidence

`FTpi05/runs/libero_alphabet_soup/replay_curvature-s40-d2.json` and `replay_curvature-s40-d2_r2.json`
(2026-10-14).

## Hypotheses

Allowing the observation to stay on the current frame when the nearest recorded state is the current one
(search from t instead of t + 1, keeping the forward tie-break) removes the forced advance; a stall guard
is then needed so the loop still ends.

## Next Experiment

Replay e = 1 and 2 at rates 1 and 2 with the search starting at t, and compare executed frames and path
deviation with the current rows.

## Resolution

# ISSUE-005

Status: Open

Area: `rate_ceiling` execution model on real robots

## Observation

DROID's measured joint and gripper positions follow the commanded targets about 4 control frames later
(15 Hz; median best lag 4 frames, 10th–90th percentile 4–5, residual median 0.042 rad ≈ 1.7 path steps at
the best lag, 301 successful episodes of file-000). The rate ceiling treats the executed commands as the
robot's motion: tracking lag and tracking error are not modelled.

## Impact

At r times the recorded speed a joint impedance controller tracks the commands with a larger error and
lag than in the recording; the rate ceiling within the velocity limits is an upper bound on what the
robot executes faithfully.

## Evidence

`FTpi05/reports/rate-ceiling-src/extract_droid.py` docstring; measurement on
`FTpi05/data/droid-raw/data/chunk-000/file-000.parquet` (lerobot/droid_1.0.1 revision 0eabc778), 2026-10-16.

## Hypotheses

The tracking error grows roughly with the commanded velocity and acceleration; DROID's recorded
command / measured pairs at different operator speeds would give that relation.

## Next Experiment

Per DROID episode, fit measured(t) against the command shifted by the lag, and relate the residual to the
command's speed; extrapolate the residual to the speeds the rate ceiling reaches.

## Resolution

# ISSUE-006

Status: Open

Area: `arc_core` label points at frame 0

## Observation

Points are placed at fractional action indices, the observed state sitting at index t − 1. At t = 0 the
state is at −1 and placements below 0 are clamped to 0, so every point that falls inside the first segment
(state → action 0) lands on action 0. With a first segment that carries more than one point spacing of
progress, two or more points of the frame-0 label coincide.

## Impact

The frame-0 label holds the first action for one or more points; the deployed chunk dwells there instead of
following the first segment. One label in the DROID subset (episode 37, `curvature` span 10) has two points
at action 0; `sweep`'s time-indexed `true` figure leaves such labels out.

## Evidence

`FTpi05/runs/droid_pick_place`, `data/droid-pick-place-timed`, 2026-10-16: true times of episode 37,
frame 0, `curvature` s10: 0, 1, 1, 1.41, 1.88, …; no other frame of the 60 episodes at spans 10–64.

## Hypotheses

Placing the frame-0 points inside the first segment by interpolating between the state and action 0, as
for t ≥ 1, removes the coincident points.

## Next Experiment

Count coincident points per dataset and span; compare frame-0 labels with interpolated placement.

## Resolution

# ISSUE-007

Status: Open

Area: `sweep` jump column with long end padding

## Observation

`jump` is the median over every row of the episode, padding rows included, of the first point's distance
from the current state. Padding rows repeat the final target, so their jump is 0; when an episode has more
padding rows than recorded rows, the median is 0.

## Impact

DROID subset (256 padding rows, median 216 recorded rows): `jump` is 0.0 path steps in every cell. A
`--max-jump` bound would pass every cell. The column is not used for admissibility unless `--max-jump` is
given.

## Evidence

`FTpi05/runs/droid_pick_place/sweep.json`, 2026-10-16.

## Hypotheses

Taking the median over recorded rows only gives the intended figure.

## Next Experiment

Recompute `jump` over recorded rows for LIBERO (40 padding rows) and DROID and compare with the current
column.

## Resolution

# ISSUE-008

Status: Open

Area: Deck JavaScript port of `label_replay.closed_loop`

## Observation

`closed_loop` executes the chunk of the last recorded frame and keeps observing that frame until the
executed pose is within the final tolerance of the final recorded row. The deck's JavaScript port
(`FTpi05/reports/action-sparsification-src/deck/core.js`, `closedLoop`) still stops when an observation
reaches the last recorded frame, without executing its chunk.

## Impact

The two no longer give the same observed frames and executed targets at the end of a run: the port ends one
chunk earlier and, at small e, short of the final pose. The published deck's figures come from the port and
are unaffected until the deck is rebuilt; `check_core.py` parity would fail on the final chunk.

## Evidence

`label_replay.py` end rule and `test_label_replay.py::test_sparse_chunks_keep_observing_the_last_frame_until_at_the_final_row`
(2026-10-17); `core.js` `while (t < lastFrame ...)`.

## Hypotheses

Porting the end rule (final tolerance from the port's path step) restores parity.

## Next Experiment

Port the end rule, run `check_core.py` against `closed_loop` on the synthetic recordings.

## Resolution
