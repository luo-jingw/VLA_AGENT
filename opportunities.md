# OPT-001

Status: Proposed

Area: Action labels

## Observation

Progress-spaced labels (`arc`, `curvature`) lose the time of each point, and the server has to assume
it; that assumption is the dominant deployment error of those metrics.

## Opportunity

Let the model predict each point's frame offset in an unused output dimension and interpolate with the
predicted times.

## Expected Mechanism

Placement error falls to the true-time reconstruction error that `sweep` reports in its `true` column.

## Required Evidence

A trained model whose predicted offsets match the label offsets on held-out episodes, and a lower
stitched error than uniform stride at the same span.

## Promotion Condition

A dataset where a progress metric beats uniform stride even with true times by a clear margin.

# OPT-002

Status: Proposed

Area: Keyframes

## Observation

Place, contact and lift moments without a command or sensor can only be found from kinematics; joint
speed is an indirect proxy.

## Opportunity

A tool that writes forward-kinematics quantities (end-effector height, gripper-to-object distance when
known) as dataset columns, read by keyframe detectors through `column` signals.

## Expected Mechanism

Direct detection of place and contact moments.

## Required Evidence

Kinematic parameters and base mounting per robot; contact sheets confirming the detected frames.

## Promotion Condition

A task whose label or evaluation depends on a place or contact moment.

# OPT-003

Status: Proposed

Area: Data tools

## Observation

`dataset_writer` refuses row changes on video datasets, so still-frame trimming and end padding are
unavailable for them.

## Opportunity

Re-encode the affected episodes' videos when rows are dropped or repeated.

## Expected Mechanism

The same timing step for parquet-image and video datasets.

## Required Evidence

Frame-exact decode of the re-encoded videos against the selected rows.

## Promotion Condition

A video dataset that needs still-frame edits or end padding.

# OPT-004

Status: Proposed

Area: Training

## Observation

Loss weights raise the mean weight above 1, which changes the effective learning rate against an
unweighted baseline.

## Opportunity

Normalise each dataset's weights to mean 1 at build time.

## Expected Mechanism

Comparisons with the baseline at equal effective learning rate.

## Required Evidence

Two trainings differing only in the normalisation.

## Promotion Condition

A comparison where a weighted model's advantage could be a learning-rate effect.
