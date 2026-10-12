# Plan

Plan Status: completed

# Problem

## Current

The harness is centred on data preprocessing -- audit, cleaning, keyframes, still frames and end
padding, diversity and held-out split, action labels -- with training, evaluation and serving
downstream of the training-ready dataset it produces. Its README, loop figures and skill set give all
stages equal weight, and five gaps weaken it as an agent harness:

1. The shared tool package lives inside one skill (`.claude/skills/vla-pipeline/tools/vla_tools`) and
   the openpi package inside `integrations/`; neither is installable.
2. Every validation in `docs/validation.md` was run once in a scratch script; nothing re-runs it.
3. A run's progress exists only as scattered files under `runs/<task>/`; the gates are prose, and there
   is no single document describing the dataset the run produced.
4. Decision authority between agent and person is not defined; `PROJECT.md` is not loaded
   automatically and the VLA skills do not list it among their inputs.
5. Training, evaluation and serving are three full skills, as detailed as the data stages.

## Problem

The data-preprocessing core is not presented or tooled as the centre, tools are not installable or
regression-tested, run state is not machine-readable, decision authority is implicit, and the downstream
part is heavier than its role.

## Goal

- README and loop figures are about data preprocessing; downstream is one short section and one box.
- One `vla-downstream` skill replaces `vla-training`, `vla-eval`, `vla-serving`.
- `pip install -e .` provides `vla_tools` and `vla_openpi`; `pytest` re-checks the data tools on
  synthetic data without GPU or network.
- `run_status` reports each stage from its artifacts; `data_report` writes the run's data report.
- `vla-pipeline` states decision authority; agents load `PROJECT.md`.

# Structure

## Modules

- `src/vla_tools/`, `src/vla_openpi/`; patches in `integrations/openpi/patches/`
- `.claude/skills/`: `vla-pipeline` (data preprocessing orchestrator), five data stage skills
  (`vla-data-cleaning`, `vla-keyframes`, `vla-episode-timing`, `vla-data-diversity`,
  `vla-action-labels`), one `vla-downstream`
- `tests/`
- `vla_tools.run_status`, `vla_tools.data_report`

## Responsibilities

Skills hold procedure and judgment; packages hold executable logic; tests hold the evidence that
`docs/validation.md` cites. The data stages own the dataset versions and their reports; downstream
only reads the label dataset and its `arc_resample.json`.

## State Ownership

A run's state is its artifacts plus `runs/<task>/run.json` (the index of the run's dataset versions,
written by `vla-pipeline` when a version is created) and `runs/<task>/decisions.md` (pending human
decisions). `run_status` and `data_report` read them; `data_report` writes only `data_report.md`.

# Interface

## Interfaces

- `python -m vla_tools.<module>` after `pip install -e .`; module names and arguments unchanged.
- `python -m vla_tools.run_status runs/<task>` → one row per stage: status, artifact checked, reason.
- `python -m vla_tools.run_index runs/<task> --clean <dataset>` (and `--source`, `--timed`, `--labels`) → updates
  `runs/<task>/run.json`.
- `python -m vla_tools.data_report runs/<task>` → `runs/<task>/data_report.md`.
- `runs/<task>/decisions.md`: one `## D-NNN` entry per decision, seven fixed fields, read by
  `vla_tools.decision_log`.

## Inputs

`robot_spec.json`, `run.json`, stage artifacts and the reports under each dataset's `meta/`.

## Outputs

Packages, tests, two tools, restructured skills, README and figures.

## State Changes

Files move; three skills merge into one; stages S8–S11 become S8 downstream and S9 close.

# Flow

## Main Flow

An agent starting or resuming a run calls `run_status`, works the first data stage that is not
`passed` through its skill, lists any decision needing a person in `decisions.md` and stops on it; when
S7 passes it writes the data report and hands the label dataset downstream.

# Code Mapping

## Modules

`src/vla_tools/*`, `src/vla_openpi/*`, `tests/*`, `pyproject.toml`, `CLAUDE.md`,
`.claude/skills/vla-*/**`, `README.md`, `docs/figures/*`.

## Interfaces

`src/vla_tools/run_status.py`, `src/vla_tools/data_report.py`, `src/vla_tools/run_index.py`,
`src/vla_tools/decision_log.py`.

## State

`runs/<task>/run.json`, `runs/<task>/decisions.md`, `runs/<task>/data_report.md` in projects using
the harness.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Installable packages; every command in skills, README and examples works after `pip install -e .`.

### Files

`pyproject.toml`, `src/vla_tools/`, `src/vla_openpi/`, `.claude/skills/vla-*/**`, `README.md`,
`integrations/openpi/README.md`, `examples/ur3_dual_arm/*`, `PROJECT.md`.

### Structures

`vla_tools.lora_merge` (numpy-only merge, split from `lora_to_pytorch`).

### Affected Modules

Packaging.

### Dependencies

None.

### Observation

`pip install -e .` in a fresh venv; every `python -m vla_tools.<module> --help` that does not need
openpi runs there.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Skill set centred on data: `vla-pipeline` rewritten as the data preprocessing orchestrator with a
decision authority table and `PROJECT.md` among its inputs; `vla-downstream` replaces the three
downstream skills; stage numbering S0–S9; `CLAUDE.md` importing `AGENTS.md` and `PROJECT.md`.

### Files

`.claude/skills/vla-*/**`, `CLAUDE.md`.

### Structures

Decision authority table; downstream hand-off contract.

### Affected Modules

Skills.

### Dependencies

Phase 1 (command paths).

### Observation

No skill refers to a removed skill or to S10/S11; `CLAUDE.md` imports resolve.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Synthetic-data tests for the data tools (cleaning, timing, keyframes, writer, labels, interpolation,
diversity smoke, an end-to-end S1–S7 chain) and the LoRA merge; openpi-marked tests for `vla_openpi`.

### Files

`tests/*`, `docs/validation.md`.

### Structures

A synthetic LeRobot dataset fixture.

### Affected Modules

Tests.

### Dependencies

Phase 1.

### Observation

`pytest` passes in the fresh venv without GPU or network; openpi-marked tests pass in the openpi env.

## Phase 4

Phase Status: completed

Round: 3

### Goal

`run.json` index, `run_status`, `data_report`, `decisions.md` convention.

### Files

`src/vla_tools/run_index.py`, `src/vla_tools/decision_log.py`, `src/vla_tools/run_status.py`,
`src/vla_tools/data_report.py`, `tests/test_run_tools.py`, `tests/pipeline_chain.py`,
`.claude/skills/vla-pipeline/**`.

### Structures

Stage table derived from the gates.

### Affected Modules

Orchestration.

### Dependencies

Phases 1–2.

### Observation

On the synthetic end-to-end run, each stage's reported status matches its gate, and the report
lists every dataset version, decision and coverage finding.

## Phase 5

Phase Status: completed

Round: 4

### Goal

README and loop figures centred on data preprocessing; downstream as one short section and one box.

### Files

`README.md`, `docs/figures/make_figures.py`, `docs/figures/*.svg`, `docs/validation.md`.

### Structures

None.

### Affected Modules

Docs.

### Dependencies

Phases 2 and 4.

### Observation

Figures checked in light and dark rendering; README commands run as written.
