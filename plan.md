# Plan

Plan Status: proposed

# Problem

## Current

The harness has the right layers -- conventions (`AGENTS.md`), an orchestrator skill, eight stage
skills, a tool package, an openpi integration, an example and a validation record -- and every stage
artifact has one writer. Four things make it weaker as an agent harness than it needs to be:

1. The shared tool package lives inside one skill (`.claude/skills/vla-pipeline/tools/vla_tools`) and
   the openpi package inside `integrations/`; neither is installable, both need `PYTHONPATH` set by hand,
   and a project that symlinks the skills inherits that layout.
2. Every validation in `docs/validation.md` was run once in a scratch script; nothing re-runs it, so a
   change to a tool is not checked against the behaviour the docs claim.
3. A run's progress (which stage, which gate passed, which decision is pending) exists only as
   scattered files under `runs/<task>/`. An agent resuming in a new session has to reconstruct it, and
   the gates are prose that nothing checks. `plan.md` is for engineering changes; how it relates to a
   VLA run is not written down.
4. Which decisions an agent may make alone (detector thresholds) and which need a person (excluding
   episodes, accepting keyframe issues, choosing labels to train, long GPU jobs, publishing to public
   repositories) is not defined. `PROJECT.md` is also not loaded automatically, and the VLA skills do
   not list it among their inputs.

## Problem

Tools are not installable or regression-tested, run state is not machine-readable, and decision
authority is implicit.

## Goal

`pip install -e .` provides `vla_tools` and `vla_openpi`; `pytest` re-checks the documented behaviour
on synthetic data without GPU or network; `python -m vla_tools.run_status runs/<task>` reports each
stage's status from its artifacts; `vla-pipeline` states decision authority; agents load `PROJECT.md`.

# Structure

## Modules

- `src/vla_tools/` (moved from `.claude/skills/vla-pipeline/tools/vla_tools/`)
- `src/vla_openpi/` (moved from `integrations/openpi/vla_openpi/`); patches stay in `integrations/openpi/patches/`
- `tests/` — synthetic-data tests; tests needing openpi are marked and skipped without it
- `vla_tools.run_status` — stage status computed from artifacts
- `.claude/skills/vla-*` — procedures only

## Responsibilities

Skills describe procedure and judgment; packages hold all executable logic; tests hold the evidence
that `docs/validation.md` cites.

## State Ownership

A run's state is its artifacts. `run_status` reads them and writes nothing; `runs/<task>/decisions.md`
(new) is the single place where pending human decisions are listed, owned by `vla-pipeline`.

# Interface

## Interfaces

- `python -m vla_tools.<module>` after `pip install -e .`, unchanged module names and arguments.
- `python -m vla_tools.run_status runs/<task> --data-root data/` → one row per stage: status
  (`not started` / `in progress` / `passed` / `blocked`), the artifact checked, the reason.

## Inputs

Existing artifacts and their reports (`audit.json`, `meta/*_report.json`, `keyframes.json`,
`diversity.json`, `sweep.json`, label `arc_resample.json`, eval JSON).

## Outputs

Package metadata (`pyproject.toml`), tests, the status tool, skill and README updates.

## State Changes

Files move; no tool behaviour changes.

# Flow

## Main Flow

An agent starting or resuming a run calls `run_status`, works the first stage that is not `passed`
through its skill, lists any decision that needs a person in `runs/<task>/decisions.md` and stops on
it, and repeats.

# Code Mapping

## Modules

`src/vla_tools/*`, `src/vla_openpi/*`, `tests/test_*.py`, `pyproject.toml`, `CLAUDE.md`.

## Interfaces

`src/vla_tools/run_status.py`.

## State

`runs/<task>/decisions.md` in projects using the harness.

# Implementation

## Phase 1

Phase Status: pending

Round: 1

### Goal

Installable packages; skills and README point at `python -m vla_tools.<module>` after install.

### Files

`pyproject.toml`, `src/vla_tools/`, `src/vla_openpi/`, `.claude/skills/vla-*/**`, `README.md`,
`integrations/openpi/README.md`, `examples/ur3_dual_arm/*`, `PROJECT.md`.

### Structures

None new.

### Affected Modules

Packaging only.

### Dependencies

None.

### Observation

`pip install -e .` in a fresh venv; every `python -m vla_tools.<module> --help` runs; the integration
equivalence check from `docs/validation.md` gives the same result.

## Phase 2

Phase Status: pending

Round: 1

### Goal

`CLAUDE.md` importing `AGENTS.md` and `PROJECT.md`; VLA skills list `PROJECT.md` in their inputs;
decision authority table in `vla-pipeline`.

### Files

`CLAUDE.md`, `.claude/skills/vla-*/SKILL.md`.

### Structures

Decision authority table.

### Affected Modules

Skills.

### Dependencies

None.

### Observation

A new session in the repository has `PROJECT.md` in context.

## Phase 3

Phase Status: pending

Round: 2

### Goal

Synthetic-data tests for arc_core, arc_interp, cleaning and timing ops, keyframe detectors,
dataset_writer round trip, LoRA delta derivation; openpi-marked tests for the integration equivalence.

### Files

`tests/*`, `docs/validation.md`.

### Structures

Tiny synthetic LeRobot dataset fixture.

### Affected Modules

Tests.

### Dependencies

Phase 1.

### Observation

`pytest` passes without GPU or network; each validation row cites its test.

## Phase 4

Phase Status: pending

Round: 2

### Goal

`run_status` and `runs/<task>/decisions.md`.

### Files

`src/vla_tools/run_status.py`, `tests/test_run_status.py`, `.claude/skills/vla-pipeline/SKILL.md`,
`.claude/skills/vla-pipeline/references/artifacts.md`.

### Structures

Stage table derived from the existing gates.

### Affected Modules

Orchestration.

### Dependencies

Phase 1.

### Observation

On a run directory at each stage, the reported status matches the gate.
