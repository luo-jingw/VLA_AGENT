# PROJECT.md

Project-specific context that supplements `AGENTS.md`.

`AGENTS.md` is fixed across projects.
`PROJECT.md` accumulates what is specific to this one.

Update this file when a new project-specific constraint, environment
detail, or convention is discovered during work.

Do not duplicate:
- verified architecture → `docs/`;
- current work → `plan.md`;
- unresolved problems → `issues.md`.

## Template

- Initialized from (source, version, or commit): explicit-agent 0.1.0 (964b9d9c376f)
- Last synced to (source, version, or commit): explicit-agent 0.1.0 (964b9d9c376f)
- Last applied migration id (see the template's `migrations.json`): 0

## Purpose

An agent harness for VLA policy work: skills and tools that take robot demonstration data through
cleaning, keyframes, still-frame handling, diversity evaluation, action labels, openpi training,
evaluation and serving. It serves agents (and people) fine-tuning openpi π0 / π0.5 on new robots and
tasks.

## Environment

- Platform / hardware: Linux; data tools run on CPU; training, serving, LoRA conversion and parity
  checks need NVIDIA GPUs and an openpi environment (Python 3.11) with `integrations/openpi/patches`
  applied.
- Shared with (other users or projects on the same account or machine): none assumed.
- Constraints imposed by the environment: openpi's JAX policy preallocates GPU memory; run JAX and
  PyTorch policies in separate processes or set `XLA_PYTHON_CLIENT_MEM_FRACTION`. openpi's data loader
  starts workers with spawn, so scripts that build a data loader need a `__main__` guard.

## Credentials

Services that require a scoped identity in this project (git remotes,
package registries, model hubs, cloud APIs).

| Service | Scope | Source | Notes |
|---|---|---|---|
| GitHub | push to `luo-jingw/VLA_AGENT` (public) | personal access token kept in a local file outside the repository | Never committed |
| Git commit identity | the maintainer, GitHub `luo-jingw` (noreply address) | repository-local `user.name` / `user.email` | A machine's global git identity is not used |
| Weights & Biases | not used | — | Training runs with `WANDB_MODE=offline` or `wandb_enabled=False` |

Do not assume a global or default identity applies.
See `AGENTS.md` → Credentials and Environment Isolation.

Do not record token values or other secrets here.

## Project-Specific Constraints

Rules that apply only to this project, in addition to `AGENTS.md`.

- The repository holds no datasets, checkpoints, credentials or machine-specific paths; examples are
  configuration files only.
- Tools run as `python -m vla_tools.<module>` after `pip install -e .` (or with `src/` on `PYTHONPATH`).
  `vla_openpi.launch` and `vla_tools.lora_to_pytorch` additionally import openpi's `scripts/` and
  `examples/` modules, which must be on `PYTHONPATH`.
- `src/vla_tools/arc_interp.py` imports nothing else from the package: it is delivered to serving machines
  as a single file.

## Onboarding

What a new contributor must read or know before making the first change.

Read `AGENTS.md`, `README.md`, `.claude/skills/vla-pipeline/SKILL.md` and its `references/`, then
`docs/validation.md` for what has been verified and how.
