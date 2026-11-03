"""Training recipes of the dual-arm UR3 example: steps, learning rate, warm-up, weight averaging, kept checkpoints.

A recipe holds the optimisation numbers only; which parameters train (full action expert or LoRA adapters) and the
data are chosen by `configs.train_config`. Every recipe uses batch 32, AdamW and a cosine learning-rate decay to
`DECAY_FRACTION` of its peak.
"""

from __future__ import annotations

import dataclasses

DECAY_FRACTION = 0.1


@dataclasses.dataclass(frozen=True)
class Recipe:
    name: str
    num_train_steps: int
    peak_lr: float
    warmup_steps: int
    ema_decay: float
    keep_period: int  # a checkpoint at a multiple of this is kept besides the latest one

    @property
    def decay_lr(self) -> float:
        return self.peak_lr * DECAY_FRACTION


# Full action expert from pi05_base on real recordings.
FULL_20K = Recipe("full_20k", num_train_steps=20_000, peak_lr=2.5e-5, warmup_steps=500, ema_decay=0.999, keep_period=5000)
# Full action expert from pi05_base on the real recordings pick_and_place_quick_new (62 training episodes, 17,605 rows): the
# 10.3 passes over the training rows of full_20k, 5,500 steps of batch 32 (10.0 passes).
FULL_5500 = Recipe("full_5500", num_train_steps=5_500, peak_lr=2.5e-5, warmup_steps=150, ema_decay=0.999, keep_period=1000)
# Full action expert from pi05_base on simulated recordings, the first stage of a two-stage recipe.
SIM_10K = Recipe("sim_10k", num_train_steps=10_000, peak_lr=5e-5, warmup_steps=500, ema_decay=0.999, keep_period=5000)
# Full action expert from pi05_base on the simulated recordings of revision 5cd89567 (1000 episodes), first stage.
SIM_15K = Recipe("sim_15k", num_train_steps=15_000, peak_lr=5e-5, warmup_steps=500, ema_decay=0.999, keep_period=5000)
# LoRA adapters of the action expert on top of a trained model.
LORA_10K = Recipe("lora_10k", num_train_steps=10_000, peak_lr=1e-4, warmup_steps=500, ema_decay=0.999, keep_period=5000)
# LoRA calibration on a few real recordings, the second stage of a two-stage recipe. EMA decay 0.998 is not
# representable in bfloat16, so the averaged frozen arrays of the saved `params` drift (ISSUE-020); 0.999 does not.
LORA_4K = Recipe("lora_4k", num_train_steps=4_000, peak_lr=1e-4, warmup_steps=200, ema_decay=0.998, keep_period=2000)
# The same with decay 0.999, which leaves the frozen bfloat16 arrays of the saved `params` unchanged.
LORA_4K_EMA999 = Recipe("lora_4k_ema999", num_train_steps=4_000, peak_lr=1e-4, warmup_steps=200, ema_decay=0.999, keep_period=2000)

# Full action expert from a simulation-trained checkpoint on real rows mixed with simulated rows (--mix-repo-id).
MIX_8K = Recipe("mix_8k", num_train_steps=8_000, peak_lr=2.5e-5, warmup_steps=300, ema_decay=0.999, keep_period=2000)

# Full action expert from pi05_base on the real recordings pick_and_place_quick_new (62 training episodes, 17,605 rows) mixed
# with the simulated recordings of revision 98ee80d7 (288 training episodes, 55,359 rows), one stage. The passes of the
# single-source recipes are kept: 10 over the real rows (full_5500) and 2 over the simulated rows (sim_10k), 286,768 samples
# of batch 32, 8,962 steps rounded to 9,000; the simulated fraction runs from 0.6 to 0.15 (mean 0.375, 1.95 passes of the
# simulated rows, 10.2 of the real rows). Checkpoints of 15 GB are kept every 3,000 steps.
JOINT_9K = Recipe("joint_9k", num_train_steps=9_000, peak_lr=2.5e-5, warmup_steps=300, ema_decay=0.999, keep_period=3000)

RECIPES: dict[str, Recipe] = {
    r.name: r for r in (FULL_20K, FULL_5500, SIM_10K, SIM_15K, LORA_10K, LORA_4K, LORA_4K_EMA999, MIX_8K, JOINT_9K)
}
