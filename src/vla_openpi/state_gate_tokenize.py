"""Tokenization for a gated state input (openpi patch 0005): both prompts of a sample, with and without state tokens.

`TokenizePromptWithGate` replaces openpi's `TokenizePrompt` in the model transforms of a training configuration with a
gate. Its `tokenized_prompt` is exactly what `TokenizePrompt(discrete_state_input=True)` returns (the instruction and the
state as 256-bin tokens); the same sample's tokens without the state (what a model trained without state input sees)
are under `openpi.training.state_gate.TOKENS_WITHOUT_STATE` and `MASK_WITHOUT_STATE`. `GatedDataset` keeps one of the two.
"""

from __future__ import annotations

import dataclasses

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.training import state_gate as _state_gate


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGate(_transforms.DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if (state := data.get("state", None)) is None:
            raise ValueError("State is required.")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        tokens_without_state, mask_without_state = self.tokenizer.tokenize(prompt, None)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            _state_gate.TOKENS_WITHOUT_STATE: tokens_without_state,
            _state_gate.MASK_WITHOUT_STATE: mask_without_state,
        }


def gate_model_transforms(group: _transforms.Group) -> _transforms.Group:
    """`group` with its (single) `TokenizePrompt` input replaced by `TokenizePromptWithGate` on the same tokenizer."""
    inputs = list(group.inputs)
    positions = [i for i, t in enumerate(inputs) if isinstance(t, _transforms.TokenizePrompt)]
    if len(positions) != 1 or not inputs[positions[0]].discrete_state_input:
        raise ValueError("a state gate needs exactly one TokenizePrompt with discrete_state_input in the model transforms")
    inputs[positions[0]] = TokenizePromptWithGate(inputs[positions[0]].tokenizer)
    return dataclasses.replace(group, inputs=inputs)
