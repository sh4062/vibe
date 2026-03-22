# PPO with TRL

This folder contains a minimal PPO demo built directly on TRL.

It uses:

- a local causal LM policy
- a tiny trainable value model
- a rule-based reward model

This avoids a separately trained reward model and keeps the dependency chain shorter than the LLaMA-Factory PPO path.
