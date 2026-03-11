# All You (Probably Not, But Maybe) Need is Logic (and DDTs)

This project explores a neuro-symbolic reward modeling framework that combines
Differentiable Decision Trees (DDTs) with fuzzy logic constraints for learning
interpretable reward functions from pairwise trajectory preferences.

<img width="584" height="168" alt="image" src="https://github.com/user-attachments/assets/fe764372-9e1f-4ce2-9c79-e847d23f5734" />

## Key Ideas

1. Represent reward models using Differentiable Decision Trees.
2. Express trajectory preferences as logical implications.
3. Relax logical constraints using t-norm fuzzy logic.
4. Recover known losses (e.g., Bradley-Terry) from logical formulations.
5. Introduce additional logical constraints (RSS, OT, RP) to shape the reward landscape.


## Method Overview

Given trajectory pairs (τ+, τ−) with human preferences:

1. The reward model rθ(x) is parameterized using a DDT.
2. Rewards are aggregated over trajectories.
3. Logical constraints over trajectory preferences are relaxed into differentiable losses using fuzzy logic.

Example constraint:

If τ+ is preferred over τ−, then

IsGood(τ−) → IsGood(τ+)

Relaxing this implication yields a loss similar to RRHF.

See [final.pdf](final.pdf) for details on math and experiments.

