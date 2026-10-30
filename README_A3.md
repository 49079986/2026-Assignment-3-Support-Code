# COMP3702 Assignment 3

Implement tabular Q-learning and feature-based Q-learning.

Initial configuration:
- QL learning rate: 0.2
- FQ learning rate: 0.03
- Epsilon: 0.25 to 0.02
- Epsilon decay fraction: 0.8

AI assistance:
OpenAI Codex assisted with code and development guidance.
Verification results will be recorded during implementation.

## Fixed learning-rate experiment

- Map: L1
- Fixed learning rates: 0.1, 0.2, 0.4
- Other hyperparameters, seed and budget settings held constant.
- Each run starts with a fresh environment and solver.
- Training and evaluation use the official harness.
- Final evaluation: 1000 greedy episodes.
- Learning curves: moving average over 50 training episodes.
- Run: python run_ql_rates.py
- Results: results/ql_rates_<timestamp>/

## Feature-based representation

- Fixed feature vector length: 21.
- Features describe action type/direction, map distances,
  nearby terrain and crystal-collection progress.
- Target changes to launch sites after collecting enough crystals.
- All features, including bias, are scaled together using L2 norm.
- Weights initialise to all zeros.
- Map distances are precomputed from the public map.
- Run checks: python check_fq_features.py

## FQ training and four-level baseline

- FQ uses linear Q-learning with alpha=0.03.
- Every weight update uses the same TD error computed from old weights.
- Terminal targets contain only the immediate reward.
- Training checks: python check_fq_training.py
- Baseline experiments: python run_baseline.py
- Levels: L1, L2, L3, L4.
- Both algorithms use their own official training budgets.
- Final greedy evaluation uses 1000 trials.
- FQ also uses the official generalisation evaluation.
- Results include episode logs, metrics, curves and labelled weights.