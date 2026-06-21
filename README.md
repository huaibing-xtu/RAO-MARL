# MAPDN — Multi-Agent Power Distribution Network Voltage Control

**Offline Safe Multi-Agent Reinforcement Learning for Volt-Var Control in Active Distribution Networks**

This repository contains the official implementation of the MAPDN framework, which applies **offline safe MARL** to voltage control in power distribution systems. The method combines behavior cloning, conservative Q-learning, and Lagrangian-constrained optimization with dual reward/cost critics to learn safe control policies from fixed offline datasets — without any online environment interaction during training.

## Supported Scenarios

| Test Case | Nodes | Agents | Description |
|-----------|-------|--------|-------------|
| **case33** | 33 | 6 | Standard benchmark distribution network |
| **case141** | 141 | 22 | Large-scale distribution network (scalability test) |

## Architecture

```
Offline Replay Dataset  ──►  Behavior VAE + BCQ Perturbation Actor
                              │
                     ┌────────┼────────┐
                     ▼        ▼        ▼
              Reward Critics  Cost Critics   Lagrangian Multiplier λ
              (CQL + min)     (CQL + max)    (adaptive budget)
                     │        │        │
                     └────────┼────────┘
                              ▼
               Candidate Action Screening
               score = Q_r_lower - λ * Q_c_upper - margin * budget_violation
                              │
                              ▼
                    MAPDN Closed-Loop Execution
```

## Directory Structure

```
MAPDN/
├── all_mabcqlag_v3.py              # Main entry point for V3 training pipeline
├── build_offline.py                # Build offline replay dataset (case33)
├── build_offline_141.py            # Build offline replay dataset (case141)
├── build_offline_141_mixed.py      # Build mixed-quality replay (case141)
├── expand_offline_141.py           # Expand offline dataset with additional data
├── estimate_cost_limit.py          # Estimate safety cost limits from baselines
├── all_baseline.py                 # Train online MADDPG baselines
├── all_baseline_generalization.py  # Generalization test for baselines
├── baseline_checkpoints.py         # Checkpoint evaluation for baselines
├── review_checkpoints.py           # Review and rank training checkpoints
├── review_checkpoints_split.py     # Split-window checkpoint review (V3)
├── offline_safe/                   # Core offline safe MARL framework
│   ├── algos/
│   │   ├── ma_bcq_retain_lag.py    # Base MABCQ-Retain-Lag algorithm
│   │   ├── ma_bcq_retain_lag_v2.py # V2: +evaluation improvements
│   │   └── ma_bcq_retain_lag_v3.py # V3: +safety budget, margin, cost critic reweight
│   ├── dataset/
│   │   ├── build_dataset.py        # Dataset construction pipeline
│   │   ├── common.py               # Shared data utilities
│   │   ├── mix_replay.py           # Mixed replay buffer
│   │   ├── policies.py             # Policy wrappers for data collection
│   │   ├── replay_buffer.py        # Base replay buffer
│   │   └── replay_buffer_v3.py     # V3 replay with safety features
│   ├── eval/
│   │   ├── evaluate_policy.py      # Base policy evaluation
│   │   ├── evaluate_policy_v2.py   # V2/V3 closed-loop evaluation
│   │   ├── evaluate_baseline.py    # Baseline policy evaluation
│   │   ├── safety_metrics.py       # Safety cost computation
│   │   └── judgement_relaxed.py    # Relaxed voltage judgement
│   ├── models/
│   │   ├── mapdn_backbones.py      # Network backbones (actor/critic/VAE)
│   │   ├── nets.py                 # Neural network utilities
│   │   └── registry.py             # Model builder registry
│   └── trainers/
│       ├── train_offline.py        # Base offline trainer
│       ├── train_offline_retain.py # Retain-based offline trainer
│       └── train_offline_retain_v3.py  # V3 trainer with full safety pipeline
├── agents/                         # MADDPG agent implementations
├── critics/                        # Critic network variants
├── models/                         # Multi-agent model registry
├── learning_algorithms/            # Base RL algorithm classes
├── utilities/                      # Utility functions (conversion, logging)
├── environments/                   # MAPDN simulation environment
├── args/                           # YAML configuration files
├── environment.yml                 # Conda environment (Linux)
├── environment_win.yml             # Conda environment (Windows)
└── CLAUDE.md                       # Project development guidelines
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.13+
- pandapower 2.13+

### Setup

```bash
# Linux
conda env create -f environment.yml
conda activate mapdn

# Windows
conda env create -f environment_win.yml
conda activate mapdn
```

## Quick Start

### 1. Build the offline replay dataset

```bash
# case33 (6 agents)
python build_offline.py

# case141 (22 agents)
python build_offline_141.py
```

### 2. Estimate safety cost limit

```bash
python estimate_cost_limit.py
```

### 3. Train MAPDN with offline safe MARL (V3)

```bash
# This runs the full pipeline: training → checkpoint review → final evaluation
python all_mabcqlag_v3.py
```

The V3 pipeline includes:
- Behavior VAE training for action support learning
- BCQ-style conservative policy optimization
- Dual reward/cost critics with CQL regularization
- Lagrangian multiplier for adaptive safety budget
- Candidate action screening with safety margin
- Multi-window checkpoint review and safety-first model selection

### Key V3 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cost_limit` | 5.6 | Global safety cost budget |
| `bc_coef` | 0.58 | Behavior cloning regularization strength |
| `safety_margin_coef` | 0.27 | Safety margin for candidate screening |
| `budget_mix_ratio` | 0.7 | State-budget vs global-budget mixing weight |
| `cost_weight_coef` | 1.55 | Cost critic unsafe-sample reweighting |

## Algorithm Overview

MAPDN V3 extends BCQ (Batch-Constrained Q-learning) to the constrained multi-agent setting:

1. **Behavior VAE**: Learns a latent action support from offline data to constrain the policy
2. **Conservative Critics**: Twin reward critics (min) and twin cost critics (max) with CQL regularization
3. **Lagrangian Optimization**: Adaptive multiplier λ balances reward maximization against cost constraints
4. **Safety Budget**: State-dependent budget blending for fine-grained safety control
5. **Candidate Screening**: At deployment, actions are scored as `Q_r_lower - λ * Q_c_upper - margin * budget_violation`
6. **Offline-Only**: No environment interaction during training — purely data-driven

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{mapdn2025,
  title={MAPDN: Multi-Agent Power Distribution Network Voltage Control via Offline Safe Reinforcement Learning},
  author={...},
  journal={IEEE Transactions on Power Systems},
  year={2025},
  note={under review}
}
```

## License

This project is provided for research purposes. See the paper for details.
