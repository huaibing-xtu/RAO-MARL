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
└── environment_win.yml             # Conda environment (Windows)
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

## Dataset

The simulation data for all three test cases (case33, case141, case322) is hosted on Hugging Face Datasets:

[https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks](https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks)

### Download

```bash
# Download the dataset archive (~9.3 GB)
wget https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks/resolve/main/voltage_control_data.zip

# Extract into the MAPDN project
unzip voltage_control_data.zip -d environments/var_voltage_control/
```

After extraction, the directory structure should be:

```
environments/var_voltage_control/data/
├── case33_3min_final/
│   ├── load_active.csv          # Active power load profiles (336 MB)
│   ├── load_reactive.csv        # Reactive power load profiles (340 MB)
│   ├── pv_active.csv            # PV generation profiles (52 MB)
│   └── model.p                  # Grid topology model (39 KB)
├── case141_3min_final/
│   ├── load_active.csv          # Active power load profiles (849 MB)
│   ├── load_reactive.csv        # Reactive power load profiles (858 MB)
│   ├── pv_active.csv            # PV generation profiles (159 MB)
│   └── model.p                  # Grid topology model (53 KB)
└── case322_3min_final/
    ├── load_active.csv          # Active power load profiles (3.2 GB)
    ├── load_reactive.csv        # Reactive power load profiles (3.2 GB)
    ├── pv_active.csv            # PV generation profiles (284 MB)
    └── model.p                  # Grid topology model (112 KB)
```

### Data Description

Each CSV file is a time-series matrix where:
- **Rows** correspond to time steps (3-minute resolution)
- **Columns** correspond to buses/nodes in the distribution network
- **load_active**: Active power consumption (MW)
- **load_reactive**: Reactive power consumption (MVAr)
- **pv_active**: Photovoltaic active power generation (MW)

The `.p` files are pickled pandapower grid models containing network topology, line parameters, and bus configurations for each test case.

### Usage in Code

The dataset is loaded automatically by the environment during offline replay construction:

```python
# The environment reads CSV data via pandapower
# See: environments/var_voltage_control/voltage_control_env.py
# Data path is configured in: args/env_args/var_voltage_control.yaml
```

### High-Performance I/O (Recommended for case322)

The case322 dataset contains files exceeding 3 GB. For faster loading on HPC clusters, convert CSVs to Parquet:

```bash
python -c "
import pandas as pd
from pathlib import Path
data_root = Path('environments/var_voltage_control/data')
for csv_file in data_root.rglob('*.csv'):
    pq_file = csv_file.with_suffix('.parquet')
    df = pd.read_csv(csv_file)
    df.to_parquet(pq_file)
    print(f'Converted: {csv_file} -> {pq_file}')
"
```

## Quick Start

### 0. Download simulation data

Download and extract the voltage control dataset before proceeding. See the [Dataset](#dataset) section above for download links and instructions.

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

<!--
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
-->

## License

This project is provided for research purposes. See the paper for details.
