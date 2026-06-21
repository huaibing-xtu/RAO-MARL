# RAO-MARL: Risk-Aware Offline Multi-Agent Reinforcement Learning for Voltage-Secure Active Distribution Networks

This repository contains the official implementation of RAO-MARL, an **offline safe multi-agent reinforcement learning** framework for voltage-secure control in active distribution networks. The method is trained purely on fixed historical datasets without environment interaction, combining a Behavior VAE support constraint, conservative reward-cost twin critics, Lagrangian safety budgeting, and deployment-time candidate-action risk filtering. Simulation is conducted under the MAPDN active-voltage-control benchmark protocol on IEEE 33-bus and 141-bus systems.

## Supported Scenarios

| Test Case | Nodes | Agents | Description |
|-----------|-------|--------|-------------|
| **case33** | 33 | 6 | IEEE 33-bus distribution network |
| **case141** | 141 | 22 | IEEE 141-bus large-scale network |

## Architecture

```
Offline Replay Dataset  ──►  Behavior VAE + Behavior-Cloning Regularization
                                  │
                         ┌────────┼────────┐
                         ▼        ▼        ▼
                  Reward Critics  Cost Critics   Lagrangian Multiplier λ
                  (CQL + min)     (CQL + max)    (mixed budget: global + state)
                         │        │        │
                         └────────┼────────┘
                                  ▼
                   Candidate-Action Risk Filtering
                   score = Q_r⁻ − λ·Q_c⁺ − η·[Q_c⁺ − b(s)]₊
                                  │
                                  ▼
                    Closed-Loop Voltage-Secure Execution
```

## Directory Structure

```
RAO-MARL/
├── all_mabcqlag_v3.py              # Main entry — Offline training + checkpoint review + evaluation
├── build_offline.py                # Build offline replay (case33)
├── build_offline_141.py            # Build offline replay (case141)
├── build_offline_141_mixed.py      # Build mixed-quality replay (case141)
├── expand_offline_141.py           # Expand offline dataset
├── estimate_cost_limit.py          # Estimate safety cost limits from baselines
├── all_baseline.py                 # Train online MADDPG baseline
├── all_baseline_generalization.py  # Generalization test for baselines
├── baseline_checkpoints.py         # Baseline checkpoint evaluation
├── review_checkpoints.py           # Checkpoint review
├── review_checkpoints_split.py     # Split-window checkpoint review
├── offline_safe/                   # Core offline safe MARL framework
│   ├── algos/
│   │   ├── ma_bcq_retain_lag.py    # Base MABCQ-Retain-Lag
│   │   ├── ma_bcq_retain_lag_v2.py # V2: evaluation improvements
│   │   └── ma_bcq_retain_lag_v3.py # V3: safety budget, margin, cost reweight
│   ├── dataset/
│   │   ├── build_dataset.py        # Dataset construction
│   │   ├── common.py               # Shared utilities
│   │   ├── mix_replay.py           # Mixed replay buffer
│   │   ├── policies.py             # Policy wrappers
│   │   ├── replay_buffer.py        # Base replay buffer
│   │   └── replay_buffer_v3.py     # V3 replay with safety features
│   ├── eval/
│   │   ├── evaluate_policy.py      # Base policy evaluation
│   │   ├── evaluate_policy_v2.py   # V2/V3 closed-loop evaluation
│   │   ├── evaluate_baseline.py    # Baseline evaluation
│   │   ├── safety_metrics.py       # Safety cost (violation + destroy)
│   │   └── judgement_relaxed.py    # Relaxed voltage judgement
│   ├── models/
│   │   ├── mapdn_backbones.py      # Actor / Critic / VAE backbones
│   │   ├── nets.py                 # Network utilities
│   │   └── registry.py             # Model registry
│   └── trainers/
│       ├── train_offline.py        # Base offline trainer
│       ├── train_offline_retain.py # Retain-based trainer
│       └── train_offline_retain_v3.py # V3 trainer (full safety pipeline)
├── agents/                         # Multi-agent implementations
├── critics/                        # Critic network variants
├── models/                         # Model registry
├── learning_algorithms/            # Base RL algorithm classes
├── utilities/                      # Utility functions
├── environments/                   # MAPDN voltage control environment
├── args/                           # YAML configuration
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
conda activate <your-env-name>

# Windows
conda env create -f environment_win.yml
conda activate <your-env-name>
```

## Dataset

The simulation data for all three test cases (case33, case141, case322) is hosted on Hugging Face Datasets:

[https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks](https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks)

### Download

```bash
# Download the dataset archive (~9.3 GB)
wget https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks/resolve/main/voltage_control_data.zip

# Extract into the project
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

Each CSV file is a time-series matrix (3-minute resolution):
- **load_active**: Active power consumption (MW)
- **load_reactive**: Reactive power consumption (MVAr)
- **pv_active**: Photovoltaic active power generation (MW)

The `.p` files are pickled pandapower grid models containing network topology, line parameters, and bus configurations.

### Offline Replay Composition

The offline dataset follows the RAO-MARL mixed-quality protocol (300 episodes, Poor : Medium : Good = 1 : 5 : 4):

| Type | Source | Ratio | Role |
|------|--------|-------|------|
| Poor | Random policy | 10% | Covers low-quality, high-risk states |
| Medium | Noisy rule policy | 50% | Feasible regulation samples |
| Good | Historical online checkpoints | 40% | High-quality cooperative control |

### High-Performance I/O

For the case322 dataset (>3 GB files), convert CSVs to Parquet:

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

## Prerequisites

Before running RAO-MARL, you must have:

1. **Simulation data** — downloaded from Hugging Face (see [Dataset](#dataset) section above)
2. **Pre-trained online baseline checkpoints** — stored in `model_save/` directory. These are produced by `run.py` (included in this repository). You can either:
   - Train baselines yourself using `run.py` (see Step 1 below)
   - Use pre-trained checkpoints from the authors (contact for availability)
   - The expected checkpoint path format is: `model_save/var_voltage_control-{scenario}-{mode}-{alg}-{barrier_type}-{alias}/model.pt`

> **Note**: Baseline-specific offline training entry points for comparison methods (CQL-Lag, IQL-Lag, SAC-BC-Lag, MATD3-BC, PPO, BC-only) are not included in this release. The repository focuses on the RAO-MARL offline training pipeline (V3), which is the main contribution of the paper.

## Complete Run Order

The full experimental pipeline consists of 5 steps. Each step depends on the outputs of the previous step.

---

### Step 1: Train Online Baseline Models

```bash
python run.py --scenario {case33_3min_final|case141_3min_final} --mode distributed --voltage-barrier-type l1
```

**Output**: `model_save/var_voltage_control-{scenario}-{mode}-{alg}-{barrier_type}-{alias}/model.pt`

> **Note**: You can skip this step if you already have pre-trained baseline checkpoints available.

---

### Step 2: Build Offline Replay Dataset

Collect trajectories from trained baseline policies and (for case141) mix poor/medium/good quality data.

#### 2a. case33 (6 agents) — Expert-only replay

```bash
python build_offline.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scenario` | str | `case33_3min_final` | MAPDN scenario name |
| `--mode` | str | `distributed` | Control mode |
| `--voltage-barrier-type` | str | `l1` | Voltage barrier penalty type |
| `--episode-limit` | int | `480` | Max timesteps per episode |
| `--episodes` | int | `50` | Number of episodes per baseline |
| `--seed` | int | `1` | Random seed |
| `--start-day` | int | `730` | Start day index in dataset |
| `--day-step` | int | `1` | Day increment per episode |
| `--hour` | int | `23` | Hour of day for episode start |
| `--quarter` | int | `2` | Quarter-hour offset |
| `--dataset-root` | str | `./offline_safe/data/offline_case33_all` | Output directory |
| `--models` | str list | all 10 baselines | Baselines to collect data from |
| `--manual-reset` | flag | False | Use manual environment reset |

**Output**: `{dataset-root}/{alg}/replay/ep_XXXXX.npz` + `meta.json` + `dataset_manifest.json`

#### 2b. case141 (22 agents) — Mixed-quality replay

```bash
python build_offline_141_mixed.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scenario` | str | `case141_3min_final` | MAPDN scenario name |
| `--mode` | str | `distributed` | Control mode |
| `--voltage-barrier-type` | str | `l1` | Voltage barrier penalty type |
| `--episode-limit` | int | `480` | Max timesteps per episode |
| `--seed` | int | `1` | Random seed |
| `--start-day` | int | `0` | Start day index |
| `--day-step` | int | `1` | Day increment per episode |
| `--hour` | int | `0` | Hour of day |
| `--quarter` | int | `0` | Quarter-hour offset |
| `--dataset-root` | str | `./offline_safe/data/offline_case141_all` | Output directory |
| `--alg` | str | `maddpg` | Baseline algorithm for expert data |
| `--checkpoint` | str | `./model_save/.../model.pt` | Path to expert checkpoint |
| `--poor-episodes` | int | `200` | Number of random-policy episodes |
| `--medium-episodes` | int | `500` | Number of noisy-droop episodes |
| `--replay-total` | int | `800` | Total episodes after mixing |
| `--poor-ratio` | float | `0.1` | Poor data proportion |
| `--medium-ratio` | float | `0.5` | Medium data proportion |
| `--droop-gain` | float | `4.0` | Droop controller gain |
| `--droop-noise-std` | float | `0.02` | Droop noise standard deviation |
| `--skip-poor` | flag | False | Skip poor data collection |
| `--skip-medium` | flag | False | Skip medium data collection |
| `--skip-good` | flag | False | Skip good data collection |
| `--skip-mix` | flag | False | Skip final mixing step |
| `--manual-reset` | flag | False | Use manual environment reset |

**Output**: `{dataset-root}/{alg}/replay/ep_XXXXX.npz` (mixed Poor:Medium:Good = 1:5:4)

#### 2c. case141 — Expand existing dataset

If you need to increase the dataset size after initial construction:

```bash
python expand_offline_141.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-root` | str | `./offline_safe/data/offline_case141_all` | Existing dataset directory |
| `--alg` | str | `maddpg` | Baseline algorithm |
| `--poor-episodes` | int | `200` | Target number of poor episodes |
| `--medium-episodes` | int | `500` | Target number of medium episodes |
| `--replay-total` | int | `1200` | Target total episodes |
| `--poor-ratio` | float | `0.1` | Poor data proportion |
| `--medium-ratio` | float | `0.5` | Medium data proportion |
| `--droop-gain` | float | `2.0` | Droop controller gain |
| `--droop-noise-std` | float | `0.02` | Droop noise std |
| `--manual-reset` | flag | True | Use manual environment reset |

**Output**: Expanded `{dataset-root}/{alg}/replay/ep_XXXXX.npz`

---

### Step 3: Estimate Safety Cost Limit

Calculate the cost limit from the offline dataset for Lagrangian dual regulation.

```bash
python estimate_cost_limit.py --data-root {dataset_dir} [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data-root` | str | **required** | Path to replay dataset directory |
| `--gamma` | float | `0.99` | Discount factor for cost calculation |
| `--method` | str | `quantile` | Estimation method: `quantile`, `baseline`, or `manual` |
| `--quantile` | float | `0.8` | Quantile for method=`quantile` |
| `--baseline-json` | str | None | Baseline evaluation JSON for method=`baseline` |
| `--manual-value` | float | None | Manual cost limit for method=`manual` |
| `--output-json` | str | None | Save suggested cost limit to file |

**Output**: Printed cost statistics + optional JSON file with `cost_limit` recommendation.

---

### Step 4: Batch Evaluate Online Baselines (Optional)

Evaluate all 10 baseline models against the 15 safety/performance metrics.

```bash
python all_baseline.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scenario` | str | `case33_3min_final` | MAPDN scenario |
| `--mode` | str | `distributed` | Control mode |
| `--voltage-barrier-type` | str | `l1` | Voltage barrier type |
| `--baseline-alias` | str | `0` | Checkpoint alias suffix |
| `--baseline-save-root` | str | `./` | Root directory for `model_save/` |
| `--episode-limit` | int | `480` | Max timesteps per episode |
| `--eval-episodes` | int | `5` | Evaluation episodes per baseline |
| `--start-day` | int | `730` | Start day for evaluation |
| `--day-step` | int | `1` | Day increment |
| `--hour` | int | `23` | Hour of day |
| `--quarter` | int | `2` | Quarter-hour offset |
| `--cost-limit` | float | `5.6` | Cost limit for constraint check |
| `--results-dir` | str | `./results/baseline_all` | Output directory |
| `--models` | str list | all 10 | Baselines to evaluate |
| `--manual-reset` | flag | False | Use manual environment reset |

**Output**: `{results_dir}/{alg}.json` with summary and per-episode records.

---

### Step 5: Train RAO-MARL (Full Pipeline)

This is the main entry point. It runs the complete pipeline: **offline training → checkpoint review → final evaluation**.

```bash
python all_mabcqlag_v3.py [OPTIONS]
```

**Core Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-root` | str | `./offline_safe/data/offline_case33_all` | Pre-built offline replay directory |
| `--baseline-results-dir` | str | `./results/baseline_all` | Baseline evaluation JSON directory |
| `--save-root` | str | `./results/offline_retain_v3` | Output directory for trained models |
| `--scenario` | str | `case33_3min_final` | MAPDN scenario |
| `--mode` | str | `distributed` | Control mode |
| `--voltage-barrier-type` | str | `l1` | Voltage barrier type |
| `--models` | str list | all 10 | Baseline sources to process |

**Training Hyperparameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | `120` | Training epochs |
| `--batch-size` | int | `256` | Batch size |
| `--episode-limit` | int | `480` | Max timesteps per episode |
| `--seed` | int | `1` | Random seed |
| `--device` | str | `cuda` | Compute device (`cuda` or `cpu`) |
| `--actor-lr` | float | `3e-4` | Actor learning rate |
| `--critic-lr` | float | `3e-4` | Critic learning rate |
| `--vae-lr` | float | `3e-4` | VAE learning rate |
| `--lag-lr` | float | `1e-2` | Lagrangian multiplier learning rate |

**Safety & Budget Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cost-limit` | float | None (auto) | Global safety cost budget |
| `--cost-limit-ratio` | float | `0.8` | Ratio × quantile for auto cost limit |
| `--cost-limit-quantile` | float | `0.5` | Quantile for auto cost limit estimation |
| `--bc-coef` | float | `0.20` | Behavior cloning regularization coefficient |
| `--bc-coef-end` | float | `0.05` | BC coefficient decay target |
| `--safety-margin-coef` | float | `0.25` | Safety margin η for candidate screening |
| `--cost-weight-coef` | float | `1.5` | Unsafe-sample reweight coefficient |
| `--budget-mix-ratio` | float | `0.7` | State-budget vs global-budget blend |
| `--state-budget-quantile` | float | `0.5` | Quantile for state budget estimation |
| `--unsafe-weight-coef` | float | `2.0` | Unsafe transition weight coefficient |

**CQL Regularization**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cql-alpha-reward` | float | `0.05` | CQL penalty weight for reward critics |
| `--cql-alpha-cost` | float | `0.10` | CQL penalty weight for cost critics |

**Actor & Data Filtering**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-cost-quantile` | float | `0.7` | Episode cost quantile for actor filter |
| `--actor-disc-cost-quantile` | float | `0.7` | Discounted cost quantile filter |
| `--actor-return-quantile` | float | `0.3` | Return quantile filter (keep top) |
| `--actor-min-keep-ratio` | float | `0.5` | Minimum episode retention ratio |
| `--actor-good-weight` | float | `1.0` | Good episode weight |
| `--actor-medium-weight` | float | `0.5` | Medium episode weight |
| `--actor-bad-weight` | float | `0.1` | Bad episode weight |
| `--use-actor-filter-for-vae` | flag | False | Apply actor filter to VAE training |
| `--use-actor-filter-for-actor` | flag | False | Apply actor filter to actor training |

**Training Control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--warmup-actor-steps` | int | `700` | Steps before actor updates begin |
| `--lambda-warmup-steps` | int | `5000` | Steps before λ updates begin |
| `--eval-every` | int | `10` | Epochs between evaluations |
| `--eval-episodes` | int | `5` | Evaluation episodes per checkpoint |
| `--early-stop-patience` | int | `20` | Early stop patience (epochs) |
| `--early-stop-min-epochs` | int | `40` | Minimum epochs before early stop |
| `--topk-checkpoints` | int | `5` | Top-K checkpoints for review |
| `--freeze-actor-after-patience` | flag | False | Freeze actor on plateau |
| `--use-actor-lr-plateau` | flag | False | LR scheduler for actor |
| `--actor-lr-patience` | int | `10` | LR plateau patience |
| `--actor-lr-decay` | float | `0.5` | LR decay factor |
| `--actor-lr-min` | float | `1e-5` | Minimum actor LR |
| `--stop-on-no-improve` | flag | False | Halt training on plateau |
| `--no-cache` | flag | False | Disable dataset memory caching |

**Model Architecture**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--review-actor-model` | str | `mlp` | Actor backbone (`mlp` or `rnn`) |
| `--review-critic-model` | str | `maac` | Critic backbone (`mlp`, `maac`, `rnn`) |
| `--review-actor-hidden-dims` | int list | `[256, 256]` | Actor hidden layer dimensions |
| `--review-critic-hidden-dims` | int list | `[512, 512]` | Critic hidden layer dimensions |
| `--review-critic-attend-heads` | int | `4` | MAAC attention heads |

**Evaluation Windows**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--val-start-days` | int list | `[730, 760, 790]` | Validation window start days |
| `--val-eval-episodes` | int | `3` | Episodes per validation window |
| `--test-start-days` | int list | `[820, 850]` | Test window start days |
| `--test-eval-episodes` | int | `5` | Episodes per test window |
| `--start-day` | int | `730` | Training data start day |
| `--day-step` | int | `1` | Day increment |
| `--hour` | int | `23` | Hour of day |
| `--quarter` | int | `2` | Quarter-hour offset |
| `--soft-return-drop-ratio` | float | `0.1` | Acceptable return drop ratio |
| `--max-return-drop-ratio` | float | `0.3` | Maximum return drop ratio |
| `--target-baseline-json` | str | None | Override specific baseline JSON |

**Output** (per baseline in `{save_root}/{alg}/`):
- `model_epoch_XXXX.pt` — Checkpoint files
- `model_latest.pt`, `model_best.pt` — Best and latest model
- `dataset_stats.json` — Normalization statistics
- `train_history.json` — Training metrics history
- `final_split_review.json` — Split-window review results
- `offline_eval.json` — Offline evaluation summary
- `all_offline_results.json` — Aggregated results across all baselines
- `ranking.json` — Safety-first ranking of all methods

---

### Step-by-Step Quick Reference

```bash
# 0. Download data (once)
wget https://huggingface.co/datasets/hsvgbkhgbv/Multi-Agent-Power-Distribution-Networks/resolve/main/voltage_control_data.zip
unzip voltage_control_data.zip -d environments/var_voltage_control/

# 1. Train online baselines (requires run.py — NOT included)
python run.py --scenario case33_3min_final --mode distributed --voltage-barrier-type l1

# 2. Build offline replay
python build_offline.py                                    # case33
python build_offline_141_mixed.py --alg maddpg             # case141

# 3. Estimate cost limit
python estimate_cost_limit.py --data-root ./offline_safe/data/offline_case33_all/maddpg/replay

# 4. Evaluate baselines (optional)
python all_baseline.py --scenario case33_3min_final

# 5. Train RAO-MARL
python all_mabcqlag_v3.py --scenario case33_3min_final --epochs 120
```

### Key Configuration

| Parameter | Description |
|-----------|-------------|
| `cost_limit` | Global safety cost budget |
| `bc_coef` | Behavior cloning regularization coefficient |
| `safety_margin_coef` | Safety margin for candidate screening |
| `budget_mix_ratio` | State-budget vs global-budget mixing weight |
| `cost_weight_coef` | Cost critic unsafe-sample reweight coefficient |

## Algorithm Overview

RAO-MARL jointly addresses offline distribution shift, risk-estimation bias, and deployment safety through six components:

1. **Behavior VAE**: Learns a latent action distribution from offline data, preventing out-of-distribution actions
2. **Conservative Reward and Cost Critics**: Twin reward critics (lower bound via min) and twin cost critics (upper bound via max) with CQL regularization and unsafe-sample reweighting
3. **Lagrangian Multiplier**: Adaptive dual variable λ updated when cost upper bound exceeds the mixed budget
4. **Safety Budget**: State-dependent budget blended with a global cost limit for fine-grained local risk reference
5. **Candidate-Action Risk Filtering**: At deployment, K candidate actions are scored as `Q_r⁻ − λ·Q_c⁺ − η·[Q_c⁺ − b(s)]₊`, and only the safest is executed
6. **Offline-Only Training**: Training uses only fixed replay data — no MAPDN environment interaction during policy updates

### Evaluation Metrics

RAO-MARL is evaluated on both reward and safety dimensions:

| Metric | Type | Description |
|--------|------|-------------|
| Avg Cost | Safety | Average episodic safety cost (voltage violations + destructive states) |
| V_out | Safety | Voltage violation rate |
| CVaR95 Cost | Safety | Average cost of the worst 5% episodes (tail risk) |
| CR | Safety | Controllable ratio |
| No-Destroy Rate | Safety | Rate of episodes without protection-triggering states |
| Avg Return | Reward | Average episodic return |
| PL | Reward | Power loss |

### Compared Methods

RAO-MARL is benchmarked against Online MADDPG, BC-only, MATD3+BC, CQL-Lag, IQL-Lag, MASAC+BC+Lag, and Multi-PPO on the MAPDN protocol.

<!--
## Citation

If you use this code in your research, please cite:

```bibtex
@article{rao-marl2026,
  title={RAO-MARL: Risk-Aware Offline Multi-Agent Reinforcement Learning for Voltage-Secure Active Distribution Networks},
  author={...},
  journal={IEEE Transactions on Smart Grid},
  year={2026},
  note={under review}
}
```
-->

## License

This project is provided for research purposes.
