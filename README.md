# MLIP-Finetune

A flexible framework for fine-tuning Machine Learning Interatomic Potentials (MLIPs).

## Features

### 🔧 Fine-tuning Strategies
- **Naive**: Standard fine-tuning without regularization
- **EWC**: Elastic Weight Consolidation for continual learning (prevents catastrophic forgetting)
- More strategies coming soon (PNC, Replay, etc.)

### 🧠 MLIP Backend Support
- **NequIP** (currently supported)
- **MACE** (coming soon)

### 📊 Comprehensive Metrics
- Energy MAE/RMSE (total and per-atom)
- Force MAE/RMSE  
- Force cosine similarity

### 📈 Parity Plots
- Automatic generation of energy and force parity plots during training
- Saved to experiments folder and logged to W&B
- Publication-ready styling (Arial font, proper sizing, clean aesthetics)

### 🔌 Callbacks System
- **TensorBoard** logging
- **Weights & Biases (wandb)** integration
- Checkpoint saving (best model & interval-based)
- Early stopping
- Parity plot generation
- W&B Sweep support for hyperparameter optimization

### 🔄 Data Preprocessing
- **VASP to extxyz converter** (`mlip-ft-vasp2xyz`)
  - Batch conversion of VASP calculation directories
  - Robust fallback strategies for incomplete calculations
  - Supports OUTCAR, XDATCAR, vasprun.xml, OSZICAR

### 🧪 Built-in Testing
- Integration test module (`python -m mlip_finetune.test`)
- Bundled test data for quick verification

---

## Installation

```bash
# From source (recommended for development)
git clone https://github.com/yourusername/mlip-finetune.git
cd mlip-finetune
pip install -e ".[nequip]"

# From PyPI (coming soon)
pip install mlip-finetune
```

---

## Quick Start

### Command Line Interface

```bash
# Train with a config file
mlip-ft train --config config.yaml

# Evaluate a model
mlip-ft evaluate --checkpoint best_model.pt --data test.xyz

# Extract Fisher Information Matrix (for EWC)
mlip-ft extract-fisher --model model.nequip.zip --data pretrain_data.xyz --output fisher.pt

# Convert VASP calculations to extxyz format
mlip-ft-vasp2xyz /path/to/vasp/calculations -o dataset.xyz -v
```

### Python API

```python
from mlip_finetune import Trainer
from mlip_finetune.configs import load_config

# Load configuration
config = load_config("config.yaml")

# Create and run trainer
trainer = Trainer(config)
trainer.setup()
trainer.fit()

# Evaluate
metrics = trainer.evaluate()
print(f"Validation loss: {metrics['loss']:.4f}")
print(f"Force MAE: {metrics['force_mae']:.4f} eV/Å")
```

### Run Integration Test

```bash
# Quick test to verify installation
python -m mlip_finetune.test --model /path/to/model.nequip.zip --epochs 2

# Skip training, just test data loading
python -m mlip_finetune.test --skip-training
```

---

## Configuration

### Example Configuration File

```yaml
# config.yaml
model:
  architecture: nequip
  package_path: /path/to/model.nequip.zip
  r_max: 6.0

data:
  finetune_data: /path/to/finetune_data.xyz
  batch_size: 8
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

training:
  epochs: 100
  lr: 1.0e-4
  optimizer: adam
  scheduler:
    name: plateau
    factor: 0.5
    patience: 10
    min_lr: 1.0e-6

loss_coeffs:
  energy: 1.0
  force: 100.0
  stress: 0.0

strategy:
  name: ewc  # or 'naive'
  ewc_lambda: 1000.0
  precomputed_fisher_path: /path/to/fisher.pt

# Metrics to track (logged to wandb)
metrics:
  - energy_mae
  - energy_mae_per_atom
  - force_mae
  - force_rmse
  - force_cosine

logging:
  tensorboard: true
  save_checkpoint_interval: 10
  
  wandb:
    enabled: true
    project: mlip-finetune
    entity: null  # uses default logged-in entity
    tags: ["ewc", "nequip"]
    log_model: true
  
  parity_plots:
    enabled: true
    save_interval: 1  # save every epoch
    plot_energy: true
    plot_force: true
    per_atom_energy: true
```

---

## Fine-tuning Strategies

### Naive Strategy

Simple fine-tuning without any regularization. Best when you don't need to preserve pre-training knowledge.

```python
from mlip_finetune.strategies import NaiveStrategy

strategy = NaiveStrategy(model, config)
```

### EWC Strategy

Elastic Weight Consolidation prevents catastrophic forgetting by penalizing changes to important parameters (identified via Fisher Information Matrix).

```python
from mlip_finetune.strategies import EWCStrategy

config = {
    'ewc_lambda': 1000.0,
    'precomputed_fisher_path': 'fisher.pt'
}
strategy = EWCStrategy(model, config)
```

**Extract Fisher Information Matrix:**

```bash
mlip-ft extract-fisher \
  --model model.nequip.zip \
  --data pretrain_data.xyz \
  --output fisher.pt \
  --n_samples 1000 \
  --diagonal_only
```

---

## VASP Data Preprocessing

Convert VASP calculation outputs to extxyz format for training:

```bash
# Basic usage
mlip-ft-vasp2xyz /path/to/vasp/directory -o dataset.xyz

# Verbose mode (shows detailed progress)
mlip-ft-vasp2xyz /path/to/vasp/directory -o dataset.xyz -v

# Include stress tensor
mlip-ft-vasp2xyz /path/to/vasp/directory -o dataset.xyz --include-stress
```

### Fallback Strategy

The converter handles incomplete calculations gracefully:

| Situation | Strategy Used |
|-----------|---------------|
| Normal OUTCAR | Direct OUTCAR parsing |
| OUTCAR fails + valid vasprun.xml | XDATCAR + vasprun.xml (XML parsing) |
| OUTCAR fails + incomplete vasprun.xml | XDATCAR + OSZICAR + vasprun.xml (streaming) |

This ensures maximum data recovery even from crashed or interrupted calculations.

---

## Weights & Biases Integration

### Enable via Config

```yaml
logging:
  wandb:
    enabled: true
    project: mlip-finetune
    name: my_experiment
    tags: ["ewc", "nequip", "BTO"]
    log_model: true
```

### Manual Callback

```python
from mlip_finetune.callbacks import WandBCallback

callback = WandBCallback(
    project="mlip-finetune",
    name="ewc_experiment",
    tags=["ewc", "BTO"],
    log_model=True
)
trainer.callbacks.append(callback)
```

### Hyperparameter Sweeps

```python
import wandb
from mlip_finetune.callbacks import WandBSweepCallback

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'min': 1e-5, 'max': 1e-3},
        'ewc_lambda': {'values': [100, 1000, 10000]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="mlip-finetune")

def train():
    trainer = Trainer(config)
    trainer.callbacks.append(WandBSweepCallback())
    trainer.fit()

wandb.agent(sweep_id, train, count=20)
```

---

## Project Structure

```
mlip-finetune/
├── mlip_finetune/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── test.py             # Integration tests
│   │
│   ├── configs/            # Configuration management
│   │   └── config.py
│   │
│   ├── data/               # Data loading and preprocessing
│   │   ├── datasets.py     # AtomicDataset, XYZDataset
│   │   ├── collate.py      # Batch collation
│   │   └── vasp_converter.py  # VASP → extxyz conversion
│   │
│   ├── models/             # MLIP backend wrappers
│   │   └── nequip_wrapper.py
│   │
│   ├── trainers/           # Training loop orchestration
│   │   └── trainer.py
│   │
│   ├── strategies/         # Fine-tuning strategies
│   │   ├── base.py
│   │   ├── naive.py
│   │   └── ewc.py
│   │
│   ├── callbacks/          # Logging, checkpointing
│   │   ├── base.py
│   │   ├── checkpoint.py
│   │   ├── tensorboard.py
│   │   ├── early_stopping.py
│   │   ├── wandb_callback.py
│   │   └── parity_plot_callback.py
│   │
│   ├── metrics/            # Evaluation metrics
│   │   ├── registry.py
│   │   └── atomic.py
│   │
│   ├── keys/               # Key registry (property name mapping)
│   │   ├── standard.py
│   │   ├── nequip.py
│   │   ├── mace.py
│   │   └── registry.py
│   │
│   ├── utils/              # Utilities
│   │   ├── fisher.py       # Fisher Information computation
│   │   └── plotting.py     # Parity plot generation
│   │
│   └── test_data/          # Bundled test data
│       └── bto_100.xyz
│
├── examples/
│   ├── train_naive.py
│   ├── train_ewc.py
│   └── configs/
│       └── naive_finetune.yaml
│
├── tests/                  # Unit tests
│
├── pyproject.toml          # Package configuration
└── README.md
```

---

## Metrics Reference

| Metric Name | Description |
|-------------|-------------|
| `energy_mae` | Mean Absolute Error of total energy (eV) |
| `energy_rmse` | Root Mean Square Error of total energy (eV) |
| `energy_mae_per_atom` | MAE of energy per atom (eV/atom) |
| `energy_rmse_per_atom` | RMSE of energy per atom (eV/atom) |
| `force_mae` | MAE of force components (eV/Å) |
| `force_rmse` | RMSE of force components (eV/Å) |
| `force_cosine` | Mean cosine similarity of force vectors |

---

## Development

### Editable Installation

For development, install in editable mode:

```bash
pip install -e .
```

Changes to the source code are reflected immediately without reinstallation.

### Running Tests

```bash
# Quick integration test
python -m mlip_finetune.test --model /path/to/model.nequip.zip

# Unit tests
pytest tests/
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{mlip_finetune,
  title = {MLIP-Finetune: A Framework for Fine-tuning Machine Learning Interatomic Potentials},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/mlip-finetune}
}
```
