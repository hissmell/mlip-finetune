# MLIP-Finetune

A flexible framework for fine-tuning Machine Learning Interatomic Potentials (MLIPs).

## Features

- 🔧 **Multiple Fine-tuning Strategies**
  - Naive: Standard fine-tuning without regularization
  - EWC: Elastic Weight Consolidation for continual learning
  - More strategies coming soon (PNC, Replay, etc.)

- 🧠 **MLIP Backend Support**
  - NequIP (currently supported)
  - MACE (coming soon)

- 📊 **Comprehensive Metrics**
  - Energy MAE/RMSE (total and per-atom)
  - Force MAE/RMSE
  - Force cosine similarity

- 🔌 **Flexible Callbacks**
  - TensorBoard logging
  - **Weights & Biases (wandb)** integration
  - Checkpoint saving
  - Early stopping
  - W&B Sweep support for hyperparameter optimization

## Installation

```bash
# From PyPI (coming soon)
pip install mlip-finetune

# From source
git clone https://github.com/yourusername/mlip-finetune.git
cd mlip-finetune
pip install -e ".[nequip]"
```

## Quick Start

### Command Line Interface

```bash
# Train with a config file
mlip-ft train --config config.yaml

# Evaluate a model
mlip-ft evaluate --checkpoint best_model.pt --data test.xyz

# Extract Fisher Information Matrix
mlip-ft extract-fisher --model model.nequip.zip --data pretrain_data.xyz --output fisher.pt
```

### Python API

```python
from mlip_finetune import Trainer, EWCStrategy
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
```

### Configuration Example

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
  val_split: 0.2

training:
  epochs: 100
  lr: 1e-4
  optimizer: adam

loss_coeffs:
  energy: 0.0
  force: 100.0
  stress: 0.0

strategy:
  name: ewc
  ewc_lambda: 1000.0
  precomputed_fisher_path: /path/to/fisher.pt

# Metrics to track (computed for train/val/test, logged to wandb)
metrics:
  - energy_mae
  - energy_mae_per_atom
  - force_mae
  - force_cosine

logging:
  tensorboard: true
  save_checkpoint_interval: 10
  wandb:
    enabled: true
    project: mlip-finetune
    tags: ["ewc", "nequip"]
```

## Fine-tuning Strategies

### Naive Strategy
Simple fine-tuning without any regularization. Best for when you don't need to preserve pre-training knowledge.

```python
from mlip_finetune import NaiveStrategy

strategy = NaiveStrategy(model, config)
```

### EWC Strategy
Elastic Weight Consolidation prevents catastrophic forgetting by penalizing changes to important parameters.

```python
from mlip_finetune import EWCStrategy

config = {
    'ewc_lambda': 1000.0,
    'precomputed_fisher_path': 'fisher.pt'
}
strategy = EWCStrategy(model, config)
```

## Weights & Biases Integration

Enable W&B logging in your config:

```yaml
logging:
  wandb:
    enabled: true
    project: mlip-finetune
    name: my_experiment
    tags: ["ewc", "nequip"]
    log_model: true
```

Or add the callback manually:

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

## Project Structure

```
mlip_finetune/
├── configs/        # Configuration management
├── data/           # Data loading and preprocessing
├── models/         # MLIP backend wrappers
├── trainers/       # Training loop orchestration
├── strategies/     # Fine-tuning strategies
├── callbacks/      # Logging, checkpointing
├── metrics/        # Evaluation metrics
└── utils/          # Utilities (Fisher computation, etc.)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

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

