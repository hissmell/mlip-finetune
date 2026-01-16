#!/usr/bin/env python
"""Example script for training with MLIP-Finetune."""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    from mlip_finetune import Trainer
    from mlip_finetune.configs import load_config
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "naive_finetune.yaml"
    
    # Or define config programmatically
    config = {
        'model': {
            'architecture': 'nequip',
            'package_path': '/path/to/model.nequip.zip',
            'r_max': 6.0,
        },
        'data': {
            'finetune_data': '/path/to/data.xyz',
            'batch_size': 8,
            'train_split': 0.8,
            'val_split': 0.2,
        },
        'training': {
            'epochs': 50,
            'lr': 1e-4,
        },
        'loss_coeffs': {
            'energy': 0.0,
            'force': 100.0,
        },
        'strategy': {
            'name': 'naive',
        },
        'device': 'cuda',
    }
    
    # Create trainer
    trainer = Trainer(config)
    trainer.setup(exp_dir='experiments/example_run')
    
    # Run training
    logger.info("Starting training...")
    final_metrics = trainer.fit()
    
    logger.info(f"Training complete! Final val loss: {final_metrics['loss']:.4f}")
    
    # Evaluate on test set (if available)
    if trainer.test_loader is not None:
        test_metrics = trainer.evaluate(trainer.test_loader)
        logger.info(f"Test loss: {test_metrics['loss']:.4f}")


if __name__ == '__main__':
    main()

