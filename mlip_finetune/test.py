#!/usr/bin/env python
"""
Quick integration test for MLIP-Finetune.

Run with:
    python -m mlip_finetune.test
    python -m mlip_finetune.test --model /path/to/model.nequip.zip --data /path/to/data.xyz
"""

import argparse
import logging
import tempfile
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_metric_registry():
    """Test MetricRegistry functionality."""
    from mlip_finetune.metrics import MetricRegistry
    
    logger.info("[1/3] Testing MetricRegistry...")
    
    registry = MetricRegistry(['energy_mae', 'force_mae', 'force_cosine'])
    
    # Dummy data
    pred_energy = torch.tensor([1.0, 2.0, 3.0])
    true_energy = torch.tensor([1.1, 2.2, 2.8])
    pred_forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    true_forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    registry.update(
        pred_energy=pred_energy,
        true_energy=true_energy,
        pred_forces=pred_forces,
        true_forces=true_forces,
    )
    
    metrics = registry.compute()
    
    assert 'energy_mae' in metrics, "energy_mae missing"
    assert 'force_mae' in metrics, "force_mae missing"
    assert abs(metrics['force_cosine'] - 1.0) < 1e-5, "force_cosine should be 1.0"
    
    logger.info("  ✓ MetricRegistry OK")
    return True


def test_data_loading(data_path: str, r_max: float = 6.0):
    """Test data loading pipeline."""
    from mlip_finetune.data import AtomicDataset, collate_atomic_data
    from torch.utils.data import DataLoader
    
    logger.info("[2/3] Testing data loading...")
    
    dataset = AtomicDataset(data_path, r_max=r_max)
    logger.info(f"  Loaded {len(dataset)} structures")
    
    # Test single item
    item = dataset[0]
    assert 'pos' in item, "pos missing"
    assert 'atom_types' in item, "atom_types missing"
    assert 'edge_index' in item, "edge_index missing"
    
    # Test batching
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_atomic_data)
    batch = next(iter(loader))
    
    assert 'batch' in batch, "batch index missing"
    assert batch['pos'].shape[0] > 0, "empty positions"
    
    logger.info("  ✓ Data loading OK")
    return True


def test_training(model_path: str, data_path: str, epochs: int = 2):
    """Test complete training pipeline."""
    from mlip_finetune import Trainer
    
    logger.info(f"[3/3] Testing training ({epochs} epochs)...")
    
    config = {
        'model': {
            'architecture': 'nequip',
            'package_path': model_path,
            'r_max': 6.0,
        },
        'data': {
            'finetune_data': data_path,
            'batch_size': 2,
            'train_split': 0.8,
            'val_split': 0.2,
        },
        'training': {
            'epochs': epochs,
            'lr': 1e-4,
        },
        'loss_coeffs': {
            'energy': 0.0,
            'force': 100.0,
        },
        'metrics': ['force_mae', 'force_cosine'],
        'strategy': {'name': 'naive'},
        'logging': {
            'tensorboard': False,
            'wandb': {'enabled': False},
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(config)
        trainer.setup(exp_dir=tmpdir)
        
        logger.info(f"  Model: {trainer.model.get_num_parameters():,} parameters")
        logger.info(f"  Train: {len(trainer.train_loader.dataset)} samples")
        logger.info(f"  Val: {len(trainer.val_loader.dataset)} samples")
        logger.info(f"  Device: {trainer.device}")
        
        # Run training
        metrics = trainer.fit()
        
        logger.info(f"  Final val_loss: {metrics['val_loss']:.4f}")
        if 'val_force_mae' in metrics:
            logger.info(f"  Final val_force_mae: {metrics['val_force_mae']:.4f}")
        
        # Check checkpoint saved
        ckpt_path = Path(tmpdir) / 'checkpoints' / 'best_model.pt'
        assert ckpt_path.exists(), "Checkpoint not saved"
        
    logger.info("  ✓ Training OK")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test MLIP-Finetune')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to model (.nequip.zip)')
    parser.add_argument('--data', '-d', type=str, default=None,
                       help='Path to test data (.xyz)')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                       help='Number of epochs for training test')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training test (useful for quick check)')
    args = parser.parse_args()
    
    # Default paths (update these for your system)
    default_model = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/NequIP-OAM-L-0.1.nequip.zip"
    # Use bundled test data
    from mlip_finetune.test_data import BTO_100
    default_data = str(BTO_100)
    
    model_path = args.model or default_model
    data_path = args.data or default_data
    
    logger.info("=" * 60)
    logger.info("MLIP-Finetune Quick Test")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: MetricRegistry
    try:
        results.append(('MetricRegistry', test_metric_registry()))
    except Exception as e:
        logger.error(f"  ✗ MetricRegistry FAILED: {e}")
        results.append(('MetricRegistry', False))
    
    # Test 2: Data loading
    if Path(data_path).exists():
        try:
            results.append(('DataLoading', test_data_loading(data_path)))
        except Exception as e:
            logger.error(f"  ✗ DataLoading FAILED: {e}")
            results.append(('DataLoading', False))
    else:
        logger.warning(f"  ⚠ Data not found: {data_path}")
        results.append(('DataLoading', None))
    
    # Test 3: Training
    if not args.skip_training and Path(model_path).exists() and Path(data_path).exists():
        try:
            results.append(('Training', test_training(model_path, data_path, args.epochs)))
        except Exception as e:
            logger.error(f"  ✗ Training FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(('Training', False))
    else:
        if args.skip_training:
            logger.info("[3/3] Training test skipped")
        else:
            logger.warning(f"  ⚠ Model or data not found, skipping training test")
        results.append(('Training', None))
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    
    all_passed = True
    for name, passed in results:
        if passed is True:
            logger.info(f"  ✓ {name}: PASSED")
        elif passed is False:
            logger.info(f"  ✗ {name}: FAILED")
            all_passed = False
        else:
            logger.info(f"  ⚠ {name}: SKIPPED")
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("All tests passed! 🎉")
        return 0
    else:
        logger.error("Some tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

