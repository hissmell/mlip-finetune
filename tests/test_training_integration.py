#!/usr/bin/env python
"""
Integration test for MLIP-Finetune training pipeline.

This script tests the complete training pipeline with a small subset of data
to verify that everything works correctly.

Usage:
    python tests/test_training_integration.py
"""

import sys
import logging
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_naive_training():
    """Test naive fine-tuning strategy with small data subset."""
    import torch
    
    # Add package to path
    package_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(package_dir))
    
    from mlip_finetune import Trainer
    
    # Paths - update these to your actual paths
    MODEL_PATH = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/MLIP_Finetuning_Strategies/pretrained_models/MPTRJ-eFS.nequip.zip"
    DATA_PATH = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/datasets/01_Mptrj/01_preprocessed/02_BTO_1k/train.xyz"
    
    # Check if paths exist
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return False
    
    if not Path(DATA_PATH).exists():
        logger.error(f"Data not found: {DATA_PATH}")
        return False
    
    # Create config for small test
    config = {
        'model': {
            'architecture': 'nequip',
            'package_path': MODEL_PATH,
            'r_max': 6.0,
        },
        'data': {
            'finetune_data': DATA_PATH,
            'batch_size': 2,
            'train_split': 0.8,
            'val_split': 0.2,
        },
        'training': {
            'epochs': 3,  # Just 3 epochs for quick test
            'lr': 1e-4,
            'optimizer': 'adam',
        },
        'loss_coeffs': {
            'energy': 0.0,
            'force': 100.0,
            'stress': 0.0,
        },
        'metrics': [
            'energy_mae',
            'force_mae',
            'force_cosine',
        ],
        'strategy': {
            'name': 'naive',
        },
        'logging': {
            'tensorboard': False,
            'wandb': {'enabled': False},
            'save_checkpoint_interval': 10,
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }
    
    logger.info("=" * 60)
    logger.info("Testing Naive Fine-tuning Strategy")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            trainer = Trainer(config)
            trainer.setup(exp_dir=tmpdir)
            
            logger.info(f"Model parameters: {trainer.model.get_num_parameters():,}")
            logger.info(f"Train samples: {len(trainer.train_loader.dataset)}")
            logger.info(f"Val samples: {len(trainer.val_loader.dataset)}")
            
            # Run training
            final_metrics = trainer.fit()
            
            logger.info("\n" + "=" * 60)
            logger.info("Final Metrics:")
            for name, value in final_metrics.items():
                logger.info(f"  {name}: {value:.6f}")
            logger.info("=" * 60)
            
            # Check that loss decreased
            assert 'val_loss' in final_metrics, "val_loss not in metrics"
            
            logger.info("✓ Naive training test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"✗ Naive training test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_ewc_training():
    """Test EWC fine-tuning strategy with pre-computed Fisher."""
    import torch
    
    package_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(package_dir))
    
    from mlip_finetune import Trainer
    
    # Paths
    MODEL_PATH = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/MLIP_Finetuning_Strategies/pretrained_models/MPTRJ-eFS.nequip.zip"
    DATA_PATH = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/datasets/01_Mptrj/01_preprocessed/02_BTO_1k/train.xyz"
    FISHER_PATH = "/DATA/user_scratch/pn50212/2025/01_Moire/Finetuning/MLIP_Finetuning_Strategies/fisher_matrices/ti_contained_1k_diagonal.pt"
    
    # Check paths
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Model not found: {MODEL_PATH}")
        return None
    
    if not Path(DATA_PATH).exists():
        logger.warning(f"Data not found: {DATA_PATH}")
        return None
    
    if not Path(FISHER_PATH).exists():
        logger.warning(f"Fisher not found: {FISHER_PATH}, skipping EWC test")
        return None
    
    config = {
        'model': {
            'architecture': 'nequip',
            'package_path': MODEL_PATH,
            'r_max': 6.0,
        },
        'data': {
            'finetune_data': DATA_PATH,
            'batch_size': 2,
            'train_split': 0.8,
            'val_split': 0.2,
        },
        'training': {
            'epochs': 3,
            'lr': 1e-4,
        },
        'loss_coeffs': {
            'energy': 0.0,
            'force': 100.0,
        },
        'metrics': [
            'force_mae',
            'force_cosine',
        ],
        'strategy': {
            'name': 'ewc',
            'ewc_lambda': 1000.0,
            'precomputed_fisher_path': FISHER_PATH,
        },
        'logging': {
            'tensorboard': False,
            'wandb': {'enabled': False},
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    logger.info("=" * 60)
    logger.info("Testing EWC Fine-tuning Strategy")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            trainer = Trainer(config)
            trainer.setup(exp_dir=tmpdir)
            
            final_metrics = trainer.fit()
            
            logger.info("\n" + "=" * 60)
            logger.info("Final Metrics:")
            for name, value in final_metrics.items():
                logger.info(f"  {name}: {value:.6f}")
            logger.info("=" * 60)
            
            logger.info("✓ EWC training test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"✗ EWC training test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_metric_registry():
    """Test that MetricRegistry computes metrics correctly."""
    import torch
    import numpy as np
    
    package_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(package_dir))
    
    from mlip_finetune.metrics import MetricRegistry
    
    logger.info("=" * 60)
    logger.info("Testing MetricRegistry")
    logger.info("=" * 60)
    
    try:
        registry = MetricRegistry([
            'energy_mae',
            'force_mae',
            'force_cosine',
        ])
        
        # Create dummy data
        pred_energy = torch.tensor([1.0, 2.0, 3.0])
        true_energy = torch.tensor([1.1, 2.2, 2.8])
        
        pred_forces = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        true_forces = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        registry.update(
            pred_energy=pred_energy,
            true_energy=true_energy,
            pred_forces=pred_forces,
            true_forces=true_forces,
        )
        
        metrics = registry.compute()
        
        logger.info(f"Computed metrics: {metrics}")
        
        assert 'energy_mae' in metrics, "energy_mae not computed"
        assert 'force_mae' in metrics, "force_mae not computed"
        assert 'force_cosine' in metrics, "force_cosine not computed"
        
        # Perfect force alignment should give cosine = 1.0
        assert abs(metrics['force_cosine'] - 1.0) < 1e-5, f"force_cosine should be 1.0, got {metrics['force_cosine']}"
        
        logger.info("✓ MetricRegistry test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ MetricRegistry test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("MLIP-Finetune Integration Tests")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: MetricRegistry
    results['metric_registry'] = test_metric_registry()
    
    # Test 2: Naive training
    results['naive_training'] = test_naive_training()
    
    # Test 3: EWC training (optional, depends on Fisher file)
    ewc_result = test_ewc_training()
    if ewc_result is not None:
        results['ewc_training'] = ewc_result
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.error("Some tests FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

