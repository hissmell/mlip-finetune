"""Main trainer class for MLIP fine-tuning."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlip_finetune.configs import load_config, get_default_config, merge_configs
from mlip_finetune.models import get_model_wrapper
from mlip_finetune.strategies import get_strategy
from mlip_finetune.data import create_dataloader
from mlip_finetune.callbacks import (
    CallbackList, 
    CheckpointCallback, 
    TensorBoardCallback,
    WandBCallback,
)
from mlip_finetune.metrics import MetricRegistry

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for MLIP fine-tuning.
    
    Orchestrates the training loop, model, strategy, and callbacks.
    
    Example:
        >>> from mlip_finetune import Trainer
        >>> trainer = Trainer.from_config("config.yaml")
        >>> trainer.fit()
        >>> trainer.evaluate()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = merge_configs(get_default_config(), config)
        self.device = torch.device(self.config.get('device', 'cuda'))
        
        # Set random seed
        self._set_seed(self.config.get('seed', 42))
        
        # Initialize components
        self.model = None
        self.strategy = None
        self.optimizer = None
        self.scheduler = None
        self.callbacks = CallbackList()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Experiment directory
        self.exp_dir = None
        
        # Metric registry (initialized from config in setup)
        self.metric_registry = None
    
    @classmethod
    def from_config(cls, config_path: str) -> "Trainer":
        """Create trainer from config file path."""
        config = load_config(config_path)
        return cls(config)
    
    def setup(self, exp_dir: Optional[str] = None) -> None:
        """
        Setup all training components.
        
        Args:
            exp_dir: Optional experiment directory path
        """
        # Create experiment directory
        if exp_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = self.config.get('strategy', {}).get('name', 'unknown')
            exp_dir = f"experiments/{timestamp}_{strategy_name}"
        
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        
        logger.info(f"Experiment directory: {self.exp_dir}")
        
        # Setup model
        self._setup_model()
        
        # Setup data
        self._setup_data()
        
        # Setup strategy
        self._setup_strategy()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup metrics
        self._setup_metrics()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_metrics(self) -> None:
        """Setup metric registry from config."""
        metrics_list = self.config.get('metrics', [])
        
        if metrics_list:
            self.metric_registry = MetricRegistry(metrics_list)
            logger.info(f"Tracking metrics: {metrics_list}")
        else:
            # Default metrics if none specified
            self.metric_registry = None
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_model(self) -> None:
        """Initialize the MLIP model."""
        model_config = self.config['model']
        architecture = model_config.get('architecture', 'nequip')
        
        ModelWrapper = get_model_wrapper(architecture)
        self.model = ModelWrapper(model_config)
        self.model.to(self.device)
        
        n_params = self.model.get_num_parameters()
        logger.info(f"Model: {architecture} with {n_params:,} trainable parameters")
    
    def _setup_data(self) -> None:
        """Setup data loaders."""
        data_config = self.config['data']
        r_max = self.config['model'].get('r_max', 6.0)
        
        # Create full dataset loader
        full_loader = create_dataloader(
            file_path=data_config['finetune_data'],
            batch_size=data_config.get('batch_size', 8),
            r_max=r_max,
            shuffle=False,
        )
        
        # Split dataset
        dataset = full_loader.dataset
        n_total = len(dataset)
        
        train_split = data_config.get('train_split', 0.8)
        val_split = data_config.get('val_split', 0.2)
        test_split = data_config.get('test_split', 0.0)
        
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test]
        )
        
        from mlip_finetune.data import collate_atomic_data
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 8),
            shuffle=True,
            collate_fn=collate_atomic_data,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('batch_size', 8),
            shuffle=False,
            collate_fn=collate_atomic_data,
        )
        
        if n_test > 0:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=data_config.get('batch_size', 8),
                shuffle=False,
                collate_fn=collate_atomic_data,
            )
        
        logger.info(f"Data: train={n_train}, val={n_val}, test={n_test}")
    
    def _setup_strategy(self) -> None:
        """Initialize the fine-tuning strategy."""
        strategy_config = self.config.get('strategy', {'name': 'naive'})
        strategy_name = strategy_config.get('name', 'naive')
        
        # Merge strategy config with main config
        full_config = {**self.config, **strategy_config}
        
        Strategy = get_strategy(strategy_name)
        self.strategy = Strategy(self.model, full_config)
        self.strategy.to(self.device)
        
        logger.info(f"Strategy: {strategy_name}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Optimizer
        lr = training_config.get('lr', 1e-4)
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        
        params = self.strategy.get_trainable_parameters()
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(params, lr=lr)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(params, lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(params, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6),
            )
        elif scheduler_type == 'CosineAnnealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('epochs', 100),
            )
        
        logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")
    
    def _setup_callbacks(self) -> None:
        """Setup training callbacks."""
        logging_config = self.config.get('logging', {})
        
        # Checkpoint callback
        self.callbacks.append(CheckpointCallback(
            save_dir=self.exp_dir / 'checkpoints',
            save_interval=logging_config.get('save_checkpoint_interval', 1),
        ))
        
        # TensorBoard callback
        if logging_config.get('tensorboard', True):
            self.callbacks.append(TensorBoardCallback(
                log_dir=self.exp_dir / 'logs',
            ))
        
        # Weights & Biases callback
        wandb_config = logging_config.get('wandb', {})
        if wandb_config.get('enabled', False):
            self.callbacks.append(WandBCallback(
                project=wandb_config.get('project', 'mlip-finetune'),
                name=wandb_config.get('name'),
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags'),
                notes=wandb_config.get('notes'),
                log_model=wandb_config.get('log_model', False),
            ))
    
    def fit(self) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Returns:
            Dictionary with final metrics
        """
        if self.model is None:
            self.setup()
        
        epochs = self.config['training'].get('epochs', 100)
        
        # Before training
        self.strategy.before_task(task_id=0, dataloader=self.train_loader)
        self.callbacks.on_train_begin(self)
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self, epoch)
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Evaluate on train set (for metrics only, not for training)
            train_eval_metrics = self.evaluate(self.train_loader, prefix="train_")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, prefix="val_")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Combine all metrics
            metrics = {
                **train_metrics,
                **train_eval_metrics,
                **val_metrics,
            }
            self.callbacks.on_epoch_end(self, epoch, metrics)
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pt')
            
            # Log progress
            log_msg = (
                f"Epoch {epoch+1}/{epochs} - "
                f"train_loss: {train_eval_metrics['train_loss']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}"
            )
            
            # Add detailed metrics if available
            if 'val_energy_mae' in val_metrics:
                log_msg += f", val_energy_mae: {val_metrics['val_energy_mae']:.4f}"
            if 'val_force_mae' in val_metrics:
                log_msg += f", val_force_mae: {val_metrics['val_force_mae']:.4f}"
            
            logger.info(log_msg)
        
        # After training
        self.strategy.after_task(task_id=0, dataloader=self.train_loader)
        self.callbacks.on_train_end(self)
        
        # Final evaluation on all sets
        final_metrics = self.evaluate(self.val_loader, prefix="val_")
        
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, prefix="test_")
            final_metrics.update(test_metrics)
            
            # Log test metrics
            logger.info("=" * 50)
            logger.info("Test Set Evaluation:")
            for name, value in test_metrics.items():
                logger.info(f"  {name}: {value:.6f}")
            logger.info("=" * 50)
        
        return final_metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Enable gradients for positions (needed for force computation)
            if 'pos' in batch:
                batch['pos'].requires_grad_(True)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            # Compute loss
            loss = self.strategy.compute_loss(output, batch, task_id=0)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(
        self, 
        dataloader: Optional[DataLoader] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on (default: validation set)
            prefix: Prefix for metric names (e.g., "train_", "val_", "test_")
            
        Returns:
            Dictionary of evaluation metrics including:
            - loss: Average loss
            - Configured metrics (energy_mae, force_mae, etc.)
        """
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0
        n_batches = 0
        
        # Reset metric registry if available
        if self.metric_registry is not None:
            self.metric_registry.reset()
        
        for batch in dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            if 'pos' in batch:
                batch['pos'].requires_grad_(True)
            
            with torch.enable_grad():
                output = self.model(batch)
            
            loss = self.strategy.compute_loss(output, batch, task_id=0)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update metric registry with batch predictions
            if self.metric_registry is not None:
                self.metric_registry.update(
                    pred_energy=output.get('energy'),
                    true_energy=batch.get('total_energy'),
                    pred_forces=output.get('forces'),
                    true_forces=batch.get('forces'),
                    n_atoms=batch.get('num_nodes'),
                )
        
        # Compute metrics
        metrics = {f'{prefix}loss': total_loss / n_batches}
        
        # Add configured metrics
        if self.metric_registry is not None:
            detailed = self.metric_registry.compute()
            for name, value in detailed.items():
                metrics[f'{prefix}{name}'] = value
        
        return metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.exp_dir / 'checkpoints' / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'strategy_state_dict': self.strategy.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.strategy.load_state_dict(checkpoint['strategy_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path}")

