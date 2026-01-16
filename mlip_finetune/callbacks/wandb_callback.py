"""Weights & Biases logging callback."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

from mlip_finetune.callbacks.base import Callback

if TYPE_CHECKING:
    from mlip_finetune.trainers import Trainer

logger = logging.getLogger(__name__)


class WandBCallback(Callback):
    """
    Callback for Weights & Biases experiment tracking.
    
    Logs metrics, hyperparameters, and optionally model checkpoints to W&B.
    
    Args:
        project: W&B project name
        name: Run name (optional, auto-generated if not provided)
        entity: W&B entity/team name (optional)
        config: Additional config to log (merged with trainer config)
        tags: List of tags for the run
        notes: Notes for the run
        log_model: Whether to log model checkpoints to W&B
        log_freq: How often to log batch-level metrics (default: every batch)
    
    Example:
        >>> callback = WandBCallback(
        ...     project="mlip-finetune",
        ...     name="ewc_experiment_1",
        ...     tags=["ewc", "nequip", "BTO"]
        ... )
        >>> trainer.callbacks.append(callback)
    """
    
    def __init__(
        self,
        project: str = "mlip-finetune",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        log_model: bool = False,
        log_freq: int = 1,
    ):
        self.project = project
        self.name = name
        self.entity = entity
        self.extra_config = config or {}
        self.tags = tags
        self.notes = notes
        self.log_model = log_model
        self.log_freq = log_freq
        
        self.run = None
        self._step = 0
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Initialize W&B run."""
        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb not installed. Install with: pip install wandb"
            )
            return
        
        # Merge trainer config with extra config
        config = {**trainer.config, **self.extra_config}
        
        # Initialize run
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=config,
            tags=self.tags,
            notes=self.notes,
            reinit=True,
        )
        
        # Log model architecture info
        if trainer.model is not None:
            wandb.run.summary["n_parameters"] = trainer.model.get_num_parameters()
        
        logger.info(f"W&B run initialized: {wandb.run.url}")
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log epoch metrics to W&B."""
        if self.run is None:
            return
        
        import wandb
        
        # Log all metrics
        log_dict = {"epoch": epoch, **metrics}
        
        # Add learning rate
        if trainer.optimizer is not None:
            log_dict["learning_rate"] = trainer.optimizer.param_groups[0]['lr']
        
        # Add strategy-specific metrics (e.g., EWC components)
        if hasattr(trainer.strategy, 'get_loss_components'):
            loss_components = trainer.strategy.get_loss_components()
            for k, v in loss_components.items():
                log_dict[f"strategy/{k}"] = v
        
        wandb.log(log_dict, step=epoch)
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float
    ) -> None:
        """Log batch-level metrics to W&B."""
        if self.run is None:
            return
        
        if batch_idx % self.log_freq != 0:
            return
        
        import wandb
        
        wandb.log({
            "batch/loss": loss,
            "batch/step": self._step,
        }, step=self._step)
        
        self._step += 1
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Finish W&B run and optionally log final model."""
        if self.run is None:
            return
        
        import wandb
        
        # Log final metrics
        wandb.run.summary["best_val_loss"] = trainer.best_val_loss
        wandb.run.summary["total_epochs"] = trainer.current_epoch + 1
        
        # Log model artifact if requested
        if self.log_model and trainer.exp_dir is not None:
            best_model_path = trainer.exp_dir / 'checkpoints' / 'best_model.pt'
            if best_model_path.exists():
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description="Best model checkpoint"
                )
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)
                logger.info("Model artifact logged to W&B")
        
        # Finish run
        wandb.finish()
        logger.info("W&B run finished")


class WandBSweepCallback(Callback):
    """
    Callback for W&B hyperparameter sweeps.
    
    Use this callback when running hyperparameter optimization with W&B Sweeps.
    It automatically reports metrics for the sweep agent.
    
    Example:
        >>> # In your sweep config
        >>> sweep_config = {
        ...     'method': 'bayes',
        ...     'metric': {'name': 'val_loss', 'goal': 'minimize'},
        ...     'parameters': {
        ...         'lr': {'min': 1e-5, 'max': 1e-3},
        ...         'ewc_lambda': {'values': [100, 1000, 10000]}
        ...     }
        ... }
    """
    
    def __init__(self):
        self.run = None
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Get sweep config from wandb."""
        try:
            import wandb
            self.run = wandb.run
            
            if self.run is not None:
                # Override trainer config with sweep parameters
                sweep_config = dict(wandb.config)
                
                # Update learning rate if specified
                if 'lr' in sweep_config:
                    trainer.config['training']['lr'] = sweep_config['lr']
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = sweep_config['lr']
                
                # Update EWC lambda if specified
                if 'ewc_lambda' in sweep_config:
                    trainer.config['strategy']['ewc_lambda'] = sweep_config['ewc_lambda']
                    if hasattr(trainer.strategy, 'ewc_lambda'):
                        trainer.strategy.ewc_lambda = sweep_config['ewc_lambda']
                
                logger.info(f"Sweep config applied: {sweep_config}")
        
        except Exception as e:
            logger.warning(f"Failed to apply sweep config: {e}")
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Report metrics to sweep agent."""
        if self.run is None:
            return
        
        import wandb
        wandb.log(metrics, step=epoch)

