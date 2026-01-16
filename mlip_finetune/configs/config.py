"""Configuration loading and management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with support for inheritance.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the merged configuration.
        
    Example:
        >>> config = load_config("configs/ewc_training.yaml")
        >>> print(config['training']['epochs'])
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle config inheritance via 'defaults' key
    if 'defaults' in config:
        base_configs = config['defaults']
        if not isinstance(base_configs, list):
            base_configs = [base_configs]
        
        merged_config = {}
        for base_path in base_configs:
            base_full_path = config_path.parent / base_path
            if not str(base_full_path).endswith('.yaml'):
                base_full_path = Path(str(base_full_path) + '.yaml')
            
            with open(base_full_path, 'r') as bf:
                base_config = yaml.safe_load(bf)
                merged_config = merge_configs(merged_config, base_config)
        
        config = merge_configs(merged_config, config)
        del config['defaults']
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration (takes precedence).
        
    Returns:
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    required_sections = ['model', 'data', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")
    
    # Model validation
    if 'package_path' not in config['model'] and 'pretrained_path' not in config['model']:
        raise ValueError("Model config must have 'package_path' or 'pretrained_path'")
    
    # Data validation
    if 'finetune_data' not in config['data']:
        raise ValueError("Data config must have 'finetune_data' path")
    
    # Training validation
    if 'epochs' not in config['training']:
        raise ValueError("Training config must have 'epochs'")
    
    return True


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dictionary with default configuration.
    """
    return {
        'model': {
            'architecture': 'nequip',
            'compile_mode': 'eager',
            'r_max': 6.0,
        },
        'data': {
            'batch_size': 8,
            'train_split': 0.8,
            'val_split': 0.2,
            'test_split': 0.0,
        },
        'training': {
            'epochs': 100,
            'lr': 1e-4,
            'optimizer': 'adam',
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6,
            },
            'early_stopping': {
                'patience': 20,
                'min_delta': 1e-6,
            },
        },
        'loss_coeffs': {
            'energy': 1.0,
            'force': 100.0,
            'stress': 0.0,
        },
        'strategy': {
            'name': 'naive',
        },
        'logging': {
            'log_interval': 1,
            'save_checkpoint_interval': 1,
            'tensorboard': True,
        },
        'device': 'cuda',
        'seed': 42,
        'deterministic': True,
    }


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {path}")

