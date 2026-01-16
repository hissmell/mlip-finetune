#!/usr/bin/env python
"""Example script for extracting Fisher Information Matrix."""

import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    from mlip_finetune.models import NequIPWrapper
    from mlip_finetune.data import create_dataloader
    from mlip_finetune.utils.fisher import (
        compute_fisher_information_matrix,
        normalize_fisher,
        apply_fisher_threshold,
        save_fisher_info
    )
    
    # Configuration
    model_path = '/path/to/model.nequip.zip'
    data_path = '/path/to/pretrain_data.xyz'
    output_path = 'fisher_matrices/fim.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model_config = {
        'package_path': model_path,
        'r_max': 6.0,
    }
    model = NequIPWrapper(model_config)
    model.to(device)
    
    # Create dataloader
    logger.info(f"Loading data from: {data_path}")
    dataloader = create_dataloader(
        file_path=data_path,
        batch_size=8,
        r_max=6.0,
        shuffle=False,
    )
    
    # Compute Fisher Information Matrix
    logger.info("Computing Fisher Information Matrix...")
    fisher_dict = compute_fisher_information_matrix(
        model=model.model,
        dataloader=dataloader,
        n_samples=None,  # Use all samples
        diagonal_only=True,
        device=device,
        loss_coeffs={'energy': 0.0, 'force': 100.0}
    )
    
    # Normalize Fisher values
    logger.info("Normalizing Fisher values...")
    fisher_dict = normalize_fisher(fisher_dict, method='max')
    
    # Apply threshold (optional)
    # fisher_dict = apply_fisher_threshold(fisher_dict, threshold=1e-5)
    
    # Save optimal parameters for EWC
    optimal_params = {}
    for name, param in model.model.named_parameters():
        if name in fisher_dict:
            optimal_params[name] = param.data.clone()
    
    # Save Fisher information
    metadata = {
        'model_path': model_path,
        'data_path': data_path,
        'n_samples': len(dataloader.dataset),
        'diagonal_only': True,
        'normalize': 'max',
    }
    
    save_fisher_info(
        fisher_dict=fisher_dict,
        filepath=output_path,
        optimal_params=optimal_params,
        metadata=metadata
    )
    
    # Print statistics
    n_params = sum(f.numel() for f in fisher_dict.values())
    n_important = sum((f > 0.01).sum().item() for f in fisher_dict.values())
    
    logger.info(f"Fisher statistics:")
    logger.info(f"  Total parameters: {n_params:,}")
    logger.info(f"  Important (>0.01): {n_important:,} ({100*n_important/n_params:.1f}%)")
    logger.info(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()

