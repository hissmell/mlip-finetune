"""Command-line interface for MLIP-Finetune."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='mlip-ft',
        description='MLIP Fine-tuning Framework'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train/fine-tune a model')
    train_parser.add_argument('--config', '-c', type=str, required=True,
                             help='Path to configuration file')
    train_parser.add_argument('--exp-dir', type=str, default=None,
                             help='Experiment directory')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--checkpoint', '-ckpt', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--data', '-d', type=str, required=True,
                            help='Path to evaluation data (XYZ file)')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use')
    
    # Extract Fisher command
    fisher_parser = subparsers.add_parser('extract-fisher', 
                                          help='Extract Fisher Information Matrix')
    fisher_parser.add_argument('--model', '-m', type=str, required=True,
                              help='Path to model (.nequip.zip)')
    fisher_parser.add_argument('--data', '-d', type=str, required=True,
                              help='Path to data for Fisher computation')
    fisher_parser.add_argument('--output', '-o', type=str, default='fisher.pt',
                              help='Output file path')
    fisher_parser.add_argument('--n-samples', type=str, default='all',
                              help='Number of samples (int or "all")')
    fisher_parser.add_argument('--device', type=str, default='cuda',
                              help='Device to use')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'extract-fisher':
        return cmd_extract_fisher(args)
    
    return 0


def cmd_train(args):
    """Run training."""
    from mlip_finetune import Trainer
    from mlip_finetune.configs import load_config
    
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    trainer = Trainer(config)
    trainer.setup(exp_dir=args.exp_dir)
    
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    trainer.fit()
    
    logger.info("Training complete!")
    return 0


def cmd_evaluate(args):
    """Run evaluation."""
    import torch
    from mlip_finetune.data import create_dataloader
    from mlip_finetune.metrics import compute_all_metrics
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    config = checkpoint.get('config', {})
    
    # Load model
    from mlip_finetune.models import get_model_wrapper
    model_config = config.get('model', {})
    
    ModelWrapper = get_model_wrapper(model_config.get('architecture', 'nequip'))
    model = ModelWrapper(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load data
    dataloader = create_dataloader(
        args.data,
        batch_size=8,
        r_max=model_config.get('r_max', 6.0),
        shuffle=False
    )
    
    # Evaluate
    all_pred_energy = []
    all_true_energy = []
    all_pred_forces = []
    all_true_forces = []
    
    logger.info("Running evaluation...")
    for batch in dataloader:
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        if 'pos' in batch:
            batch['pos'].requires_grad_(True)
        
        with torch.enable_grad():
            output = model(batch)
        
        if 'energy' in output:
            all_pred_energy.append(output['energy'].detach())
            all_true_energy.append(batch['total_energy'])
        
        if 'forces' in output:
            all_pred_forces.append(output['forces'].detach())
            all_true_forces.append(batch['forces'])
    
    # Compute metrics
    metrics = compute_all_metrics(
        pred_energy=torch.cat(all_pred_energy) if all_pred_energy else None,
        true_energy=torch.cat(all_true_energy) if all_true_energy else None,
        pred_forces=torch.cat(all_pred_forces) if all_pred_forces else None,
        true_forces=torch.cat(all_true_forces) if all_true_forces else None,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print("=" * 50)
    
    return 0


def cmd_extract_fisher(args):
    """Extract Fisher Information Matrix."""
    import torch
    from mlip_finetune.models import NequIPWrapper
    from mlip_finetune.data import create_dataloader
    from mlip_finetune.utils.fisher import (
        compute_fisher_information_matrix,
        normalize_fisher,
        save_fisher_info
    )
    
    device = torch.device(args.device)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model_config = {'package_path': args.model, 'r_max': 6.0}
    model = NequIPWrapper(model_config)
    model.to(device)
    
    # Load data
    logger.info(f"Loading data: {args.data}")
    dataloader = create_dataloader(args.data, batch_size=8, r_max=6.0, shuffle=False)
    
    # Parse n_samples
    n_samples = None if args.n_samples.lower() == 'all' else int(args.n_samples)
    
    # Compute Fisher
    logger.info("Computing Fisher Information Matrix...")
    fisher_dict = compute_fisher_information_matrix(
        model=model.model,
        dataloader=dataloader,
        n_samples=n_samples,
        device=device
    )
    
    # Normalize
    fisher_dict = normalize_fisher(fisher_dict, method='max')
    
    # Save optimal params
    optimal_params = {}
    for name, param in model.model.named_parameters():
        if name in fisher_dict:
            optimal_params[name] = param.data.clone()
    
    # Save
    save_fisher_info(fisher_dict, args.output, optimal_params)
    
    logger.info(f"Fisher information saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

