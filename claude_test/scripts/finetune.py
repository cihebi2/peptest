#!/usr/bin/env python
"""
Fine-tuning Script

Usage:
    python scripts/finetune.py --config configs/finetune_config.yaml
"""

import sys
sys.path.append('.')

import torch
import yaml
import argparse

from models import ImprovedPepLand
from finetuning import ModelWithAdapters, ModelWithLoRA, MultiTaskLearning
from training import DownstreamTrainer, get_optimizer, get_scheduler


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ImprovedPepLand')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, default='binding')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained model
    model = ImprovedPepLand(**config['model'])
    checkpoint = torch.load(config['model']['pretrained_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply fine-tuning strategy
    strategy = config['finetuning']['strategy']
    if strategy == 'adapter':
        model = ModelWithAdapters(model, **config['finetuning']['adapter'])
    elif strategy == 'lora':
        model = ModelWithLoRA(model, **config['finetuning']['lora'])
    elif strategy == 'multitask':
        model = MultiTaskLearning(model, config['finetuning']['multitask']['tasks'])
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config['training']['optimizer'])
    scheduler = get_scheduler(optimizer, config['training']['scheduler'])
    
    # Criterion
    task_config = config['tasks'][args.task]
    if task_config['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif task_config['loss'] == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer
    trainer = DownstreamTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        **config['training']
    )
    
    print(f"\nFine-tuning on task: {args.task}")
    print(f"Strategy: {strategy}")
    
    # TODO: Load task-specific dataset
    # Training loop
    # for epoch in range(config['training']['epochs']):
    #     train_loss = trainer.train_epoch(train_loader)
    #     val_loss = trainer.evaluate(val_loader)
    
    print("\n✓ Fine-tuning completed!")


if __name__ == '__main__':
    main()
