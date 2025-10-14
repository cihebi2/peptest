"""
Training Framework

Implements trainers for:
1. Pre-training (contrastive + generative)
2. Fine-tuning on downstream tasks
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os


class PreTrainer:
    """
    Pre-trainer for self-supervised learning

    Combines:
    - Contrastive learning
    - Generative tasks
    - Mixed precision training
    """

    def __init__(
        self,
        model,
        contrastive_model,
        optimizer,
        scheduler,
        device='cuda',
        use_amp=True,
        grad_clip=1.0,
        log_interval=100
    ):
        self.model = model
        self.contrastive_model = contrastive_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader, augmentation_fn=None):
        """Train for one epoch"""
        self.model.train()
        self.contrastive_model.train()

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = batch.to(self.device)

            # Create two augmented views
            if augmentation_fn:
                view1 = augmentation_fn(batch)
                view2 = augmentation_fn(batch)
            else:
                view1, view2 = batch, batch

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss, _, _ = self.contrastive_model(view1, view2)
            else:
                loss, _, _ = self.contrastive_model(view1, view2)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            total_loss += loss.item()
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        self.epoch += 1
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def save_checkpoint(self, path, **extra_info):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **extra_info
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Checkpoint loaded from {path}")


class DownstreamTrainer:
    """
    Trainer for downstream tasks

    Supports:
    - Fine-tuning
    - Adapter/LoRA
    - Multi-task learning
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device='cuda',
        use_amp=True,
        grad_clip=1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        self.scaler = GradScaler() if use_amp else None
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            graphs, labels = batch
            graphs = graphs.to(self.device)
            labels = labels.to(self.device)

            # Forward
            if self.use_amp:
                with autocast():
                    outputs = self.model(graphs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(graphs)
                loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        self.epoch += 1
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            graphs, labels = batch
            graphs = graphs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(graphs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

        return total_loss / len(dataloader)
