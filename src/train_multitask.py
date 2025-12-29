"""Multi-task learning training for passenger detection + attribute classification."""

import os
import argparse
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

from config import YOLO_CONFIG, TRAIN_CONFIG, ATTRIBUTES
from models.multitask_model import MultiTaskDetectionModel, MultiTaskLoss
from utils.multitask_data import create_multitask_dataloader


class MultiTaskTrainer:
    """Trainer for multi-task passenger detection and attribute classification."""
    
    def __init__(
        self,
        output_dir: str = "results/multitask",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize trainer.
        
        Args:
            output_dir: Output directory for weights and logs
            device: Device to train on
        """
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = MultiTaskDetectionModel().to(self.device)
        
        # Initialize loss
        self.criterion = MultiTaskLoss(
            detection_weight=0.5,
            gender_weight=0.1,
            age_weight=0.1,
            height_weight=0.1,
            bmi_weight=0.1,
            clothing_weight=0.1,
        ).to(self.device)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        self.global_step = 0
    
    def train(
        self,
        train_image_dir: str,
        train_annotations: str,
        val_image_dir: str,
        val_annotations: str,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        save_interval: int = 5,
    ):
        """Train multi-task model.
        
        Args:
            train_image_dir: Training images directory
            train_annotations: Training annotations JSON
            val_image_dir: Validation images directory
            val_annotations: Validation annotations JSON
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_interval: Save checkpoint every N epochs
        """
        # Create dataloaders
        train_loader = create_multitask_dataloader(
            image_dir=train_image_dir,
            annotations_file=train_annotations,
            batch_size=batch_size,
            split="train",
            shuffle=True,
        )
        
        val_loader = create_multitask_dataloader(
            image_dir=val_image_dir,
            annotations_file=val_annotations,
            batch_size=batch_size,
            split="val",
            shuffle=False,
        )
        
        # Setup optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=TRAIN_CONFIG.get("momentum", 0.937),
            weight_decay=TRAIN_CONFIG.get("weight_decay", 5e-4),
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        print(f"\n{'='*50}")
        print(f"Multi-Task Training Configuration")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Output Dir: {self.output_dir}")
        print(f"{'='*50}\n")
        
        best_val_loss = float("inf")
        
        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_losses = self._train_epoch(train_loader, optimizer)
            
            # Validate epoch
            val_loss, val_losses = self._validate_epoch(val_loader)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss, train_losses, val_losses)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or val_loss < best_val_loss:
                self._save_checkpoint(epoch, train_loss, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*50}\n")
        
        self.writer.close()
    
    def _train_epoch(self, loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        losses_dict = {
            "detection": 0.0,
            "gender": 0.0,
            "age": 0.0,
            "height": 0.0,
            "bmi": 0.0,
            "clothing": 0.0,
        }
        
        for batch_idx, batch in enumerate(loader):
            # Move to device
            images = batch["image"].to(self.device)
            boxes = batch["boxes"].to(self.device)
            gender = batch["gender"].to(self.device)
            age_group = batch["age_group"].to(self.device)
            height_range = batch["height_range"].to(self.device)
            bmi_category = batch["bmi_category"].to(self.device)
            clothing = batch["clothing"].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss, loss_dict = self.criterion(
                detection_outputs=outputs["detection"],
                detection_targets=boxes,
                gender_outputs=outputs["gender"],
                gender_targets=gender,
                age_outputs=outputs["age"],
                age_targets=age_group,
                height_outputs=outputs["height"],
                height_targets=height_range,
                bmi_outputs=outputs["bmi"],
                bmi_targets=bmi_category,
                clothing_outputs=outputs["clothing"],
                clothing_targets=clothing,
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
            
            total_loss += loss.item()
            for key, val in loss_dict.items():
                losses_dict[key] += val
            
            self.global_step += 1
        
        # Average losses
        num_batches = len(loader)
        total_loss /= num_batches
        for key in losses_dict:
            losses_dict[key] /= num_batches
        
        return total_loss, losses_dict
    
    def _validate_epoch(self, loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        losses_dict = {
            "detection": 0.0,
            "gender": 0.0,
            "age": 0.0,
            "height": 0.0,
            "bmi": 0.0,
            "clothing": 0.0,
        }
        
        with torch.no_grad():
            for batch in loader:
                # Move to device
                images = batch["image"].to(self.device)
                boxes = batch["boxes"].to(self.device)
                gender = batch["gender"].to(self.device)
                age_group = batch["age_group"].to(self.device)
                height_range = batch["height_range"].to(self.device)
                bmi_category = batch["bmi_category"].to(self.device)
                clothing = batch["clothing"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    detection_outputs=outputs["detection"],
                    detection_targets=boxes,
                    gender_outputs=outputs["gender"],
                    gender_targets=gender,
                    age_outputs=outputs["age"],
                    age_targets=age_group,
                    height_outputs=outputs["height"],
                    height_targets=height_range,
                    bmi_outputs=outputs["bmi"],
                    bmi_targets=bmi_category,
                    clothing_outputs=outputs["clothing"],
                    clothing_targets=clothing,
                )
                
                total_loss += loss.item()
                for key, val in loss_dict.items():
                    losses_dict[key] += val
        
        # Average losses
        num_batches = len(loader)
        total_loss /= num_batches
        for key in losses_dict:
            losses_dict[key] /= num_batches
        
        return total_loss, losses_dict
    
    def _log_metrics(self, epoch, train_loss, val_loss, train_losses, val_losses):
        """Log metrics to tensorboard."""
        self.writer.add_scalar("Loss/train/total", train_loss, epoch)
        self.writer.add_scalar("Loss/val/total", val_loss, epoch)
        
        for key, val in train_losses.items():
            self.writer.add_scalar(f"Loss/train/{key}", val, epoch)
        
        for key, val in val_losses.items():
            self.writer.add_scalar(f"Loss/val/{key}", val, epoch)
    
    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        
        # Save periodic checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train multi-task passenger detection model")
    parser.add_argument("--train-images", type=str, default="data/train/images",
                       help="Path to training images")
    parser.add_argument("--train-annotations", type=str, default="data/train/annotations.json",
                       help="Path to training annotations")
    parser.add_argument("--val-images", type=str, default="data/val/images",
                       help="Path to validation images")
    parser.add_argument("--val-annotations", type=str, default="data/val/annotations.json",
                       help="Path to validation annotations")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="results/multitask",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0, cpu, etc)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiTaskTrainer(output_dir=args.output, device=args.device)
    
    # Start training
    trainer.train(
        train_image_dir=args.train_images,
        train_annotations=args.train_annotations,
        val_image_dir=args.val_images,
        val_annotations=args.val_annotations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
