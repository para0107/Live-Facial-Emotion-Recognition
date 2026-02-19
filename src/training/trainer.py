import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import AverageMeter, accuracy


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, run_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.run_dir = run_dir

        self.writer = SummaryWriter(log_dir=run_dir)
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_checkpoint_path = os.path.join(config['paths']['checkpoints'], 'best_model.pth')
        os.makedirs(config['paths']['checkpoints'], exist_ok=True)

    def train_epoch(self, loader, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc = accuracy(logits, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.4f}'})

        return loss_meter.avg, acc_meter.avg

    def val_epoch(self, loader, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]', leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                acc = accuracy(logits, labels)

                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc, images.size(0))

        return loss_meter.avg, acc_meter.avg

    def fit(self, train_loader, val_loader):
        freeze_epochs = self.config['model'].get('freeze_epochs', 5)
        patience = self.config['training'].get('early_stopping_patience', 10)
        total_epochs = self.config['training']['epochs']

        for epoch in range(1, total_epochs + 1):
            if epoch == 1:
                print(f'Freezing backbone for first {freeze_epochs} epochs.')
                self.model.freeze_backbone()
            elif epoch == freeze_epochs + 1:
                print('Unfreezing full backbone.')
                self.model.unfreeze_backbone()

            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.val_epoch(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f'Epoch {epoch:03d}/{total_epochs} | '
                f'Train loss {train_loss:.4f} acc {train_acc:.4f} | '
                f'Val loss {val_loss:.4f} acc {val_acc:.4f} | '
                f'{elapsed:.1f}s'
            )

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            self.writer.add_scalar('Acc/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_acc)
                print(f'  -> New best: {val_acc:.4f}  (saved checkpoint)')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch} epochs.')
                    break

        self.writer.close()
        print(f'\nTraining complete. Best val accuracy: {self.best_val_acc:.4f}')

    def _save_checkpoint(self, epoch, val_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }, self.best_checkpoint_path)
