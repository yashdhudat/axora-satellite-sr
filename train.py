"""
SRGAN Training Pipeline
Team AXORA - Satellite Image Super-Resolution
GSA Pan India Hackathon | PS_S7_03

Usage:
    python train.py --dataset_path ./data/satellite --epochs 100 --scale 4
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path

from models.srgan import Generator, Discriminator, PerceptualLoss
from utils.metrics import calculate_psnr, calculate_ssim


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SatelliteDataset(Dataset):
    """
    Satellite image dataset loader.
    Expects HR images; auto-generates LR by downsampling.
    Supports: .jpg, .jpeg, .png, .tif, .tiff
    """
    def __init__(self, image_dir, hr_size=256, scale_factor=4, augment=True):
        self.image_dir = Path(image_dir)
        self.scale_factor = scale_factor
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor

        # Collect all image paths
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.image_dir.glob(ext)))
            self.image_paths.extend(sorted(self.image_dir.rglob(ext)))
        self.image_paths = list(set(self.image_paths))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Transforms
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(hr_size) if augment else transforms.CenterCrop(hr_size),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # Ensure minimum size
        if min(img.size) < self.hr_size:
            img = img.resize(
                (max(self.hr_size, img.width), max(self.hr_size, img.height)),
                Image.BICUBIC
            )

        hr_img = self.hr_transform(img)
        # Denormalize → resize → re-normalize for LR
        pil_hr = transforms.ToPILImage()(hr_img * 0.5 + 0.5)
        lr_img = self.lr_transform(pil_hr)

        return lr_img, hr_img


# ─── Trainer ─────────────────────────────────────────────────────────────────

class SRGANTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Models
        self.generator = Generator(scale_factor=args.scale, num_residual_blocks=16).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Losses
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixel_loss = nn.L1Loss()

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

        # LR schedulers
        self.scheduler_G = optim.lr_scheduler.MultiStepLR(
            self.optimizer_G, milestones=[args.epochs // 2], gamma=0.1
        )
        self.scheduler_D = optim.lr_scheduler.MultiStepLR(
            self.optimizer_D, milestones=[args.epochs // 2], gamma=0.1
        )

        # Dataset
        self.dataset = SatelliteDataset(
            args.dataset_path, hr_size=args.hr_size, scale_factor=args.scale
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )

        # Output dirs
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)

        self.best_psnr = 0
        self.history = {'g_loss': [], 'd_loss': [], 'psnr': [], 'ssim': []}

    def pretrain_generator(self, pretrain_epochs=5):
        """Pixel-wise pretraining for stable GAN warm-up."""
        print(f"\n{'='*50}")
        print("Phase 1: Generator Pre-training (Pixel Loss)")
        print(f"{'='*50}")

        for epoch in range(pretrain_epochs):
            self.generator.train()
            total_loss = 0
            for lr_imgs, hr_imgs in self.dataloader:
                lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                sr_imgs = self.generator(lr_imgs)
                loss = self.pixel_loss(sr_imgs, hr_imgs)

                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()
                total_loss += loss.item()

            avg = total_loss / len(self.dataloader)
            print(f"  Pre-train Epoch [{epoch+1}/{pretrain_epochs}] | Pixel Loss: {avg:.4f}")

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss, epoch_d_loss = 0, 0
        epoch_psnr, epoch_ssim = 0, 0
        start_time = time.time()

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.dataloader):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            batch_size = lr_imgs.size(0)

            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            # ── Train Discriminator ──────────────────────────────────────────
            self.optimizer_D.zero_grad()
            sr_imgs = self.generator(lr_imgs).detach()

            real_pred = self.discriminator(hr_imgs)
            fake_pred = self.discriminator(sr_imgs)

            d_real_loss = self.adversarial_loss(real_pred, real_labels)
            d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            self.optimizer_D.step()

            # ── Train Generator ──────────────────────────────────────────────
            self.optimizer_G.zero_grad()
            sr_imgs = self.generator(lr_imgs)
            fake_pred = self.discriminator(sr_imgs)

            # Combined loss: pixel + perceptual + adversarial
            g_pixel_loss = self.pixel_loss(sr_imgs, hr_imgs) * 1.0
            g_percep_loss = self.perceptual_loss(sr_imgs, hr_imgs) * 0.006
            g_adv_loss = self.adversarial_loss(fake_pred, real_labels) * 1e-3
            g_loss = g_pixel_loss + g_percep_loss + g_adv_loss

            g_loss.backward()
            self.optimizer_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # Metrics on first batch
            if batch_idx == 0:
                with torch.no_grad():
                    sr_np = ((sr_imgs[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                    hr_np = ((hr_imgs[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                    epoch_psnr = calculate_psnr(sr_np, hr_np)
                    epoch_ssim = calculate_ssim(sr_np, hr_np)

        elapsed = time.time() - start_time
        n = len(self.dataloader)
        return epoch_g_loss/n, epoch_d_loss/n, epoch_psnr, epoch_ssim, elapsed

    def save_samples(self, epoch, lr_imgs, hr_imgs):
        """Save side-by-side LR/SR/HR comparison images."""
        self.generator.eval()
        with torch.no_grad():
            sr_imgs = self.generator(lr_imgs[:4].to(self.device))

        # Upsample LR for visual comparison
        lr_up = nn.functional.interpolate(
            lr_imgs[:4], scale_factor=self.args.scale, mode='bicubic', align_corners=False
        )
        comparison = torch.cat([lr_up.cpu(), sr_imgs.cpu(), hr_imgs[:4]], dim=0)
        comparison = (comparison + 1) / 2  # [-1,1] → [0,1]
        save_image(
            comparison,
            os.path.join(self.args.sample_dir, f"epoch_{epoch:04d}.png"),
            nrow=4, normalize=False
        )

    def train(self):
        print(f"\n{'='*60}")
        print("AXORA | Satellite Image Super-Resolution Training")
        print(f"Dataset: {len(self.dataset)} images | Scale: {self.args.scale}x")
        print(f"Device: {self.device} | Epochs: {self.args.epochs}")
        print(f"{'='*60}")

        # Phase 1: Pretrain
        self.pretrain_generator(pretrain_epochs=min(5, self.args.epochs // 10))

        # Phase 2: Full GAN training
        print(f"\n{'='*50}")
        print("Phase 2: GAN Training (Adversarial)")
        print(f"{'='*50}")

        first_batch = None
        for epoch in range(1, self.args.epochs + 1):
            g_loss, d_loss, psnr, ssim_val, elapsed = self.train_epoch(epoch)

            self.scheduler_G.step()
            self.scheduler_D.step()

            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            self.history['psnr'].append(psnr)
            self.history['ssim'].append(ssim_val)

            print(
                f"Epoch [{epoch:3d}/{self.args.epochs}] | "
                f"G: {g_loss:.4f} | D: {d_loss:.4f} | "
                f"PSNR: {psnr:.2f}dB | SSIM: {ssim_val:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save samples
            if epoch % self.args.sample_interval == 0:
                for lr, hr in self.dataloader:
                    self.save_samples(epoch, lr, hr)
                    break

            # Save best checkpoint
            if psnr > self.best_psnr:
                self.best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'generator_state': self.generator.state_dict(),
                    'discriminator_state': self.discriminator.state_dict(),
                    'optimizer_G_state': self.optimizer_G.state_dict(),
                    'optimizer_D_state': self.optimizer_D.state_dict(),
                    'psnr': psnr,
                    'ssim': ssim_val,
                }, os.path.join(self.args.checkpoint_dir, 'best_model.pth'))
                print(f"  ✅ New best model saved! PSNR: {psnr:.2f}dB")

            # Periodic checkpoint
            if epoch % self.args.save_interval == 0:
                torch.save(
                    self.generator.state_dict(),
                    os.path.join(self.args.checkpoint_dir, f'generator_epoch_{epoch}.pth')
                )

        print(f"\n{'='*60}")
        print(f"Training complete! Best PSNR: {self.best_psnr:.2f}dB")
        print(f"Checkpoints saved to: {self.args.checkpoint_dir}")
        print(f"{'='*60}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='AXORA SRGAN Training')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to HR satellite image dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hr_size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4, 8])
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = SRGANTrainer(args)
    trainer.train()
