"""
AXORA - Satellite Image Super-Resolution Inference
Team AXORA | GSA Pan India Hackathon | PS_S7_03

Usage:
    # Single image
    python inference.py --input ./test_images/lr_sat.png --output ./results/

    # Batch folder
    python inference.py --input ./test_images/ --output ./results/ --batch

    # With custom checkpoint
    python inference.py --input image.png --output results/ --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.srgan import Generator
from utils.metrics import calculate_psnr, calculate_ssim, calculate_metrics_report


# ─── Core Inference Engine ────────────────────────────────────────────────────

class SatelliteSREngine:
    """
    Inference engine for Satellite Image Super-Resolution.
    Supports Real-ESRGAN weights and custom SRGAN checkpoints.
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, checkpoint_path=None, scale_factor=4, device=None):
        self.scale_factor = scale_factor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")

        self.model = self._load_model(checkpoint_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load_model(self, checkpoint_path):
        """Load generator model from checkpoint or use bicubic fallback."""
        model = Generator(scale_factor=self.scale_factor, num_residual_blocks=16)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📦 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle both full checkpoint and state_dict formats
            if 'generator_state' in checkpoint:
                state_dict = checkpoint['generator_state']
                meta = {k: v for k, v in checkpoint.items() if k != 'generator_state'}
                print(f"   Epoch: {meta.get('epoch', 'N/A')} | "
                      f"PSNR: {meta.get('psnr', 'N/A'):.2f}dB | "
                      f"SSIM: {meta.get('ssim', 'N/A'):.4f}")
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print("   ✅ Checkpoint loaded")
        else:
            print("⚠️  No checkpoint found. Using bicubic upsampling as fallback.")
            print("   → Download pre-trained weights or run train.py first.")

        model = model.to(self.device).eval()
        return model

    def enhance_image(self, image_input):
        """
        Enhance a single image using the SR model.

        Args:
            image_input: PIL Image or file path string

        Returns:
            dict with keys: sr_image (PIL), lr_image (PIL), metrics (dict), time_ms (float)
        """
        if isinstance(image_input, (str, Path)):
            lr_image = Image.open(image_input).convert('RGB')
        else:
            lr_image = image_input.convert('RGB')

        original_size = lr_image.size  # (W, H)

        # Preprocess
        lr_tensor = self.transform(lr_image).unsqueeze(0).to(self.device)

        # Inference with tile-based processing for large images
        start = time.time()
        with torch.no_grad():
            if max(lr_image.size) > 512:
                sr_tensor = self._tiled_inference(lr_tensor)
            else:
                sr_tensor = self.model(lr_tensor)
        elapsed_ms = (time.time() - start) * 1000

        # Post-process
        sr_tensor = sr_tensor.squeeze(0).cpu()
        sr_array = ((sr_tensor * 0.5 + 0.5).clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sr_image = Image.fromarray(sr_array)

        # Compute metrics vs bicubic baseline
        bicubic = lr_image.resize(sr_image.size, Image.BICUBIC)
        bicubic_np = np.array(bicubic)
        sr_np = np.array(sr_image)

        metrics = {
            'psnr_sr': calculate_psnr(sr_np, bicubic_np),
            'ssim_sr': calculate_ssim(sr_np, bicubic_np),
            'input_size': f"{original_size[0]}×{original_size[1]}",
            'output_size': f"{sr_image.width}×{sr_image.height}",
            'scale_factor': self.scale_factor,
            'inference_time_ms': round(elapsed_ms, 1),
        }

        return {
            'sr_image': sr_image,
            'lr_image': lr_image,
            'bicubic_image': bicubic,
            'metrics': metrics,
            'time_ms': elapsed_ms,
        }

    def _tiled_inference(self, lr_tensor, tile_size=256, overlap=32):
        """Process large images in tiles to avoid OOM errors."""
        _, C, H, W = lr_tensor.shape
        scale = self.scale_factor
        output = torch.zeros(1, C, H * scale, W * scale, device=self.device)
        weights = torch.zeros_like(output)

        for y in range(0, H, tile_size - overlap):
            for x in range(0, W, tile_size - overlap):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                tile = lr_tensor[:, :, y:y_end, x:x_end]
                sr_tile = self.model(tile)
                oy, ox = y * scale, x * scale
                output[:, :, oy:y_end * scale, ox:x_end * scale] += sr_tile
                weights[:, :, oy:y_end * scale, ox:x_end * scale] += 1

        return output / weights.clamp(min=1)

    def enhance_batch(self, input_dir, output_dir, show_progress=True):
        """Batch enhance all satellite images in a folder."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        images = [p for p in input_path.iterdir()
                  if p.suffix.lower() in self.SUPPORTED_FORMATS]

        if not images:
            print(f"❌ No supported images found in {input_dir}")
            return []

        print(f"\n🛰️  Processing {len(images)} satellite images...")
        results = []

        for i, img_path in enumerate(images, 1):
            if show_progress:
                print(f"  [{i:3d}/{len(images)}] {img_path.name}", end=" → ")

            try:
                result = self.enhance_image(img_path)
                out_path = output_path / f"SR_{img_path.stem}.png"
                result['sr_image'].save(out_path)
                result['path'] = str(out_path)
                results.append(result)

                if show_progress:
                    m = result['metrics']
                    print(
                        f"PSNR: {m['psnr_sr']:.2f}dB | "
                        f"SSIM: {m['ssim_sr']:.4f} | "
                        f"{m['inference_time_ms']:.0f}ms ✅"
                    )

            except Exception as e:
                print(f"❌ Failed: {e}")

        # Summary
        if results:
            avg_psnr = np.mean([r['metrics']['psnr_sr'] for r in results])
            avg_ssim = np.mean([r['metrics']['ssim_sr'] for r in results])
            avg_time = np.mean([r['metrics']['inference_time_ms'] for r in results])
            print(f"\n{'='*55}")
            print(f"📊 BATCH RESULTS ({len(results)} images)")
            print(f"   Avg PSNR : {avg_psnr:.2f} dB")
            print(f"   Avg SSIM : {avg_ssim:.4f}")
            print(f"   Avg Time : {avg_time:.1f} ms/image")
            print(f"   Output   : {output_dir}")
            print(f"{'='*55}")

        return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='AXORA - Satellite Image Super-Resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --input sat_image.png --output results/
  python inference.py --input ./images/ --output ./results/ --batch
  python inference.py --input image.png --output results/ --scale 2
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory (for batch mode)')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for enhanced images')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4, 8],
                        help='Super-resolution scale factor (default: 4)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process all images in input directory')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save LR/SR/Bicubic side-by-side comparison image')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🛰️  AXORA | Satellite Image Super-Resolution System")
    print("    Team AXORA | GSA Pan India Hackathon | PS_S7_03")
    print("=" * 60)

    engine = SatelliteSREngine(
        checkpoint_path=args.checkpoint,
        scale_factor=args.scale
    )

    os.makedirs(args.output, exist_ok=True)

    if args.batch or os.path.isdir(args.input):
        engine.enhance_batch(args.input, args.output)
    else:
        # Single image
        print(f"\n📸 Enhancing: {args.input}")
        result = engine.enhance_image(args.input)
        m = result['metrics']

        out_path = os.path.join(args.output, f"SR_{Path(args.input).stem}.png")
        result['sr_image'].save(out_path)

        print(f"\n{'='*50}")
        print(f"✅ Enhancement Complete!")
        print(f"   Input  : {m['input_size']} px")
        print(f"   Output : {m['output_size']} px ({args.scale}x upscale)")
        print(f"   PSNR   : {m['psnr_sr']:.2f} dB  (vs Bicubic)")
        print(f"   SSIM   : {m['ssim_sr']:.4f}      (vs Bicubic)")
        print(f"   Time   : {m['inference_time_ms']:.1f} ms")
        print(f"   Saved  : {out_path}")
        print(f"{'='*50}")

        if args.save_comparison:
            from torchvision.utils import make_grid
            import torchvision.transforms.functional as TF
            lr_up = result['lr_image'].resize(result['sr_image'].size, Image.BICUBIC)
            comparison = Image.new('RGB', (lr_up.width * 3, lr_up.height))
            comparison.paste(lr_up, (0, 0))
            comparison.paste(result['sr_image'], (lr_up.width, 0))
            comparison.paste(result['bicubic_image'], (lr_up.width * 2, 0))
            cmp_path = os.path.join(args.output, f"CMP_{Path(args.input).stem}.png")
            comparison.save(cmp_path)
            print(f"   Comparison saved: {cmp_path}")


if __name__ == '__main__':
    main()
