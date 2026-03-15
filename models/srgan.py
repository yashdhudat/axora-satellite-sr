"""
SRGAN Model Architecture
Team AXORA - Satellite Image Super-Resolution
GSA Pan India Hackathon | PS_S7_03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for Generator network."""
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """Pixel shuffle upsampling block."""
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    SRGAN Generator: Upscales low-resolution satellite images to high-resolution.
    Architecture: Conv → N×ResBlocks → Conv → M×Upsample → Conv
    """
    def __init__(self, scale_factor=4, num_residual_blocks=16, channels=64):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )

        # Post-residual conv
        self.post_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Upsampling stages (2x per stage)
        upsample_stages = int(torch.log2(torch.tensor(scale_factor)).item())
        self.upsample = nn.Sequential(
            *[UpsampleBlock(channels, 2) for _ in range(upsample_stages)]
        )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        initial_features = self.initial(x)
        residual_out = self.residual_blocks(initial_features)
        post_res = self.post_residual(residual_out) + initial_features
        upsampled = self.upsample(post_res)
        return self.output_layer(upsampled)


class DiscriminatorBlock(nn.Module):
    """Discriminator convolutional block with optional batch norm."""
    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    SRGAN Discriminator: Distinguishes real HR from generated SR satellite images.
    Uses VGG-style architecture for perceptual feature extraction.
    """
    def __init__(self, input_size=96):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            DiscriminatorBlock(3,   64,  stride=1, use_bn=False),
            DiscriminatorBlock(64,  64,  stride=2),
            DiscriminatorBlock(64,  128, stride=1),
            DiscriminatorBlock(128, 128, stride=2),
            DiscriminatorBlock(128, 256, stride=1),
            DiscriminatorBlock(256, 256, stride=2),
            DiscriminatorBlock(256, 512, stride=1),
            DiscriminatorBlock(512, 512, stride=2),
        )

        # Adaptive pooling so any input size works
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for sharper, more realistic super-resolution.
    Compares feature maps rather than pixel-level MSE for better textures.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=False)
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        except Exception:
            self.feature_extractor = None

        self.mse = nn.MSELoss()

    def forward(self, sr, hr):
        if self.feature_extractor is not None:
            sr_features = self.feature_extractor(sr)
            hr_features = self.feature_extractor(hr)
            return self.mse(sr_features, hr_features)
        return self.mse(sr, hr)


def build_srgan(scale_factor=4, num_residual_blocks=16):
    """Factory function to build Generator + Discriminator pair."""
    generator = Generator(scale_factor=scale_factor, num_residual_blocks=num_residual_blocks)
    discriminator = Discriminator()
    return generator, discriminator


if __name__ == "__main__":
    # Quick architecture test
    print("=== AXORA SRGAN Architecture Test ===")
    gen, disc = build_srgan(scale_factor=4)

    lr_input = torch.randn(1, 3, 64, 64)
    sr_output = gen(lr_input)
    disc_score = disc(sr_output)

    print(f"Generator   : Input {tuple(lr_input.shape)} → Output {tuple(sr_output.shape)}")
    print(f"Discriminator: Input {tuple(sr_output.shape)} → Score {tuple(disc_score.shape)}")
    print(f"Scale Factor: 4x ({64}px → {256}px)")

    total_params = sum(p.numel() for p in gen.parameters())
    print(f"Generator Parameters: {total_params:,}")
    print("✅ Architecture OK")
