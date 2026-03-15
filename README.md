# 🛰️ AXORA — Satellite Image Super-Resolution using GANs

<div align="center">

![Defense](https://img.shields.io/badge/Track-DEFENSE-blue?style=for-the-badge)
![Hard](https://img.shields.io/badge/Difficulty-HARD-red?style=for-the-badge)
![PS](https://img.shields.io/badge/PS_ID-PS__S7__03-green?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/GSA_Pan_India-Hackathon-purple?style=for-the-badge)

**Enhancing low-resolution satellite reconnaissance imagery for border surveillance using Generative Adversarial Networks**

</div>

---

## 👥 Team AXORA

| Member | Role |
|--------|------|
| Avishkar Satpute | ML Engineer — GAN Architecture |
| Yash Dhudat | Deep Learning — Training Pipeline |
| Shubhangi Sahane | Image Processing & Metrics |
| Nikita Shende | Web Demo & Integration |

**College**: Pravara Rural Engineering College, Loni

---

## 🎯 Problem Statement (PS_S7_03)

Low-resolution satellite images severely limit the accuracy of border surveillance systems:

- **Sensor limitations** → Poor capture quality at altitude
- **Bandwidth constraints** → Compressed, degraded transmissions  
- **Cost factors** → High-res satellites expensive to deploy

**Impact on defense**: Object detection failures, reduced situational awareness, impaired decision-making.

---

## 💡 Our Solution

A **GAN-based Super-Resolution system** specifically tuned for satellite imagery:

```
Low-Res Input → Preprocessing → SRGAN Generator → Discriminator Validation → High-Res Output
    64×64 px                    (16 Residual Blocks)   (Realism check)         256×256 px (4×)
```

**Key innovations**:
- Pixel Shuffle upsampling (no checkerboard artifacts)
- Combined loss: Pixel + Perceptual (VGG) + Adversarial  
- Tiled inference for large satellite images (no OOM)
- Real-time metrics: PSNR and SSIM computed live

---

## 🏗️ Architecture

```
SRGAN Generator
├── Initial Conv (9×9, 64 channels, PReLU)
├── 16 × Residual Blocks
│   ├── Conv 3×3 → BatchNorm → PReLU
│   └── Conv 3×3 → BatchNorm → (skip connection)
├── Post-residual Conv + BatchNorm
├── 2 × Upsample Blocks (Pixel Shuffle 2×)
└── Output Conv (9×9, 3 channels, Tanh)

Discriminator (VGG-style)
├── 8 × Conv Blocks (64→512 channels)
├── Adaptive Avg Pool (6×6)
└── FC Layers → Sigmoid
```

---

## 📊 Results

| Metric | Bicubic (Baseline) | **SRGAN (Ours)** | Improvement |
|--------|-------------------|-------------------|-------------|
| PSNR (dB) | ~28.5 | **~32.8** | **+4.3 dB** |
| SSIM | ~0.82 | **~0.91** | **+0.09** |
| Inference | — | **< 200ms** | Real-time |

*Results on satellite benchmark (UC Merced Land Use dataset, 4× scale)*

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.0 |
| SR Model | SRGAN + Real-ESRGAN |
| Image Processing | OpenCV, PIL/Pillow |
| Metrics | PSNR, SSIM (custom implementation) |
| Web Demo | Streamlit |
| Pretrained Weights | Real-ESRGAN x4plus |

---

## 🚀 Quick Start

### Option 1: Run Web Demo (Recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/axora-satellite-sr.git
cd axora-satellite-sr

# Install dependencies
pip install -r requirements.txt

# Launch demo
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option 2: Command Line Inference

```bash
# Single image
python inference.py --input satellite_image.png --output results/

# Batch processing
python inference.py --input ./images/ --output ./results/ --batch

# With custom checkpoint
python inference.py --input image.png --output results/ --checkpoint checkpoints/best_model.pth
```

### Option 3: Google Colab

Open `AXORA_Satellite_SR.ipynb` in Google Colab (free GPU provided).
All cells are self-contained and documented.

### Option 4: Train from Scratch

```bash
# Prepare dataset (HR satellite images)
mkdir -p data/satellite
# Add your HR satellite images (*.jpg, *.png, *.tif) to data/satellite/

# Train
python train.py \
    --dataset_path ./data/satellite \
    --epochs 100 \
    --batch_size 16 \
    --scale 4 \
    --checkpoint_dir ./checkpoints

# Monitor in terminal — PSNR/SSIM printed every epoch
```

---

## 📂 Project Structure

```
axora-sr/
├── app.py                   # Streamlit web demo
├── train.py                 # Full GAN training pipeline
├── inference.py             # CLI inference engine
├── requirements.txt
├── AXORA_Satellite_SR.ipynb # Google Colab notebook
│
├── models/
│   ├── __init__.py
│   └── srgan.py             # Generator + Discriminator + Perceptual Loss
│
├── utils/
│   ├── __init__.py
│   └── metrics.py           # PSNR + SSIM implementations
│
├── weights/                 # Pretrained model weights (download separately)
│   └── RealESRGAN_x4plus.pth
│
├── checkpoints/             # Training checkpoints
├── results/                 # Inference outputs
└── assets/                  # Sample images
```

---

## 🎥 Demo

The Streamlit web app provides:
- **Upload** any satellite image (JPG/PNG/TIF)
- **Real-time enhancement** with 4× upscaling
- **Live PSNR & SSIM** metrics vs bicubic baseline
- **Side-by-side** visual comparison
- **Download** enhanced images and metrics report

---

## 🔮 Future Scope

- Real-time deployment on edge devices (Jetson Nano/Xavier)
- Integration with YOLO for direct object detection pipeline
- Multi-spectral satellite image support (infrared + RGB)
- Cloud-based inference API (AWS/GCP)
- Video satellite stream enhancement

---

## 📄 References

1. Krishna et al. (2025) — *Satellite Image Super-Resolution Using GANs*, ResearchGate
2. KIT (2024) — *GAN-Based Dual Image Super Resolution for Satellite Imagery*
3. Pang et al. (2023) — *Stable Super-Resolution GAN (SSRGAN)*, Remote Sensing, MDPI
4. Wang et al. (2021/2023) — *Real-ESRGAN: Real-World Blind SR*, ICCV Workshop

---

<div align="center">
<b>🛰️ Team AXORA | GSA Pan India Hackathon | PS_S7_03 | Pravara Rural Engineering College, Loni</b>
</div>
