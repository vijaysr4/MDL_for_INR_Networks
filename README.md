# MDL for Implicit Neural Representation Networks

Code for the micro study **"Implicit Neural Representations as Minimum Description Length Models for 2D Images"**.  
We reproduce an implicit neural representation (INR) image codec and analyze it through the lens of the **Minimum Description Length (MDL)** principle.

> Report and slides: INR based image compression with SIREN + positional encoding, 8 bit quantization, entropy coding, and MDL based model selection on the Kodak dataset.

---

## Overview

This project studies **implicit neural representations** (INRs) for 2D image compression. Instead of storing pixels directly, we overfit a coordinate based neural network
$f_{\theta} : (x, y) \mapsto (R, G, B)$

to each image and then compress the weights. We focus on:

1. **Reproducing the base INR codec** from  
   Strümpler et al., *Implicit Neural Representations for Image Compression* (ECCV 2022).

2. **Defining MDL style scores** that combine
   - a **model term** based on bitrate or parameter count, and
   - a **data term** based on reconstruction error (MSE / PSNR),

   and using these scores for **model selection** across different network widths.

On the Kodak dataset, the MDL analysis yields a **single optimal width** for the full quantized INR model, which is not obvious from standard PSNR vs bitrate curves alone.

---

## Main ideas

- **INR backbone**: SIREN network with positional encoding on input coordinates:
  - Input: normalized coordinates $(x, y)$
  - Positional encoding with $L = 16$ frequencies, scale $\sigma = 1.4$
  - 3 hidden layers, sine activations, $\omega_0 = 30$
  - Width sweep $M \in \{32, 64, 128, 256\}$
- **Two main variants**:
  1. **INR proxy (SIREN + PE)**  
     32 bit float weights, MDL model term approximated from parameter count per pixel.
  2. **Full INR (SIREN + PE + Quantization)**  
     8 bit uniform weight quantization + entropy coding, MDL model term uses the **true bitrate** (bits per pixel).

- **Metrics and analysis**:
  - PSNR vs training step (for both variants)  
  - PSNR vs bitrate (classical codecs vs INRs)  
  - MDL vs number of hidden units $M$

MDL produces a **single scalar optimum** (here at $M = 64$ for the full quantized INR) where PSNR vs bitrate only shows a smooth tradeoff.

---

## Repository structure

A possible layout for this repository:

```text
MDL_for_INR_Networks/
├── README.md
├── refs.bib                # Bibliography (MDL, INR compression, etc.)
├── report/                 # LaTeX report, figures, and slides
├── imgs/                   # Base paper figures, example reconstructions
├── data/                   # Place Kodak images here (not tracked)
│   └── kodak/
├── configs/                # Example config files for experiments
├── scripts/
│   ├── train_siren_pe.py           # Train SIREN + PE (proxy model)
│   ├── train_full_inr.py           # Train full INR + quantization
│   ├── eval_codecs.py              # PSNR / bitrate evaluation
│   └── compute_mdl.py              # MDL computation and plots
└── mdl_inr/                # Library code (models, datasets, utils)
    ├── models/
    ├── data/
    ├── mdlnet/
    └── ...
```
