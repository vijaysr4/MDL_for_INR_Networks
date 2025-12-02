import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Paths and global config
# =============================================================================

IMAGES_DIR: str = "/users/eleves-b/2024/vijay-venkatesh.murugan/.cache/kagglehub/datasets/sherylmehta/kodak-dataset/versions/1"

PLOTS_DIR: str = os.path.join("plots", "Full_inr")
RECONS_DIR: str = os.path.join("reconstructed_imgs", "Full_inr")
RESULTS_DIR: str = os.path.join("results", "Full_inr")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS: int = 2000
BATCH_SIZE: int = 1024
LEARNING_RATE: float = 5e-4

# MDL / description length style terms
ALPHA: float = 1.0   # weight on model cost term (bits per pixel)
BETA: float = 1.0    # weight on data term log(MSE)
EPS: float = 1e-12

# Quantization
WEIGHT_BITWIDTH: int = 8  # paper typically uses 7 or 8
BITS_PER_PARAM_FLOAT: int = 32

# Log batch wise performance every this many batches
LOG_EVERY: int = 50

# INR architecture settings (based on paper)
NUM_HIDDEN_LAYERS: int = 3
OMEGA_0: float = 30.0
NUM_FREQS: int = 16
SIGMA_PE: float = 1.4
LAMBDA_L1: float = 1e-5

# Optional post quantization fine tuning epochs (set >0 if you want)
QAT_EPOCHS: int = 0
QAT_LR: float = 1e-4

# Widths M to sweep over
WIDTHS: List[int] = [32, 64, 128, 256]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ModelResult:
    hidden_layers: int
    hidden_width: int
    params: int

    mse: float
    psnr: float

    mdl_model: float
    mdl_data: float
    mdl_total: float

    bpp_32: float
    bpp_quant: float
    bpp_entropy: float


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_paths(images_dir: str) -> List[str]:
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths: List[str] = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(exts):
            paths.append(os.path.join(images_dir, fname))
    return paths


def load_image_as_dataset(image_path: str) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Load an RGB image, normalize to [0,1], and return:
      coords: [N, 2] in [-1, 1]^2
      pixels: [N, 3] in [0, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    H, W, C = img_arr.shape

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    # normalize coordinates to [-1, 1]
    u = (xv / (W - 1)) * 2.0 - 1.0
    v = (yv / (H - 1)) * 2.0 - 1.0

    coords = np.stack([u, v], axis=-1).reshape(-1, 2)
    pixels = img_arr.reshape(-1, C)

    coords_tensor = torch.from_numpy(coords)
    pixels_tensor = torch.from_numpy(pixels)

    return coords_tensor, pixels_tensor, H, W


# =============================================================================
# SIREN + positional encoding architecture (paper version)
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    NeRF style positional encoding with sigma scaling, concatenated with original coords.
    """
    def __init__(self, in_dim: int = 2, num_freqs: int = NUM_FREQS, sigma: float = SIGMA_PE):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.sigma = sigma

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, in_dim]
        outs = [coords]
        for k in range(self.num_freqs):
            freq = (self.sigma ** k) * math.pi
            outs.append(torch.sin(freq * coords))
            outs.append(torch.cos(freq * coords))
        return torch.cat(outs, dim=-1)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (1 + 2 * self.num_freqs)


class SIRENLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.is_first = is_first
        self.omega_0 = omega_0
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.linear.in_features,
                    1.0 / self.linear.in_features,
                )
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        hidden_dim: int = 64,
        num_hidden_layers: int = NUM_HIDDEN_LAYERS,
        omega_0: float = OMEGA_0,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        # First SIREN layer with omega_0
        layers.append(SIRENLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
        # Hidden layers with omega_0 = 1.0
        for _ in range(num_hidden_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, is_first=False, omega_0=1.0))

        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.final(x)


class SIRENPosEncNet(nn.Module):
    """
    SIREN with positional encoding in front, as used in the paper.
    """
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 3,
        hidden_dim: int = 64,
        num_hidden_layers: int = NUM_HIDDEN_LAYERS,
        num_freqs: int = NUM_FREQS,
        sigma: float = SIGMA_PE,
        omega_0: float = OMEGA_0,
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(in_dim=in_dim, num_freqs=num_freqs, sigma=sigma)
        self.siren = SIRENNet(
            in_dim=self.pe.out_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            omega_0=omega_0,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        enc = self.pe(coords)
        return self.siren(enc)


# =============================================================================
# Quantization and entropy coding helpers
# =============================================================================

def quantize_tensor(tensor: torch.Tensor, num_bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per tensor uniform quantization to integers in [0, 2^num_bits - 1].
    Returns (q, scale, t_min) so that dequantization is:

      dq = t_min + scale * q
    """
    tensor = tensor.detach().cpu().to(torch.float32)
    t_min = tensor.min()
    t_max = tensor.max()

    if t_min == t_max:
        scale = torch.tensor(1.0, dtype=torch.float32)
        q = torch.zeros_like(tensor, dtype=torch.int32)
        return q, scale, t_min

    levels = 2 ** num_bits
    scale = (t_max - t_min) / (levels - 1)
    q = torch.round((tensor - t_min) / scale).to(torch.int32)
    return q, scale, t_min


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, t_min: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.float32)
    return t_min + scale * q


def quantize_model(
    model: nn.Module,
    num_bits: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], torch.Tensor]:
    """
    Quantize all trainable parameters of the model per tensor.
    Returns:
      q_state_dict: mapping param name -> dequantized float tensor to load into a model
      quant_info:   per parameter quantization stats (for debugging/analysis)
      all_q_values: 1D tensor of all integer codes of all parameters, to estimate entropy.
    """
    q_state_dict: Dict[str, torch.Tensor] = {}
    quant_info: Dict[str, Dict] = {}
    all_q_values: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        q, scale, t_min = quantize_tensor(param.data, num_bits)
        dq = dequantize_tensor(q, scale, t_min)

        q_state_dict[name] = dq
        quant_info[name] = {
            "scale": float(scale),
            "min": float(t_min),
            "num_bits": num_bits,
            "numel": param.numel(),
        }
        all_q_values.append(q.view(-1))

    if all_q_values:
        all_q_values_tensor = torch.cat(all_q_values, dim=0)
    else:
        all_q_values_tensor = torch.empty(0, dtype=torch.int32)

    return q_state_dict, quant_info, all_q_values_tensor


def compute_entropy_bits(
    q_values: torch.Tensor,
    num_bits: int,
    num_tensors: int,
) -> Tuple[float, float]:
    """
    Given all quantized integer values, estimate the ideal arithmetic coding cost.

      bits_total = H * N_symbols + overhead_for_scales

    Returns (bits_total, entropy_per_symbol).
    """
    if q_values.numel() == 0:
        return 0.0, 0.0

    q_flat = q_values.view(-1).to(torch.int64)
    levels = 2 ** num_bits
    counts = torch.bincount(q_flat, minlength=levels).to(torch.float64)
    total = counts.sum()
    probs = counts / total
    nonzero = probs > 0
    entropy_per_symbol = float(
        -(probs[nonzero] * torch.log2(probs[nonzero])).sum().item()
    )
    ideal_bits = entropy_per_symbol * float(total.item())

    # overhead to store (scale, min) per tensor in 32 bit floats
    overhead_bits = num_tensors * 2 * 32
    bits_total = ideal_bits + overhead_bits

    return bits_total, entropy_per_symbol


# =============================================================================
# Evaluation helpers
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mse(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 4096,
) -> float:
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total_loss: float = 0.0
    N: int = coords.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_coords = coords[start:end].to(DEVICE)
            batch_targets = targets[start:end].to(DEVICE)
            preds = model(batch_coords)
            total_loss += mse_loss(preds, batch_targets).item()

    mse: float = total_loss / float(N)
    return mse


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def reconstruct_image(
    model: nn.Module,
    H: int,
    W: int,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Reconstruct RGB image [H, W, 3] from a trained model.
    """
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    u = (xv / (W - 1)) * 2.0 - 1.0
    v = (yv / (H - 1)) * 2.0 - 1.0
    coords = np.stack([u, v], axis=-1).reshape(-1, 2)
    coords_tensor = torch.from_numpy(coords).to(device)

    model.eval()
    preds_list: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, coords_tensor.shape[0], 4096):
            end = min(start + 4096, coords_tensor.shape[0])
            batch = coords_tensor[start:end]
            out = model(batch)
            preds_list.append(out.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    img_rec = preds.reshape(H, W, -1)
    img_rec = np.clip(img_rec, 0.0, 1.0)
    return img_rec


# =============================================================================
# Single model training + compression pipeline
# =============================================================================

def train_one_model_full_inr(
    coords: torch.Tensor,
    pixels: torch.Tensor,
    hidden_width: int,
) -> Tuple[ModelResult, nn.Module, List[Dict]]:
    """
    Full INR compression pipeline for a single width:
      1. Overfit SIREN+PE with L1 regularization.
      2. Quantize weights (per tensor uniform quantization).
      3. Optional post quantization fine tuning (simplified QAT style).
      4. Estimate entropy coded model bitrate.

    Returns:
      ModelResult
      final quantized model
      training_logs (batch wise metrics during overfitting)
    """
    N: int = coords.shape[0]
    C: int = pixels.shape[1]
    N_pixels: int = N   # equal to H * W for 1 sample per pixel

    # Build float INR
    model = SIRENPosEncNet(
        in_dim=2,
        out_dim=C,
        hidden_dim=hidden_width,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_freqs=NUM_FREQS,
        sigma=SIGMA_PE,
        omega_0=OMEGA_0,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss_fn = nn.MSELoss()

    train_coords = coords.to(DEVICE)
    train_pixels = pixels.to(DEVICE)

    # The parameter count is fixed for this architecture
    P: int = count_parameters(model)
    bpp_32: float = (float(P) * float(BITS_PER_PARAM_FLOAT)) / float(N_pixels)
    bpp_quant: float = (float(P) * float(WEIGHT_BITWIDTH)) / float(N_pixels)

    # For MDL tracking during training we use quantized bpp as a fixed cost proxy
    mdl_model_const: float = ALPHA * bpp_quant

    training_logs: List[Dict] = []
    global_step: int = 0

    epoch_iter = tqdm(
        range(NUM_EPOCHS),
        desc=f"Full_inr width={hidden_width}",
        ncols=100,
    )

    for epoch in epoch_iter:
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        train_coords = train_coords[perm]
        train_pixels = train_pixels[perm]

        total_loss: float = 0.0
        batch_idx_in_epoch: int = 0

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch_coords = train_coords[start:end]
            batch_pixels = train_pixels[start:end]

            optimizer.zero_grad()
            preds = model(batch_coords)
            mse = mse_loss_fn(preds, batch_pixels)

            # L1 regularization on all model parameters
            l1_term = 0.0
            for p in model.parameters():
                l1_term = l1_term + p.abs().sum()

            loss = mse + LAMBDA_L1 * l1_term
            loss.backward()
            optimizer.step()

            batch_mse_val = float(mse.item())
            total_loss += batch_mse_val * float(end - start)

            # Log every LOG_EVERY batches
            if global_step % LOG_EVERY == 0:
                mdl_data_batch = BETA * math.log(batch_mse_val + EPS)
                mdl_total_batch = mdl_model_const + mdl_data_batch
                training_logs.append(
                    {
                        "hidden_layers": NUM_HIDDEN_LAYERS,
                        "hidden_width": hidden_width,
                        "epoch": epoch + 1,
                        "batch_idx_in_epoch": batch_idx_in_epoch,
                        "global_step": global_step,
                        "stage": "overfit",
                        "batch_mse": batch_mse_val,
                        "mdl_model": mdl_model_const,
                        "mdl_data": mdl_data_batch,
                        "mdl_total": mdl_total_batch,
                        "bpp_32": bpp_32,
                        "bpp_quant": bpp_quant,
                        "params": P,
                    }
                )

            global_step += 1
            batch_idx_in_epoch += 1

        avg_loss = total_loss / float(N)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            epoch_iter.set_postfix(mse=f"{avg_loss:.6f}")

    # Stage 2: quantize model weights
    q_state_dict, quant_info, q_values = quantize_model(model, WEIGHT_BITWIDTH)

    # Optional Stage 3: simple post quantization fine tuning
    # Here we just optionally fine tune the dequantized model and re quantize.
    q_model = SIRENPosEncNet(
        in_dim=2,
        out_dim=C,
        hidden_dim=hidden_width,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_freqs=NUM_FREQS,
        sigma=SIGMA_PE,
        omega_0=OMEGA_0,
    ).to(DEVICE)
    q_model.load_state_dict(q_state_dict)

    if QAT_EPOCHS > 0:
        qat_optimizer = optim.Adam(q_model.parameters(), lr=QAT_LR)
        for epoch in range(QAT_EPOCHS):
            q_model.train()
            perm = torch.randperm(N, device=DEVICE)
            coords_q = train_coords[perm]
            pixels_q = train_pixels[perm]
            total_loss_q: float = 0.0
            for start in range(0, N, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N)
                bc = coords_q[start:end]
                bp = pixels_q[start:end]
                qat_optimizer.zero_grad()
                preds = q_model(bc)
                mse_q = mse_loss_fn(preds, bp)
                # we can keep or drop L1 here; keeping is closer to objective
                l1_term_q = 0.0
                for p in q_model.parameters():
                    l1_term_q = l1_term_q + p.abs().sum()
                loss_q = mse_q + LAMBDA_L1 * l1_term_q
                loss_q.backward()
                qat_optimizer.step()
                total_loss_q += float(mse_q.item()) * float(end - start)

        # Re quantize after QAT
        q_state_dict, quant_info, q_values = quantize_model(q_model, WEIGHT_BITWIDTH)
        q_model.load_state_dict(q_state_dict)

    # Final compressed model
    final_model = q_model

    # Evaluate distortion
    mse_final: float = compute_mse(final_model, coords, pixels)
    psnr_final: float = psnr_from_mse(mse_final)

    # Compute entropy coded bitrate
    bits_total, entropy_per_symbol = compute_entropy_bits(
        q_values,
        WEIGHT_BITWIDTH,
        num_tensors=len(quant_info),
    )
    bpp_entropy: float = bits_total / float(N_pixels)

    mdl_model: float = ALPHA * bpp_entropy
    mdl_data: float = BETA * math.log(mse_final + EPS)
    mdl_total: float = mdl_model + mdl_data

    result = ModelResult(
        hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_width=hidden_width,
        params=P,
        mse=mse_final,
        psnr=psnr_final,
        mdl_model=mdl_model,
        mdl_data=mdl_data,
        mdl_total=mdl_total,
        bpp_32=bpp_32,
        bpp_quant=bpp_quant,
        bpp_entropy=bpp_entropy,
    )

    return result, final_model, training_logs


# =============================================================================
# Experiment driver
# =============================================================================

def run_experiment(image_path: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RECONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using image: {image_path}")
    coords, pixels, H, W = load_image_as_dataset(image_path)
    N: int = coords.shape[0]
    print(f"Image size: {W}x{H}, N = {N} pixels")

    all_results: List[ModelResult] = []
    reconstructions: Dict[str, np.ndarray] = {}
    all_training_logs: List[Dict] = []

    img_stem: str = os.path.splitext(os.path.basename(image_path))[0]

    for hidden_width in WIDTHS:
        print(f"\nTraining Full_inr model: layers={NUM_HIDDEN_LAYERS}, width={hidden_width}")
        result, final_model, training_logs = train_one_model_full_inr(
            coords,
            pixels,
            hidden_width,
        )
        all_results.append(result)

        # attach image name to training logs
        for log in training_logs:
            log["image"] = img_stem
        all_training_logs.extend(training_logs)

        img_rec = reconstruct_image(final_model, H, W)
        key = f"{img_stem}_Full_inr_W{hidden_width}"
        reconstructions[key] = img_rec

        out_img = (img_rec * 255.0).astype(np.uint8)
        out_pil = Image.fromarray(out_img, mode="RGB")
        out_pil.save(os.path.join(RECONS_DIR, f"{key}.png"))

    # Per model summary CSV
    df = pd.DataFrame(
        [
            {
                "image": img_stem,
                "hidden_layers": r.hidden_layers,
                "hidden_width": r.hidden_width,
                "params": r.params,
                "mse": r.mse,
                "psnr": r.psnr,
                "mdl_model": r.mdl_model,
                "mdl_data": r.mdl_data,
                "mdl_total": r.mdl_total,
                "bpp_32": r.bpp_32,
                "bpp_quant": r.bpp_quant,
                "bpp_entropy": r.bpp_entropy,
            }
            for r in all_results
        ]
    )

    results_csv = os.path.join(RESULTS_DIR, f"{img_stem}_Full_inr_results.csv")
    df.to_csv(results_csv, index=False)
    print("\nResults (single image, Full_inr):")
    print(df)

    # Batch wise training logs CSV
    if all_training_logs:
        df_train = pd.DataFrame(all_training_logs)
        training_csv = os.path.join(RESULTS_DIR, f"{img_stem}_Full_inr_training.csv")
        df_train.to_csv(training_csv, index=False)
        print(f"Training curves saved to: {training_csv}")

    # Plot rate distortion using entropy coded bitrate
    plt.figure()
    for r in all_results:
        plt.scatter(r.bpp_entropy, r.psnr)
        label = f"W{r.hidden_width}"
        plt.text(r.bpp_entropy, r.psnr, label, fontsize=8)

    plt.xlabel("Model bitrate (entropy coded, bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Full_inr rate distortion ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_Full_inr_bpp_vs_psnr.png"), dpi=150)
    plt.close()

    # Plot MDL total vs parameter count
    plt.figure()
    for r in all_results:
        plt.scatter(r.params, r.mdl_total)
        label = f"W{r.hidden_width}"
        plt.text(r.params, r.mdl_total, label, fontsize=8)

    plt.xlabel("Parameter count P")
    plt.ylabel("MDL total (model term + data term)")
    plt.title(f"Full_inr MDL vs model size ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_Full_inr_params_vs_mdl.png"), dpi=150)
    plt.close()

    print(f"\nSummary CSV saved to: {results_csv}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Reconstructions saved to: {RECONS_DIR}")


def main() -> None:
    set_seed(0)
    image_paths: List[str] = get_image_paths(IMAGES_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in directory: {IMAGES_DIR}")
    first_image: str = image_paths[0]
    run_experiment(first_image)

    # To run this on all images in IMAGES_DIR instead of only the first one:
    # for img_path in image_paths:
    #     run_experiment(img_path)


if __name__ == "__main__":
    main()
