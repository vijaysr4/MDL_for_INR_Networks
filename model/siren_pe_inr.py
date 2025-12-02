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


IMAGES_DIR: str = "/users/eleves-b/2024/vijay-venkatesh.murugan/.cache/kagglehub/datasets/sherylmehta/kodak-dataset/versions/1"

PLOTS_DIR: str = os.path.join("plots", "SIREN_PE")
RECONS_DIR: str = os.path.join("reconstructed_imgs", "SIREN_PE")
RESULTS_DIR: str = os.path.join("results", "SIREN_PE")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS: int = 2000
BATCH_SIZE: int = 1024
LEARNING_RATE: float = 5e-4  # close to paper default

ALPHA: float = 1.0  # weight on model bpp term
BETA: float = 1.0   # weight on data term (log MSE)
EPS: float = 1e-12
BITS_PER_PARAM: int = 32

# log batch-wise performance every this many batches
LOG_EVERY: int = 50

# SIREN + positional encoding hyperparameters from the paper
NUM_HIDDEN_LAYERS: int = 3  # they use 3 hidden layers
OMEGA_0: float = 30.0       # first layer frequency scaling
NUM_FREQS: int = 16         # L for Kodak
SIGMA_PE: float = 1.4       # sigma in positional encoding
LAMBDA_L1: float = 1e-5     # L1 regularization strength

# Widths M to sweep over (you can adjust this list)
WIDTHS: List[int] = [32, 64, 128, 256]


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
    bpp_proxy: float


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
    - coords: [N, 2] in [-1, 1]^2
    - pixels: [N, 3] in [0, 1]
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


class PositionalEncoding(nn.Module):
    """
    NeRF style positional encoding:
    gamma(p) = (p, sin(sigma^0 pi p), cos(...), ..., sin(sigma^{L-1} pi p), cos(...))
    applied elementwise to p in R^2.
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
                # First layer init as in SIREN
                self.linear.weight.uniform_(-1.0 / self.linear.in_features, 1.0 / self.linear.in_features)
            else:
                # Subsequent layers
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
        # Hidden layers with omega_0 = 1.0 as in SIREN
        for _ in range(num_hidden_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, is_first=False, omega_0=1.0))

        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.final(x)


class SIRENPosEncNet(nn.Module):
    """
    Wrapper that applies positional encoding before the SIREN MLP.
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mse(
    model: nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 4096,
) -> float:
    """
    Compute MSE over all pixels and channels.
    """
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
    Reconstruct RGB image [H, W, 3] from a trained SIRENPosEncNet.
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


def train_one_model_siren_pe(
    coords: torch.Tensor,
    pixels: torch.Tensor,
    hidden_width: int,
) -> Tuple[ModelResult, nn.Module, List[Dict]]:
    N: int = coords.shape[0]
    C: int = pixels.shape[1]

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

    # model bits per pixel and related MDL model term are constant for this model
    P: int = count_parameters(model)
    bpp_proxy: float = (float(P) * float(BITS_PER_PARAM)) / float(N)
    mdl_model_const: float = ALPHA * bpp_proxy

    training_logs: List[Dict] = []
    global_step: int = 0

    epoch_iter = tqdm(
        range(NUM_EPOCHS),
        desc=f"SIREN+PE width={hidden_width}",
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

            batch_mse_val = mse.item()
            total_loss += batch_mse_val * float(end - start)

            # log every LOG_EVERY batches
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
                        "batch_mse": batch_mse_val,
                        "mdl_model": mdl_model_const,
                        "mdl_data": mdl_data_batch,
                        "mdl_total": mdl_total_batch,
                        "bpp_proxy": bpp_proxy,
                        "params": P,
                    }
                )

            global_step += 1
            batch_idx_in_epoch += 1

        avg_loss = total_loss / float(N)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            epoch_iter.set_postfix(mse=f"{avg_loss:.6f}")

    mse_final: float = compute_mse(model, coords, pixels)
    psnr_final: float = psnr_from_mse(mse_final)

    mdl_model: float = mdl_model_const
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
        bpp_proxy=bpp_proxy,
    )

    return result, model, training_logs


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
        print(f"\nTraining SIREN+PE model: layers={NUM_HIDDEN_LAYERS}, width={hidden_width}")
        result, model, training_logs = train_one_model_siren_pe(coords, pixels, hidden_width)
        all_results.append(result)

        # attach image name to training logs
        for log in training_logs:
            log["image"] = img_stem
        all_training_logs.extend(training_logs)

        img_rec = reconstruct_image(model, H, W)
        key = f"{img_stem}_SIREN_PE_W{hidden_width}"
        reconstructions[key] = img_rec

        out_img = (img_rec * 255.0).astype(np.uint8)
        out_pil = Image.fromarray(out_img, mode="RGB")
        out_pil.save(os.path.join(RECONS_DIR, f"{key}.png"))

    # Save per-model summary results to CSV per image
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
                "bpp_proxy": r.bpp_proxy,
            }
            for r in all_results
        ]
    )

    results_csv = os.path.join(RESULTS_DIR, f"{img_stem}_SIREN_PE_results.csv")
    df.to_csv(results_csv, index=False)
    print("\nResults (single image, SIREN+PE):")
    print(df)

    # Save batch-wise training logs
    if all_training_logs:
        df_train = pd.DataFrame(all_training_logs)
        training_csv = os.path.join(RESULTS_DIR, f"{img_stem}_SIREN_PE_training.csv")
        df_train.to_csv(training_csv, index=False)
        print(f"Training curves saved to: {training_csv}")

    # Plot rate distortion: bitrate proxy (bpp) vs PSNR
    plt.figure()
    for r in all_results:
        plt.scatter(r.bpp_proxy, r.psnr)
        label = f"W{r.hidden_width}"
        plt.text(r.bpp_proxy, r.psnr, label, fontsize=8)

    plt.xlabel("Model bitrate proxy (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"SIREN+PE rate distortion ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_SIREN_PE_bpp_vs_psnr.png"), dpi=150)
    plt.close()

    # Plot MDL total vs parameter count
    plt.figure()
    for r in all_results:
        plt.scatter(r.params, r.mdl_total)
        label = f"W{r.hidden_width}"
        plt.text(r.params, r.mdl_total, label, fontsize=8)

    plt.xlabel("Parameter count P")
    plt.ylabel("MDL total (model term + data term)")
    plt.title(f"SIREN+PE MDL vs model size ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_SIREN_PE_params_vs_mdl.png"), dpi=150)
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
