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
PLOTS_DIR: str = "plots"
RECONS_DIR: str = "reconstructed_imgs"
RESULTS_CSV: str = "results_single_image.csv"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS: int = 2000
BATCH_SIZE: int = 1024
LEARNING_RATE: float = 1e-3

ALPHA: float = 1.0
BETA: float = 1.0
EPS: float = 1e-8
BITS_PER_PARAM: int = 32

ARCH_CONFIGS: List[Tuple[int, int]] = [
    (2, 32),
    (2, 64),
    (2, 128),
    (3, 64),
    (3, 128),
    (4, 128),
]


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
    img = Image.open(image_path).convert("L")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    H, W = img_arr.shape

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    u = (xv + 0.5) / W
    v = (yv + 0.5) / H

    coords = np.stack([u, v], axis=-1).reshape(-1, 2)
    pixels = img_arr.reshape(-1, 1)

    coords_tensor = torch.from_numpy(coords)
    pixels_tensor = torch.from_numpy(pixels)

    return coords_tensor, pixels_tensor, H, W


class INRNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_layer = nn.ReLU if activation.lower() == "relu" else nn.SiLU

        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_layer())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_layer())
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    u = (xv + 0.5) / W
    v = (yv + 0.5) / H
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
    img_rec = preds.reshape(H, W)
    img_rec = np.clip(img_rec, 0.0, 1.0)
    return img_rec


def train_one_model(
    coords: torch.Tensor,
    pixels: torch.Tensor,
    hidden_layers: int,
    hidden_width: int,
) -> Tuple[ModelResult, nn.Module]:
    N: int = coords.shape[0]

    model = INRNet(
        in_dim=2,
        out_dim=1,
        hidden_dim=hidden_width,
        num_hidden_layers=hidden_layers,
        activation="relu",
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_coords = coords.to(DEVICE)
    train_pixels = pixels.to(DEVICE)

    epoch_iter = tqdm(
        range(NUM_EPOCHS),
        desc=f"Train L={hidden_layers},W={hidden_width}",
        ncols=100,
    )

    for epoch in epoch_iter:
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        train_coords = train_coords[perm]
        train_pixels = train_pixels[perm]

        total_loss: float = 0.0
        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch_coords = train_coords[start:end]
            batch_pixels = train_pixels[start:end]

            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = loss_fn(preds, batch_pixels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * float(end - start)

        avg_loss = total_loss / float(N)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            epoch_iter.set_postfix(mse=f"{avg_loss:.6f}")

    mse: float = compute_mse(model, coords, pixels)
    psnr: float = psnr_from_mse(mse)

    P: int = count_parameters(model)
    mdl_model: float = ALPHA * float(P)
    mdl_data: float = BETA * float(N) * math.log(mse + EPS)
    mdl_total: float = mdl_model + mdl_data
    bpp_proxy: float = (float(P) * float(BITS_PER_PARAM)) / float(N)

    result = ModelResult(
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        params=P,
        mse=mse,
        psnr=psnr,
        mdl_model=mdl_model,
        mdl_data=mdl_data,
        mdl_total=mdl_total,
        bpp_proxy=bpp_proxy,
    )

    return result, model


def run_experiment(image_path: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RECONS_DIR, exist_ok=True)

    print(f"Using image: {image_path}")
    coords, pixels, H, W = load_image_as_dataset(image_path)
    N: int = coords.shape[0]
    print(f"Image size: {W}x{H}, N = {N} pixels")

    all_results: List[ModelResult] = []
    reconstructions: Dict[str, np.ndarray] = {}

    img_stem: str = os.path.splitext(os.path.basename(image_path))[0]

    for hidden_layers, hidden_width in ARCH_CONFIGS:
        print(f"\nTraining model: layers={hidden_layers}, width={hidden_width}")
        result, model = train_one_model(coords, pixels, hidden_layers, hidden_width)
        all_results.append(result)

        img_rec = reconstruct_image(model, H, W)
        key = f"{img_stem}_L{hidden_layers}_W{hidden_width}"
        reconstructions[key] = img_rec

        out_img = (img_rec * 255.0).astype(np.uint8)
        out_pil = Image.fromarray(out_img, mode="L")
        out_pil.save(os.path.join(RECONS_DIR, f"{key}.png"))

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

    df.to_csv(RESULTS_CSV, index=False)
    print("\nResults (single image):")
    print(df)

    plt.figure()
    for r in all_results:
        plt.scatter(r.bpp_proxy, r.psnr)
        label = f"L{r.hidden_layers},W{r.hidden_width}"
        plt.text(r.bpp_proxy, r.psnr, label, fontsize=8)

    plt.xlabel("Bitrate proxy (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Rate distortion: INR bitrate proxy vs PSNR ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_bpp_vs_psnr.png"), dpi=150)
    plt.close()

    plt.figure()
    for r in all_results:
        plt.scatter(r.params, r.mdl_total)
        label = f"L{r.hidden_layers},W{r.hidden_width}"
        plt.text(r.params, r.mdl_total, label, fontsize=8)

    plt.xlabel("Parameter count P")
    plt.ylabel("MDL total (L_model + L_data)")
    plt.title(f"MDL score vs model size ({img_stem})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{img_stem}_params_vs_mdl.png"), dpi=150)
    plt.close()

    print(f"\nCSV saved to: {RESULTS_CSV}")
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
