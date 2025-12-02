import os
import glob
import math

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SIREN_RESULTS_DIR = os.path.join("results", "SIREN_PE")
FULL_RESULTS_DIR = os.path.join("results", "Full_inr")

FINAL_PLOTS_DIR = "model/final_plots"
os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)

# LaTeX-style look (no real TeX, just mathtext and serif fonts)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern", "DejaVu Serif", "Times New Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_summary_results():
    """Load all SIREN+PE and SIREN+PE+Quantization summary CSVs and return two DataFrames."""
    siren_files = glob.glob(os.path.join(SIREN_RESULTS_DIR, "*_SIREN_PE_results.csv"))
    full_files = glob.glob(os.path.join(FULL_RESULTS_DIR, "*_Full_inr_results.csv"))

    siren_dfs = [pd.read_csv(f) for f in siren_files]
    full_dfs = [pd.read_csv(f) for f in full_files]

    siren_df = pd.concat(siren_dfs, ignore_index=True) if siren_dfs else pd.DataFrame()
    full_df = pd.concat(full_dfs, ignore_index=True) if full_dfs else pd.DataFrame()
    return siren_df, full_df


def mse_to_psnr_series(mse_series, max_val=1.0):
    """Convert a pandas Series of MSE values to PSNR."""
    mse_clipped = mse_series.clip(lower=1e-12)
    # PSNR = 10 * log10(max_val^2 / mse) with max_val=1
    return -10.0 * mse_clipped.apply(math.log10)


# ---------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------

def plot_psnr_vs_bpp(df_siren, df_full, out_basename):
    """
    Plot PSNR vs bitrate for:
      - SIREN + PE (bpp_proxy)
      - SIREN + PE + Quantization (bpp_entropy)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    df_siren = df_siren.sort_values("bpp_proxy")
    df_full = df_full.sort_values("bpp_entropy")

    # SIREN + PE (32-bit)
    ax.plot(
        df_siren["bpp_proxy"],
        df_siren["psnr"],
        marker="o",
        linestyle="-",
        label="SIREN + PE (32-bit)",
    )
    for _, row in df_siren.iterrows():
        ax.annotate(
            f"M={int(row['hidden_width'])}",
            (row["bpp_proxy"], row["psnr"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # SIREN + PE + Quantization
    ax.plot(
        df_full["bpp_entropy"],
        df_full["psnr"],
        marker="s",
        linestyle="-",
        label="SIREN + PE + Quantization",
    )
    for _, row in df_full.iterrows():
        ax.annotate(
            f"M={int(row['hidden_width'])}",
            (row["bpp_entropy"], row["psnr"]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
        )

    ax.set_xlabel(r"Bitrate (bits per pixel)")
    ax.set_ylabel(r"PSNR (dB)")
    ax.set_title(r"Rate--distortion (PSNR vs bitrate)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(FINAL_PLOTS_DIR, f"{out_basename}_psnr_vs_bpp.png")
    fig.savefig(out_path)
    plt.close(fig)


def plot_mdl_vs_width(df_siren, df_full, out_basename):
    """
    Plot MDL_total vs hidden_width for:
      - SIREN + PE
      - SIREN + PE + Quantization

    Highlight SIREN + PE + Quantization with hidden_width == 64 (if present).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    df_siren = df_siren.sort_values("hidden_width")
    df_full = df_full.sort_values("hidden_width")

    ax.plot(
        df_siren["hidden_width"],
        df_siren["mdl_total"],
        marker="o",
        linestyle="-",
        label="SIREN + PE (32-bit)",
    )
    ax.plot(
        df_full["hidden_width"],
        df_full["mdl_total"],
        marker="s",
        linestyle="-",
        label="SIREN + PE + Quantization",
    )

    # Highlight M = 64 for quantized model if it exists
    highlight = df_full[df_full["hidden_width"] == 64]
    if not highlight.empty:
        x_val = highlight["hidden_width"].iloc[0]
        y_val = highlight["mdl_total"].iloc[0]
        ax.scatter(
            [x_val],
            [y_val],
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label="MDL minimum (M=64)",
        )

    ax.set_xlabel(r"Hidden units $M$")
    ax.set_ylabel(r"MDL total $L_{\mathrm{model}} + L_{\mathrm{data}}$")
    ax.set_title(r"MDL vs hidden units")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(FINAL_PLOTS_DIR, f"{out_basename}_mdl_vs_hidden_units.png")
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_psnr(training_csv_path, method_label, out_basename, x_max=2000.0):
    """
    From a training CSV, plot PSNR vs global_step, one curve per hidden_width.
    method_label is either:
      - 'SIREN + PE'
      - 'SIREN + PE + Quantization'
    """
    if not os.path.isfile(training_csv_path):
        print(f"[warn] Training CSV not found: {training_csv_path}, skipping training plot.")
        return

    df_train = pd.read_csv(training_csv_path)
    if "batch_mse" not in df_train.columns or "global_step" not in df_train.columns:
        print(f"[warn] Missing required columns in {training_csv_path}, skipping.")
        return

    widths = sorted(df_train["hidden_width"].unique())

    fig, ax = plt.subplots(figsize=(6, 4))

    for w in widths:
        sub = df_train[df_train["hidden_width"] == w].copy()
        sub = sub.sort_values("global_step")
        # Compute PSNR from batch MSE
        sub["psnr"] = mse_to_psnr_series(sub["batch_mse"])
        # Restrict to x_max
        sub = sub[sub["global_step"] <= x_max]
        if sub.empty:
            continue

        ax.plot(
            sub["global_step"],
            sub["psnr"],
            linestyle="-",
            marker=None,
            label=rf"{method_label}, $M={int(w)}$",
        )

    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"Training step")
    ax.set_ylabel(r"PSNR (dB, batch-wise)")
    ax.set_title(r"Convergence (PSNR vs training step)")
    ax.legend()
    fig.tight_layout()

    suffix = method_label.replace(" ", "_").replace("+", "plus")
    out_path = os.path.join(FINAL_PLOTS_DIR, f"{out_basename}_psnr_vs_step_{suffix}.png")
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    siren_df, full_df = load_summary_results()
    if siren_df.empty or full_df.empty:
        print("[error] Could not find SIREN_PE or Full_inr results. Check paths.")
        return

    siren_images = set(siren_df["image"].unique())
    full_images = set(full_df["image"].unique())
    common_images = sorted(siren_images.intersection(full_images))

    if not common_images:
        print("[error] No common images between SIREN_PE and Full_inr results.")
        return

    print(f"Found images: {common_images}")

    for img in common_images:
        df_s_img = siren_df[siren_df["image"] == img]
        df_f_img = full_df[full_df["image"] == img]

        # Use a generic basename for file naming that still encodes the image id
        out_basename = img

        # 1) PSNR vs BPP
        plot_psnr_vs_bpp(df_s_img, df_f_img, out_basename)

        # 2) MDL vs hidden units
        plot_mdl_vs_width(df_s_img, df_f_img, out_basename)

        # 3) Training PSNR vs step for both methods, limited to 2000 steps
        siren_train_csv = os.path.join(SIREN_RESULTS_DIR, f"{img}_SIREN_PE_training.csv")
        full_train_csv = os.path.join(FULL_RESULTS_DIR, f"{img}_Full_inr_training.csv")

        plot_training_psnr(
            full_train_csv,
            method_label="SIREN + PE + Quantization",
            out_basename=out_basename,
            x_max=2000.0,
        )
        plot_training_psnr(
            siren_train_csv,
            method_label="SIREN + PE",
            out_basename=out_basename,
            x_max=2000.0,
        )

        print(f"[info] Plots saved in {FINAL_PLOTS_DIR} for {img}")


if __name__ == "__main__":
    main()
