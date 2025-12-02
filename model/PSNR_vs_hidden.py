import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SIREN_RESULTS_DIR = os.path.join("results", "SIREN_PE")
FULL_RESULTS_DIR = os.path.join("results", "Full_inr")

FINAL_PLOTS_DIR = "final_plots"
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


def plot_psnr_vs_hidden_units(df_siren, df_full, out_basename):
    """
    For a single image:
      x axis: hidden_width (hidden units M)
      y axis: PSNR

    Two curves:
      - SIREN + PE (unquantized, 32-bit float parameters)
      - SIREN + PE + Quantization (compressed weights)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    df_siren = df_siren.sort_values("hidden_width")
    df_full = df_full.sort_values("hidden_width")

    # SIREN + PE (unquantized)
    ax.plot(
        df_siren["hidden_width"],
        df_siren["psnr"],
        marker="o",
        linestyle="-",
        label="SIREN + PE (unquantized)",
    )

    # SIREN + PE + Quantization
    ax.plot(
        df_full["hidden_width"],
        df_full["psnr"],
        marker="s",
        linestyle="-",
        label="SIREN + PE + Quantization",
    )

    ax.set_xlabel(r"Hidden units $M$")
    ax.set_ylabel(r"PSNR (dB)")
    ax.set_title(r"PSNR vs hidden units")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(FINAL_PLOTS_DIR, f"{out_basename}_psnr_vs_hidden_units.png")
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

        out_basename = img  # only used in file name, not inside figure text
        plot_psnr_vs_hidden_units(df_s_img, df_f_img, out_basename)

        print(f"[info] PSNR vs hidden_units plot saved in {FINAL_PLOTS_DIR} for {img}")


if __name__ == "__main__":
    main()
