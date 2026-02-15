#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from plot_common import add_box_strip, sanitize_filename, save_figure, set_theme, setup_matplotlib_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot C121 vs I121 boxplots for CCP packing metrics.")
    parser.add_argument("--csv", type=Path, default=Path("out/tables/b2m_ccp_cell_contacts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/figures"))
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "loose_packing_score",
            "interface_region_bsa_top_images_sum",
            "crystal_contacts_atom_region_any_per_1k_region_atoms",
            "region_sasa_mean_atom",
            "region_neighbor_contacts_6A",
            "region_nearest_symmetry_atom_distance",
            "matthews_vm_est",
            "solvent_fraction_est",
        ],
        help="Metrics to plot (default: key local packing metrics).",
    )
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(
        style="whitegrid",
        context="poster",
        rc={
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        },
    )

    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"].isin(["C 121", "I 121"])].copy()
    if df.empty:
        raise SystemExit("No valid C 121 / I 121 rows found.")

    region_start = int(df["region_start"].iloc[0]) if "region_start" in df.columns else -1
    region_end = int(df["region_end"].iloc[0]) if "region_end" in df.columns else -1

    order = ["C 121", "I 121"]
    palette = {"C 121": "#A6CEE3", "I 121": "#F4A6A6"}

    import matplotlib.pyplot as plt

    produced = 0
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipped (missing column): {metric}")
            continue
        metric_df = df.copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric])
        if metric_df.empty:
            print(f"Skipped (all NaN): {metric}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=220)
        add_box_strip(
            ax,
            data=metric_df,
            x="group",
            y=metric,
            order=order,
            hue="group",
            hue_order=order,
            palette=palette,
            box_legend=False,
        )
        ax.set_ylabel(metric)
        out = args.output_dir / f"{sanitize_filename(metric)}_box.png"
        save_figure(fig, out)
        plt.close(fig)
        produced += 1
        print(f"Saved: {out}")
        for grp in order:
            vals = pd.to_numeric(metric_df.loc[metric_df["group"] == grp, metric], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  {grp}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}")

    if produced == 0:
        raise SystemExit("No plots produced.")

    print(f"Region window in data: {region_start}-{region_end}")
    print(f"Total plots produced: {produced}")


if __name__ == "__main__":
    main()
