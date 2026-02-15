#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from plot_common import save_figure, set_theme, setup_matplotlib_agg, style_legend_monospace


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram of region crystal contacts by group.")
    parser.add_argument("--csv", type=Path, default=Path("out/tables/b2m_ccp_cell_contacts.csv"))
    parser.add_argument(
        "--metric",
        type=str,
        default="crystal_contacts_atom_region_any",
        choices=[
            "crystal_contacts_atom_region_any",
            "crystal_contacts_residue_region_any",
            "crystal_contacts_atom_region_any_per_1k_region_atoms",
            "crystal_contacts_residue_region_any_per_1k_region_atoms",
        ],
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/figures/region_contacts_hist.png"),
    )
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(style="whitegrid", context="poster")
    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"].isin(["C 121", "I 121"])]
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=220)
    palette = {"C 121": "#1f77b4", "I 121": "#d62728"}
    sns.histplot(
        data=df,
        x=args.metric,
        hue="group",
        bins=args.bins,
        element="step",
        fill=False,
        common_norm=False,
        stat="count",
        linewidth=2.0,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel(args.metric)
    ax.set_ylabel("Count")
    if ax.legend_ is not None:
        style_legend_monospace(ax, title="")
    save_figure(fig, args.output)

    region_start = int(df["region_start"].iloc[0]) if "region_start" in df.columns and not df.empty else -1
    region_end = int(df["region_end"].iloc[0]) if "region_end" in df.columns and not df.empty else -1
    print(f"Saved: {args.output}")
    print(f"Metric: {args.metric}; region window: {region_start}-{region_end}")
    for grp in ["C 121", "I 121"]:
        vals = df.loc[df["group"] == grp, args.metric]
        if len(vals) > 0:
            print(f"{grp}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}")


if __name__ == "__main__":
    main()
