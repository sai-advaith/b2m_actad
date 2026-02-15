#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from plot_common import add_box_strip, save_figure, set_theme, setup_matplotlib_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Box plot of region crystal contacts by group.")
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
            "crystal_contacts_atom",
            "crystal_contacts_residue",
            "crystal_contacts_atom_per_1k_atoms",
            "crystal_contacts_residue_per_1k_atoms",
        ],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/figures/region_contacts_box.png"),
    )
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(style="whitegrid", context="poster")

    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"].isin(["C 121", "I 121"])].copy()
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])

    import matplotlib.pyplot as plt

    order = ["C 121", "I 121"]
    palette = {"C 121": "#A6CEE3", "I 121": "#F4A6A6"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=220)
    add_box_strip(
        ax,
        data=df,
        x="group",
        y=args.metric,
        order=order,
        hue="group",
        hue_order=order,
        palette=palette,
        box_legend=False,
        box_width=0.52,
    )
    ax.set_xlabel("")
    ax.set_ylabel(args.metric)
    save_figure(fig, args.output)

    region_start = int(df["region_start"].iloc[0]) if "region_start" in df.columns and not df.empty else -1
    region_end = int(df["region_end"].iloc[0]) if "region_end" in df.columns and not df.empty else -1
    print(f"Saved: {args.output}")
    print(f"Metric: {args.metric}; region window: {region_start}-{region_end}")
    for grp in order:
        vals = df.loc[df["group"] == grp, args.metric]
        if len(vals) > 0:
            print(f"{grp}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}")


if __name__ == "__main__":
    main()
