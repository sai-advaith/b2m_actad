#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from plot_common import add_box_strip, save_figure, set_theme, setup_matplotlib_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Box plot of cell volume by group.")
    parser.add_argument("--csv", type=Path, default=Path("out/tables/b2m_ccp_cell_contacts.csv"))
    parser.add_argument("-o", "--output", type=Path, default=Path("out/figures/cell_volume_box.png"))
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(
        style="whitegrid",
        context="poster",
        rc={"axes.labelsize": 30, "xtick.labelsize": 26, "ytick.labelsize": 26},
    )

    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"].isin(["C 121", "I 121"])].copy()
    df["cell_volume"] = pd.to_numeric(df["cell_volume"], errors="coerce")
    df = df.dropna(subset=["cell_volume"])
    if df.empty:
        raise SystemExit("No valid rows for cell_volume.")

    import matplotlib.pyplot as plt

    order = ["C 121", "I 121"]
    palette = {"C 121": "#A6CEE3", "I 121": "#F4A6A6"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=220)
    add_box_strip(
        ax,
        data=df,
        x="group",
        y="cell_volume",
        order=order,
        hue="group",
        hue_order=order,
        palette=palette,
        box_legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"Cell volume ($\mathrm{\AA}^3$)")
    save_figure(fig, args.output)
    plt.close(fig)

    for grp in order:
        vals = pd.to_numeric(df.loc[df["group"] == grp, "cell_volume"], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"{grp}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
