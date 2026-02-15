#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from plot_common import (
    add_box_strip,
    dedupe_legend,
    sanitize_filename,
    save_figure,
    set_theme,
    setup_matplotlib_agg,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot C121/I121 packing boxplots split by altloc-modeled vs no-altloc."
    )
    parser.add_argument("--csv", type=Path, default=Path("out/tables/b2m_ccp_cell_contacts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/figures"))
    parser.add_argument(
        "--altloc-col",
        type=str,
        default="region_altloc_residue_fraction",
        choices=["region_altloc_residue_fraction", "region_altloc_atom_fraction"],
        help="Column used to define altloc-modeled status (> threshold).",
    )
    parser.add_argument(
        "--altloc-threshold",
        type=float,
        default=0.0,
        help="Value above which a structure is considered Altloc modeled.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "loose_packing_score",
            "interface_region_bsa_top_images_sum",
            "crystal_contacts_atom_region_any_per_1k_region_atoms",
            "region_sasa_mean_atom",
            "region_neighbor_contacts_6A",
            "region_neighbor_contacts_4A",
            "region_nearest_symmetry_atom_distance",
            "region_b_iso_mean",
            "matthews_vm_est",
            "solvent_fraction_est",
        ],
    )
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(
        style="whitegrid",
        context="poster",
        rc={"axes.labelsize": 28, "xtick.labelsize": 22, "ytick.labelsize": 22},
    )

    df = pd.read_csv(args.csv)
    df = df[df["error"].fillna("") == ""].copy()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"].isin(["C 121", "I 121"])].copy()
    if df.empty:
        raise SystemExit("No valid C 121 / I 121 rows found.")
    if args.altloc_col not in df.columns:
        raise SystemExit(f"Missing altloc column: {args.altloc_col}")

    v = pd.to_numeric(df[args.altloc_col], errors="coerce").fillna(0.0)
    df["altloc_status"] = pd.Series(
        ["Altloc modeled" if x > args.altloc_threshold else "No altloc" for x in v], index=df.index
    )

    region_start = int(df["region_start"].iloc[0]) if "region_start" in df.columns else -1
    region_end = int(df["region_end"].iloc[0]) if "region_end" in df.columns else -1

    group_order = ["C 121", "I 121"]
    altloc_order = ["No altloc", "Altloc modeled"]
    palette = {"No altloc": "#d7d7d7", "Altloc modeled": "#8f8f8f"}

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

        fig, ax = plt.subplots(1, 1, figsize=(10.2, 8.2), dpi=220)
        add_box_strip(
            ax,
            data=metric_df,
            x="group",
            y=metric,
            order=group_order,
            hue="altloc_status",
            hue_order=altloc_order,
            palette=palette,
            box_width=0.62,
            box_linewidth=1.8,
            box_fliersize=0,
            box_dodge=True,
            box_saturation=1.0,
            box_legend=True,
            strip_color="black",
            strip_alpha=0.55,
            strip_jitter=0.11,
            strip_size=5.5,
            strip_dodge=True,
        )
        dedupe_legend(ax, altloc_order, title="")
        ax.set_ylabel(metric)
        out = args.output_dir / f"{sanitize_filename(metric)}_box_altloc.png"
        save_figure(fig, out)
        plt.close(fig)
        produced += 1
        print(f"Saved: {out}")
        for grp in group_order:
            for st in altloc_order:
                vals = pd.to_numeric(
                    metric_df.loc[(metric_df["group"] == grp) & (metric_df["altloc_status"] == st), metric],
                    errors="coerce",
                ).dropna()
                if len(vals) > 0:
                    print(f"  {grp} / {st}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}")

    if produced == 0:
        raise SystemExit("No plots produced.")

    print("")
    print(f"Altloc split column: {args.altloc_col} > {args.altloc_threshold}")
    print(f"Region window in data: {region_start}-{region_end}")
    print("Counts:")
    print(df.groupby(["group", "altloc_status"]).size().to_string())
    print(f"Total plots produced: {produced}")


if __name__ == "__main__":
    main()
