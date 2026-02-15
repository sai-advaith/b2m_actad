#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from plot_common import add_box_strip, save_figure, set_theme, setup_matplotlib_agg


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"pdb_id", "space_group", "resolution", "peg_concentration", "r_work_pdb", "r_free_pdb"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {csv_path}: {', '.join(missing)}")

    out = df.copy()
    out["Group"] = out["space_group"].astype(str).str.strip()
    out["Group"] = out["Group"].replace({"C121": "C 121", "I121": "I 121"})
    out["PDB"] = out["pdb_id"].astype(str).str.lower()
    out["Resolution (A)"] = pd.to_numeric(out["resolution"], errors="coerce")
    out["PEG (%)"] = pd.to_numeric(out["peg_concentration"], errors="coerce")
    out["R_work_pdb"] = pd.to_numeric(out["r_work_pdb"], errors="coerce")
    out["R_free_pdb"] = pd.to_numeric(out["r_free_pdb"], errors="coerce")

    # Data convention: -1 means unavailable PEG concentration.
    out.loc[out["PEG (%)"] < 0, "PEG (%)"] = pd.NA
    return out
def main() -> None:
    parser = argparse.ArgumentParser(description="Box+strip plots for group metrics from data/tables/b2m_group_metrics.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/tables/b2m_group_metrics.csv"),
        help="Input CSV path (default: data/tables/b2m_group_metrics.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures"),
        help="Directory for output images (default: out/figures)",
    )
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(
        style="whitegrid",
        context="poster",
        rc={
            "axes.labelsize": 34,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
        },
    )

    df = prepare_dataframe(args.csv)
    df = df[df["Group"].isin(["C 121", "I 121"])].copy()
    if df.empty:
        raise SystemExit("No rows for groups 'C 121' / 'I 121' in input CSV.")

    palette = {"C 121": "#A6CEE3", "I 121": "#F4A6A6"}
    order = ["C 121", "I 121"]

    import matplotlib.pyplot as plt

    metric_specs = [
        ("Resolution (A)", r"Resolution ($\mathrm{\AA}$)", "resolution_box.png"),
        ("PEG (%)", "PEG (%)", "peg_box.png"),
        ("R_work_pdb", r"$\mathrm{R}_{\mathrm{work}}$", "r_work_box.png"),
        ("R_free_pdb", r"$\mathrm{R}_{\mathrm{free}}$", "r_free_box.png"),
    ]
    for metric_col, ylabel, fname in metric_specs:
        metric_df = df.dropna(subset=[metric_col])
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=220)
        add_box_strip(
            ax,
            data=metric_df,
            x="Group",
            y=metric_col,
            order=order,
            hue="Group",
            hue_order=order,
            palette=palette,
            box_legend=False,
        )
        ax.set_ylabel(ylabel)
        out = args.output_dir / fname
        save_figure(fig, out)
        plt.close(fig)
        print(f"Saved: {out}")

    for grp in order:
        grp_df = df[df["Group"] == grp]
        print(f"{grp}: n={len(grp_df)}")
        for metric, _, _ in metric_specs:
            vals = pd.to_numeric(grp_df[metric], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  {metric}: n={len(vals)}, mean={vals.mean():.4f}")


if __name__ == "__main__":
    main()
