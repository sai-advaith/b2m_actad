#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

from plot_common import save_figure, set_theme, setup_matplotlib_agg, style_xticks_monospace


def prepare_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"space_group", "cosine", "cosine_pdb", "r_work", "r_work_pdb", "r_free", "r_free_pdb"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {csv_path}: {', '.join(missing)}")

    out = df.copy()
    out["Group"] = out["space_group"].astype(str).str.strip().replace({"C121": "C 121", "I121": "I 121"})
    out = out[out["Group"].isin(["C 121", "I 121"])].copy()
    if out.empty:
        raise SystemExit("No rows for groups 'C 121' / 'I 121'.")

    # Positive values indicate improvement relative to *_pdb as reference:
    # cosine: higher is better -> (cosine - cosine_pdb) / |cosine_pdb|
    # r_work/r_free: lower is better -> (r_*_pdb - r_*) / |r_*_pdb|
    out["Cosine Relative Improvement (%)"] = (
        100.0 * (pd.to_numeric(out["cosine"], errors="coerce") - pd.to_numeric(out["cosine_pdb"], errors="coerce"))
        / pd.to_numeric(out["cosine_pdb"], errors="coerce").abs()
    )
    out["R_work Relative Improvement (%)"] = (
        100.0 * (pd.to_numeric(out["r_work_pdb"], errors="coerce") - pd.to_numeric(out["r_work"], errors="coerce"))
        / pd.to_numeric(out["r_work_pdb"], errors="coerce").abs()
    )
    out["R_free Relative Improvement (%)"] = (
        100.0 * (pd.to_numeric(out["r_free_pdb"], errors="coerce") - pd.to_numeric(out["r_free"], errors="coerce"))
        / pd.to_numeric(out["r_free_pdb"], errors="coerce").abs()
    )
    return out


def plot_metric(df: pd.DataFrame, metric_col: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["C 121", "I 121"]
    palette = {"C 121": "#A6CEE3", "I 121": "#F4A6A6"}
    metric_df = df.dropna(subset=[metric_col]).copy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=220)
    sns.violinplot(
        data=metric_df,
        x="Group",
        y=metric_col,
        hue="Group",
        order=order,
        palette=palette,
        dodge=False,
        legend=False,
        inner=None,
        linewidth=1.8,
        cut=0,
        saturation=1.0,
        ax=ax,
    )
    sns.stripplot(
        data=metric_df,
        x="Group",
        y=metric_col,
        order=order,
        color="black",
        jitter=0.10,
        size=6,
        alpha=0.80,
        ax=ax,
    )
    ax.axhline(0.0, color="#666666", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    style_xticks_monospace(ax)
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relative-improvement violins from data/tables/b2m_group_metrics.csv.")
    parser.add_argument("--csv", type=Path, default=Path("data/tables/b2m_group_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/figures"))
    args = parser.parse_args()

    setup_matplotlib_agg()
    set_theme(style="whitegrid", context="poster")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_df(args.csv)

    specs = [
        (
            "Cosine Relative Improvement (%)",
            "Relative Improvement in Cosine (%)",
            "cosine_improvement_violin.png",
        ),
        (
            "R_work Relative Improvement (%)",
            r"Relative Improvement in $\mathrm{R}_{\mathrm{work}}$ (%)",
            "r_work_improvement_violin.png",
        ),
        (
            "R_free Relative Improvement (%)",
            r"Relative Improvement in $\mathrm{R}_{\mathrm{free}}$ (%)",
            "r_free_improvement_violin.png",
        ),
    ]
    for metric_col, ylabel, fname in specs:
        out_path = args.output_dir / fname
        plot_metric(df, metric_col, ylabel, out_path)
        print(f"Saved: {out_path}")

    for grp in ["C 121", "I 121"]:
        grp_df = df[df["Group"] == grp]
        print(f"{grp}: n={len(grp_df)}")
        for metric_col, _, _ in specs:
            vals = pd.to_numeric(grp_df[metric_col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  {metric_col}: mean={vals.mean():.3f}")


if __name__ == "__main__":
    main()
