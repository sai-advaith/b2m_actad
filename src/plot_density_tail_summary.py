#!/usr/bin/env python3
import argparse
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import seaborn as sns

from plot_common import save_figure, set_theme, setup_matplotlib_agg, style_legend_monospace


def load_density_values(ccp4_path: Path) -> np.ndarray:
    ccp4 = gemmi.read_ccp4_map(str(ccp4_path), setup=False)
    values = np.array(ccp4.grid, copy=False).ravel()
    values = values[np.isfinite(values)]
    return values


def pdb_id_label(ccp4_path: Path) -> str:
    return ccp4_path.stem.split("_")[0].lower()


def normalize_values(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "minmax":
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    if mode == "zscore":
        mean = float(values.mean())
        std = float(values.std())
        if std <= 0:
            return np.zeros_like(values)
        return (values - mean) / std
    raise ValueError(f"Unknown normalization mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-panel summary for density concentration/spread by group."
    )
    parser.add_argument("maps", type=Path, nargs="+", help="Input CCP4 map files.")
    parser.add_argument("--top-pdbs", nargs="*", required=True, help="Top/C121 PDB IDs.")
    parser.add_argument("--bottom-pdbs", nargs="*", required=True, help="Bottom/I121 PDB IDs.")
    parser.add_argument("--top-label", default="C 121", help="Display label for top group.")
    parser.add_argument("--bottom-label", default="I 121", help="Display label for bottom group.")
    parser.add_argument("--top-color", default="#1f77b4", help="Color for top group.")
    parser.add_argument("--bottom-color", default="#d62728", help="Color for bottom group.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Tail threshold for right panel metric P(density > threshold).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Optional multiple thresholds for a sweep plot of P(density > t).",
    )
    parser.add_argument(
        "--drop-lowest-percentile",
        type=float,
        default=25.0,
        help="Drop the lowest N percentile per map before plotting/metrics.",
    )
    parser.add_argument(
        "--nonzero",
        action="store_true",
        help="Keep only values > 0 before percentile filtering.",
    )
    parser.add_argument("--x-min", type=float, default=0.0, help="Lower x limit.")
    parser.add_argument("--x-max", type=float, default=2.5, help="Upper x limit.")
    parser.add_argument(
        "--density-unit",
        type=str,
        default="e/A^3",
        help="Unit label for density thresholds on x-axis (default: e/A^3).",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "minmax", "zscore"],
        default="none",
        help="Per-map normalization before filtering/metrics (default: none).",
    )
    parser.add_argument("--log-y-min", type=float, default=1e-4, help="Lower y limit in log space.")
    parser.add_argument("--n-grid", type=int, default=700, help="Grid size for survival curves.")
    parser.add_argument(
        "--right-only",
        action="store_true",
        help="Plot only the right panel (tail metric panel).",
    )
    parser.add_argument(
        "--right-log-y",
        action="store_true",
        help="Use log scale on the right-panel y-axis.",
    )
    parser.add_argument(
        "--right-log-y-min",
        type=float,
        default=1e-3,
        help="Minimum y-value for right-panel log scale (default: 1e-3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/figures/density_tail_summary.png"),
    )
    args = parser.parse_args()

    setup_matplotlib_agg(cache_dir=Path("cache/mplcache"))
    import matplotlib.pyplot as plt

    top_set = {p.lower() for p in args.top_pdbs}
    bottom_set = {p.lower() for p in args.bottom_pdbs}

    map_to_values: dict[str, np.ndarray] = {}
    metric_rows = []

    for map_path in args.maps:
        map_id = pdb_id_label(map_path)
        values = load_density_values(map_path)

        # Preserve the intended semantics of --nonzero as raw-density > 0 filtering.
        if args.nonzero:
            values = values[values > 0]

        values = normalize_values(values, args.normalize)

        p = args.drop_lowest_percentile
        if 0 < p < 100 and len(values) > 0:
            lo = float(np.percentile(values, p))
            values = values[values > lo]

        if len(values) == 0:
            continue
        map_to_values[map_id] = values

        if map_id in top_set:
            group = args.top_label
        elif map_id in bottom_set:
            group = args.bottom_label
        else:
            continue

        metric_rows.append(
            {
                "map": map_id,
                "group": group,
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }
        )

    top_maps = sorted([m for m in map_to_values if m in top_set])
    bottom_maps = sorted([m for m in map_to_values if m in bottom_set])
    if not top_maps or not bottom_maps:
        raise SystemExit("Missing top or bottom maps after filtering.")

    x_grid = np.linspace(args.x_min, args.x_max, args.n_grid)

    def survival_matrix(map_ids: list[str]) -> np.ndarray:
        mats = []
        for map_id in map_ids:
            vals = np.sort(map_to_values[map_id])
            cdf = np.searchsorted(vals, x_grid, side="right") / vals.size
            mats.append(1.0 - cdf)
        return np.vstack(mats)

    top_surv = survival_matrix(top_maps)
    bottom_surv = survival_matrix(bottom_maps)

    top_mean = top_surv.mean(axis=0)
    bottom_mean = bottom_surv.mean(axis=0)
    top_std = top_surv.std(axis=0)
    bottom_std = bottom_surv.std(axis=0)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise SystemExit("No group metrics available after filtering.")

    if args.thresholds:
        thresholds = sorted({float(t) for t in args.thresholds})
    else:
        thresholds = [float(args.threshold)]

    tail_rows = []
    for map_id, values in map_to_values.items():
        if map_id in top_set:
            group = args.top_label
        elif map_id in bottom_set:
            group = args.bottom_label
        else:
            continue
        for t in thresholds:
            tail_rows.append(
                {"map": map_id, "group": group, "threshold": t, "tail_prob": float((values > t).mean())}
            )
    tail_df = pd.DataFrame(tail_rows)
    if tail_df.empty:
        raise SystemExit("No tail probability values available after filtering.")

    set_theme(style="whitegrid", context="poster")
    if args.right_only:
        fig, ax1 = plt.subplots(1, 1, figsize=(9.2, 7.0), dpi=170)
    else:
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            figsize=(15, 6.2),
            dpi=170,
            gridspec_kw={"width_ratios": [1.9, 1.0]},
        )

        ax0.plot(x_grid, top_mean, color=args.top_color, lw=2.5, label=f"{args.top_label} (n={len(top_maps)})")
        ax0.fill_between(
            x_grid,
            np.clip(top_mean - top_std, args.log_y_min, 1.0),
            np.clip(top_mean + top_std, args.log_y_min, 1.0),
            color=args.top_color,
            alpha=0.2,
        )
        ax0.plot(
            x_grid,
            bottom_mean,
            color=args.bottom_color,
            lw=2.5,
            label=f"{args.bottom_label} (n={len(bottom_maps)})",
        )
        ax0.fill_between(
            x_grid,
            np.clip(bottom_mean - bottom_std, args.log_y_min, 1.0),
            np.clip(bottom_mean + bottom_std, args.log_y_min, 1.0),
            color=args.bottom_color,
            alpha=0.2,
        )
        for t in thresholds:
            ax0.axvline(t, color="#6f6f6f", lw=1.0, ls="--", alpha=0.35)
        ax0.set_xlim(args.x_min, args.x_max)
        ax0.set_ylim(args.log_y_min, 1.0)
        ax0.set_yscale("log")
        ax0.set_xlabel(f"Density ({args.density_unit})")
        ax0.set_ylabel("1 - CDF (log scale)")
        ax0.set_title("Tail Emphasis (higher = more high-density mass)")
        ax0.legend(frameon=False)
        style_legend_monospace(ax0, title="")

    if args.right_log_y:
        tail_df = tail_df.copy()
        tail_df["tail_prob"] = np.clip(tail_df["tail_prob"], args.right_log_y_min, 1.0)

    if len(thresholds) == 1:
        t0 = thresholds[0]
        single_df = tail_df[tail_df["threshold"] == t0].copy()
        palette = {args.top_label: args.top_color, args.bottom_label: args.bottom_color}
        sns.boxplot(
            data=single_df,
            x="group",
            y="tail_prob",
            hue="group",
            order=[args.top_label, args.bottom_label],
            palette=palette,
            dodge=False,
            legend=False,
            width=0.55,
            fliersize=0,
            linewidth=1.2,
            ax=ax1,
        )
        sns.stripplot(
            data=single_df,
            x="group",
            y="tail_prob",
            order=[args.top_label, args.bottom_label],
            color="black",
            size=5.5,
            jitter=0.11,
            alpha=0.9,
            ax=ax1,
        )
        ax1.set_xlabel("")
        ax1.set_ylabel(f"P(density > {t0:g})")
        ax1.set_title("")
        if args.right_log_y:
            ax1.set_yscale("log")
            ax1.set_ylim(args.right_log_y_min, 1.0)

        for tick in ax1.get_xticklabels():
            tick.set_fontfamily("monospace")
    else:
        order = [args.top_label, args.bottom_label]
        for group, color in ((args.top_label, args.top_color), (args.bottom_label, args.bottom_color)):
            grp = tail_df[tail_df["group"] == group]
            pivot = grp.pivot(index="map", columns="threshold", values="tail_prob").reindex(columns=thresholds)
            for _, row in pivot.iterrows():
                ax1.plot(thresholds, row.to_numpy(), color=color, alpha=0.15, lw=1.0, zorder=1)
            mean = pivot.mean(axis=0).to_numpy()
            std = pivot.std(axis=0).to_numpy()
            ax1.plot(thresholds, mean, color=color, lw=2.8, marker="o", ms=5, label=group, zorder=3)
            ax1.fill_between(
                thresholds,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=color,
                alpha=0.22,
                zorder=2,
            )
        ax1.set_xlabel(f"t ({args.density_unit})")
        ax1.set_ylabel("P(density > t)")
        if args.right_log_y:
            ax1.set_yscale("log")
            ax1.set_ylim(args.right_log_y_min, 1.0)
        else:
            ax1.set_ylim(0.0, 1.0)
        ax1.set_title("")
        ax1.legend(frameon=False)
        style_legend_monospace(ax1, title="")

    save_figure(fig, args.output)

    top_iqr = metrics.loc[metrics["group"] == args.top_label, "iqr"].to_numpy()
    bot_iqr = metrics.loc[metrics["group"] == args.bottom_label, "iqr"].to_numpy()

    print(f"Saved: {args.output}")
    for t in thresholds:
        top_tail = tail_df.loc[
            (tail_df["group"] == args.top_label) & (tail_df["threshold"] == t), "tail_prob"
        ].to_numpy()
        bot_tail = tail_df.loc[
            (tail_df["group"] == args.bottom_label) & (tail_df["threshold"] == t), "tail_prob"
        ].to_numpy()
        print(
            f"Tail P(density>{t:g}) mean: "
            f"{args.top_label}={top_tail.mean():.4f}, {args.bottom_label}={bot_tail.mean():.4f}"
        )
    print(
        f"IQR mean (spread proxy): "
        f"{args.top_label}={top_iqr.mean():.4f}, {args.bottom_label}={bot_iqr.mean():.4f}"
    )


if __name__ == "__main__":
    main()
