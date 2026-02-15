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
    return values[np.isfinite(values)]


def pdb_id_label(ccp4_path: Path) -> str:
    return ccp4_path.stem.split("_")[0]


def rescale_zero_one(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot overlaid density KDE or histogram curves from one or more CCP4 maps."
    )
    parser.add_argument("maps", type=Path, nargs="+", help="One or more CCP4 map files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/figures/density_distribution.png"),
        help="Output image path (default: out/figures/density_distribution.png)",
    )
    parser.add_argument(
        "--bw-adjust",
        type=float,
        default=0.4,
        help="KDE bandwidth adjustment (<1 sharper, >1 smoother; default: 0.4)",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["kde", "hist", "cdf", "group-cdf", "group-hist", "group-kde", "mean", "sum"],
        default="kde",
        help="Plot type: kde, hist, cdf, group-cdf, group-hist, group-kde, mean, or sum (default: kde).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=150,
        help="Histogram bins when --plot-kind hist (default: 150).",
    )
    parser.add_argument(
        "--nonzero",
        action="store_true",
        help="Keep only density values > 0 before plotting.",
    )
    parser.add_argument(
        "--xmax-percentile",
        type=float,
        default=None,
        help="Clip plotting range upper bound to this percentile (e.g. 99.5).",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Explicit x-axis lower bound.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Explicit x-axis upper bound.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Explicit y-axis upper bound (ignored for CDF where y is fixed to [0, 1]).",
    )
    parser.add_argument(
        "--y-scale",
        type=float,
        default=1.0,
        help="Scale current y-axis upper bound by this factor (>1 adds headroom; default: 1.0).",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Plot log1p(density) instead of raw density.",
    )
    parser.add_argument(
        "--rescale-zero-one",
        action="store_true",
        help="Rescale each map independently to [0, 1] before plotting.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Keep only density values <= this maximum after scaling/log transforms.",
    )
    parser.add_argument(
        "--drop-lowest-percentile",
        type=float,
        default=0.0,
        help="Drop the lowest N percentile of values per map before plotting (e.g. 10).",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=4,
        help="Number of legend columns (default: 4).",
    )
    parser.add_argument(
        "--ordered-colors",
        action="store_true",
        help="For CDF plots, color lines by descending CDF at a reference x.",
    )
    parser.add_argument(
        "--order-x",
        type=float,
        default=None,
        help="Reference x for CDF color ordering (default: x-max if set, else 1.0).",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="viridis",
        help="Seaborn palette name for ordered colors (default: viridis).",
    )
    parser.add_argument(
        "--top-pdbs",
        nargs="*",
        default=[],
        help="PDB IDs to color as top group (for CDF).",
    )
    parser.add_argument(
        "--bottom-pdbs",
        nargs="*",
        default=[],
        help="PDB IDs to color as bottom group (for CDF).",
    )
    parser.add_argument(
        "--top-group-label",
        type=str,
        default="Top",
        help="Legend label for the top group.",
    )
    parser.add_argument(
        "--bottom-group-label",
        type=str,
        default="Bottom",
        help="Legend label for the bottom group.",
    )
    parser.add_argument(
        "--top-color",
        type=str,
        default="#1f77b4",
        help="Color for top group (default: #1f77b4).",
    )
    parser.add_argument(
        "--bottom-color",
        type=str,
        default="#d62728",
        help="Color for bottom group (default: #d62728).",
    )
    args = parser.parse_args()

    setup_matplotlib_agg(cache_dir=Path("cache/mplcache"))
    import matplotlib.pyplot as plt

    frames = []
    stats = []
    for map_path in args.maps:
        label = pdb_id_label(map_path)
        values = load_density_values(map_path)
        if args.rescale_zero_one:
            values = rescale_zero_one(values)
        if args.log1p:
            values = np.log1p(values)
        if args.clip_max is not None:
            values = values[values <= args.clip_max]
        if args.nonzero:
            values = values[values > 0]
        if args.drop_lowest_percentile > 0:
            p = args.drop_lowest_percentile
            if 0 < p < 100 and len(values) > 0:
                lo = float(np.percentile(values, p))
                values = values[values > lo]
        if len(values) == 0:
            print(f"{map_path.name}: skipped (no values after filters)")
            continue
        frames.append(
            pd.DataFrame({"density": values, "map": label})
        )
        stats.append(
            (
                label,
                len(values),
                float(values.min()),
                float(values.max()),
                float(values.mean()),
            )
        )

    if not frames:
        raise SystemExit("No data available to plot after filtering.")

    df = pd.concat(frames, ignore_index=True)

    x_lower = args.x_min
    x_upper = args.x_max
    if args.xmax_percentile is not None and 0 < args.xmax_percentile <= 100:
        x_upper = float(np.percentile(df["density"], args.xmax_percentile))
        if x_lower is None:
            x_lower = 0.0

    set_theme(style="whitegrid")
    if args.plot_kind in {"mean", "sum"}:
        agg_key = "mean" if args.plot_kind == "mean" else "sum"
        agg_title = "Mean" if args.plot_kind == "mean" else "Sum"
        stats_df = pd.DataFrame(
            [
                {
                    "map": name,
                    "mean": mean,
                    "sum": mean * n,
                }
                for name, n, _, _, mean in stats
            ]
        ).sort_values(agg_key, ascending=False)
        fig_h = max(6, 0.28 * len(stats_df))
        fig, ax = plt.subplots(figsize=(10, fig_h), dpi=150)
        sns.barplot(
            data=stats_df,
            x=agg_key,
            y="map",
            orient="h",
            color="#4C72B0",
            ax=ax,
        )
        ax.set_title(f"{agg_title} Density by PDB ID (n={len(stats_df)} maps)")
        if args.plot_kind == "mean":
            ax.set_xlabel("Mean log1p(Density)" if args.log1p else "Mean Density")
        else:
            ax.set_xlabel("Sum log1p(Density)" if args.log1p else "Sum Density")
        ax.set_ylabel("PDB ID")
        if x_lower is not None or x_upper is not None:
            ax.set_xlim(left=x_lower, right=x_upper)
        if args.y_max is not None:
            ax.set_ylim(top=args.y_max)
        save_figure(fig, args.output)
        print(f"Saved: {args.output}")
        for name, n, vmin, vmax, mean in stats:
            print(f"{name}: n={n}, min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}")
        return

    if args.plot_kind in {"group-cdf", "group-hist", "group-kde"}:
        top_set = {p.lower() for p in args.top_pdbs}
        bottom_set = {p.lower() for p in args.bottom_pdbs}
        if not top_set or not bottom_set:
            raise SystemExit(f"{args.plot_kind} requires both --top-pdbs and --bottom-pdbs.")

        map_to_values = {
            name: df.loc[df["map"] == name, "density"].to_numpy()
            for name, _, _, _, _ in stats
        }

        top_maps = sorted([m for m in map_to_values if m in top_set])
        bottom_maps = sorted([m for m in map_to_values if m in bottom_set])
        if not top_maps or not bottom_maps:
            raise SystemExit("No maps found for top/bottom groups after filtering.")

        top_color = args.top_color
        bot_color = args.bottom_color
        top_lbl = args.top_group_label
        bot_lbl = args.bottom_group_label

        selected = np.concatenate([map_to_values[m] for m in top_maps + bottom_maps])
        x0 = x_lower if x_lower is not None else float(selected.min())
        x1 = x_upper if x_upper is not None else float(selected.max())

        if args.plot_kind in {"group-hist", "group-kde"}:
            top_vals = np.concatenate([map_to_values[m] for m in top_maps])
            bot_vals = np.concatenate([map_to_values[m] for m in bottom_maps])
            gdf = pd.DataFrame(
                {
                    "density": np.concatenate([top_vals, bot_vals]),
                    "group": [top_lbl] * len(top_vals) + [bot_lbl] * len(bot_vals),
                }
            )
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            if args.plot_kind == "group-hist":
                sns.histplot(
                    data=gdf,
                    x="density",
                    hue="group",
                    bins=args.bins,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    linewidth=1.8,
                    palette={top_lbl: top_color, bot_lbl: bot_color},
                    ax=ax,
                )
                ax.set_title("Group Density Distribution (Histogram)")
            else:
                sns.kdeplot(
                    data=gdf,
                    x="density",
                    hue="group",
                    common_norm=False,
                    fill=False,
                    bw_adjust=args.bw_adjust,
                    linewidth=2.0,
                    palette={top_lbl: top_color, bot_lbl: bot_color},
                    ax=ax,
                )
                ax.set_title("Group Density Distribution (KDE)")
            ax.set_xlabel("log1p(Density)" if args.log1p else "Density")
            ax.set_ylabel("Density (normalized)")
            ax.set_xlim(x0, x1)
            if args.y_max is not None:
                ax.set_ylim(top=args.y_max)
            elif args.y_scale > 1.0:
                _, y1 = ax.get_ylim()
                ax.set_ylim(0.0, y1 * args.y_scale)
            if ax.legend_ is not None:
                style_legend_monospace(ax, title="Group")
            save_figure(fig, args.output)

            print(f"Saved: {args.output}")
            print(f"Top maps ({len(top_maps)}): {', '.join(top_maps)}")
            print(f"Bottom maps ({len(bottom_maps)}): {', '.join(bottom_maps)}")
            for name, n, vmin, vmax, mean in stats:
                print(f"{name}: n={n}, min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}")
            return

        x_grid = np.linspace(x0, x1, 600)
        def ecdf_matrix(map_ids):
            mats = []
            for map_id in map_ids:
                vals = np.sort(map_to_values[map_id])
                cdf = np.searchsorted(vals, x_grid, side="right") / vals.size
                mats.append(cdf)
            return np.vstack(mats)

        top_mat = ecdf_matrix(top_maps)
        bot_mat = ecdf_matrix(bottom_maps)

        top_mean = top_mat.mean(axis=0)
        bot_mean = bot_mat.mean(axis=0)
        top_std = top_mat.std(axis=0)
        bot_std = bot_mat.std(axis=0)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(x_grid, top_mean, color=top_color, lw=2, label=f"{top_lbl} (n={len(top_maps)})")
        ax.fill_between(
            x_grid,
            np.clip(top_mean - top_std, 0, 1),
            np.clip(top_mean + top_std, 0, 1),
            color=top_color,
            alpha=0.2,
        )
        ax.plot(x_grid, bot_mean, color=bot_color, lw=2, label=f"{bot_lbl} (n={len(bottom_maps)})")
        ax.fill_between(
            x_grid,
            np.clip(bot_mean - bot_std, 0, 1),
            np.clip(bot_mean + bot_std, 0, 1),
            color=bot_color,
            alpha=0.2,
        )

        ax.set_title("Group-Averaged CDF (mean ± std)")
        ax.set_xlabel("log1p(Density)" if args.log1p else "Density")
        ax.set_ylabel("CDF")
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 1)
        ax.legend(title="Group", frameon=False)
        style_legend_monospace(ax)
        save_figure(fig, args.output)

        print(f"Saved: {args.output}")
        print(f"Top maps ({len(top_maps)}): {', '.join(top_maps)}")
        print(f"Bottom maps ({len(bottom_maps)}): {', '.join(bottom_maps)}")
        for name, n, vmin, vmax, mean in stats:
            print(f"{name}: n={n}, min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    custom_group_legend = False
    if args.plot_kind == "kde":
        sns.kdeplot(
            data=df,
            x="density",
            hue="map",
            common_norm=False,
            fill=False,
            bw_adjust=args.bw_adjust,
            linewidth=1.5,
            warn_singular=False,
            ax=ax,
        )
        ax.set_title(f"Density Distribution (KDE, n={len(stats)} maps)")
    elif args.plot_kind == "hist":
        sns.histplot(
            data=df,
            x="density",
            hue="map",
            bins=args.bins,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title(f"Density Distribution (Histogram, n={len(stats)} maps)")
    else:
        hue_order = None
        palette = None
        top_set = {p.lower() for p in args.top_pdbs}
        bottom_set = {p.lower() for p in args.bottom_pdbs}

        if top_set or bottom_set:
            from matplotlib.lines import Line2D

            def group_of(pdb_id: str) -> str:
                if pdb_id in top_set:
                    return "top"
                if pdb_id in bottom_set:
                    return "bottom"
                return "other"

            # Group-level colors while still drawing one ECDF per PDB map.
            group_colors = {"top": args.top_color, "bottom": args.bottom_color, "other": "#7f7f7f"}
            group_labels = {"top": args.top_group_label, "bottom": args.bottom_group_label, "other": "Other"}
            unique_maps = sorted(df["map"].unique())
            map_groups = {m: group_of(m) for m in unique_maps}
            for map_id in unique_maps:
                group = map_groups[map_id]
                sns.ecdfplot(
                    data=df[df["map"] == map_id],
                    x="density",
                    color=group_colors[group],
                    alpha=0.9,
                    linewidth=1.2,
                    ax=ax,
                    legend=False,
                )

            present_groups = []
            if any(g == "top" for g in map_groups.values()):
                present_groups.append("top")
            if any(g == "bottom" for g in map_groups.values()):
                present_groups.append("bottom")
            if any(g == "other" for g in map_groups.values()):
                present_groups.append("other")
            handles = [
                Line2D([0], [0], color=group_colors[g], lw=2, label=group_labels[g])
                for g in present_groups
            ]
            ax.legend(handles=handles, title="Group", frameon=False)
            style_legend_monospace(ax)
            custom_group_legend = True

            print(f"Top group ({len(top_set)}): {', '.join(sorted(top_set))}")
            print(f"Bottom group ({len(bottom_set)}): {', '.join(sorted(bottom_set))}")
            if present_groups and "other" in present_groups:
                others = sorted([m for m, g in map_groups.items() if g == "other"])
                print(f"Other group ({len(others)}): {', '.join(others)}")
        elif args.ordered_colors:
            x_ref = args.order_x if args.order_x is not None else (x_upper if x_upper is not None else 1.0)
            cdf_rank = (
                df.groupby("map")["density"]
                .apply(lambda s: float((s <= x_ref).mean()))
                .sort_values(ascending=False)
            )
            hue_order = cdf_rank.index.tolist()
            palette = sns.color_palette(args.palette, n_colors=len(hue_order))
            print(f"CDF color order at x={x_ref:g}:")
            for i, (name, val) in enumerate(cdf_rank.items(), start=1):
                print(f"{i:2d}. {name}: CDF={val:.6f}")

        if not custom_group_legend:
            sns.ecdfplot(
                data=df,
                x="density",
                hue="map",
                hue_order=hue_order,
                palette=palette,
                ax=ax,
            )
        ax.set_title(f"Density Distribution (CDF, n={len(stats)} maps)")
    ax.set_xlabel("log1p(Density)" if args.log1p else "Density")
    if args.plot_kind == "cdf":
        ax.set_ylabel("CDF")
    else:
        ax.set_ylabel("Density (normalized)")
    if x_lower is not None or x_upper is not None:
        ax.set_xlim(left=x_lower, right=x_upper)
    if args.plot_kind != "cdf":
        if args.y_max is not None:
            ax.set_ylim(top=args.y_max)
        elif args.y_scale > 1.0:
            _, y1 = ax.get_ylim()
            ax.set_ylim(0.0, y1 * args.y_scale)
    if ax.legend_ is not None:
        legend_title = "Group" if custom_group_legend else "PDB ID"
        sns.move_legend(
            ax,
            "upper center",
            bbox_to_anchor=(0.5, -0.18),
            title=legend_title,
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            ncol=max(1, args.legend_cols),
        )
        style_legend_monospace(ax)
    save_figure(fig, args.output, layout_rect=(0.0, 0.18, 1.0, 1.0))

    print(f"Saved: {args.output}")
    for name, n, vmin, vmax, mean in stats:
        print(f"{name}: n={n}, min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}")


if __name__ == "__main__":
    main()
