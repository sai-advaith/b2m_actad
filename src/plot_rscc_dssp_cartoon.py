#!/usr/bin/env python3
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon

from plot_common import save_figure, set_theme, setup_matplotlib_agg

SSDRAW_SRC = Path("/Users/svedula/postdoc/ista/msa_opt/SSDraw/src")
if str(SSDRAW_SRC) not in sys.path:
    sys.path.insert(0, str(SSDRAW_SRC))

from SSDraw.core import (  # noqa: E402
    SPACING,
    SS_breakdown,
    build_helix,
    build_loop,
    build_strand,
    run_dssp,
)


def normalize_ss_for_render(ss_seq: str) -> str:
    # Treat all helix-like DSSP states as helix for drawing/shading.
    trans = {"G": "H", "I": "H", "B": "E"}
    return "".join(trans.get(ch, ch) for ch in ss_seq)


def load_group_matrix(npy_path: Path) -> np.ndarray:
    data = np.load(npy_path, allow_pickle=True).item()
    arrays = [np.asarray(v, dtype=float).ravel() for v in data.values()]
    max_len = max(len(a) for a in arrays)
    mat = np.full((len(arrays), max_len), np.nan, dtype=float)
    for i, arr in enumerate(arrays):
        arr = arr[np.isfinite(arr)]
        mat[i, : len(arr)] = arr
    return mat


def mean_std_by_position(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def infer_pdb_id(pdb_path: Path) -> str:
    m = re.search(r"([0-9][a-zA-Z0-9]{3})", pdb_path.stem)
    return m.group(1).lower() if m else pdb_path.stem.lower()


def draw_ssdraw_cartoon(ax, ss_seq: str, start: int, end: int) -> None:
    ss_seq = normalize_ss_for_render(ss_seq)
    _, _, _, _, ss_order, ss_bounds = SS_breakdown(ss_seq)
    strand_coords, loop_coords, helix_coords1, helix_coords2 = [], [], [], []

    for i in range(len(ss_order)):
        prev_ss = ss_order[i - 1] if i != 0 else None
        next_ss = ss_order[i + 1] if i != len(ss_order) - 1 else None
        if ss_order[i] == "L":
            build_loop(
                ss_bounds[i],
                0,
                -(2.0 / SPACING),
                loop_coords,
                len(ss_seq),
                1,
                prev_ss,
                next_ss,
            )
        elif ss_order[i] == "H":
            build_helix(
                ss_bounds[i],
                0,
                -(2.0 / SPACING),
                helix_coords1,
                helix_coords2,
            )
        elif ss_order[i] == "E":
            build_strand(
                ss_bounds[i],
                0,
                -(2.0 / SPACING),
                strand_coords,
                next_ss,
            )

    colors = {
        "loop": "#DADADA",
        "strand": "#FFFFFF",
        "helix_dark": "#FFFFFF",
        "helix_light": "#FFFFFF",
    }

    # Shrink cartoon vertically by transforming coordinates (no clipping).
    y_center = 2.0
    y_scale = 0.28

    def add_poly(coords, facecolor):
        arr = np.asarray(coords, dtype=float).copy()
        arr[:, 1] = (arr[:, 1] - y_center) * y_scale + y_center
        ax.add_patch(
            Polygon(
                arr,
                closed=True,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.2,
            )
        )

    for coords in loop_coords:
        add_poly(coords, colors["loop"])
    for coords in strand_coords:
        add_poly(coords, colors["strand"])
    for coords in helix_coords2:
        add_poly(coords, colors["helix_light"])
    for coords in helix_coords1:
        add_poly(coords, colors["helix_dark"])

    x0 = (start - 1) / 6.0
    x1 = (end - 1) / 6.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(1.65, 2.35)
    ax.axis("off")


def add_ss_background_shading(ax, ss_seq: str, start: int, end: int) -> None:
    ss_seq = normalize_ss_for_render(ss_seq)
    _, _, _, _, ss_order, ss_bounds = SS_breakdown(ss_seq)
    region_colors = {
        "L": "#DADADA",  # loop
        "E": "#FFFFFF",  # strand
        "H": "#FFFFFF",  # helix
        "B": "#FFFFFF",  # break/gap
    }

    lo = start - 1
    hi = end - 1
    for kind, (a, b) in zip(ss_order, ss_bounds):
        if b < lo or a > hi:
            continue
        s = max(a, lo)
        t = min(b, hi)
        x0 = s / 6.0
        x1 = t / 6.0
        if x1 <= x0:
            continue
        ax.axvspan(
            x0,
            x1,
            facecolor=region_colors.get(kind, "#FFFFFF"),
            alpha=0.35,
            edgecolor="none",
            zorder=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-residue RSCC with DSSP cartoon track using SSDraw core rendering helpers."
    )
    parser.add_argument("--pdb", type=Path, required=True, help="Reference PDB file for DSSP")
    parser.add_argument("--chain", default="A", help="Chain ID for DSSP (default: A)")
    parser.add_argument(
        "--top-npy", type=Path, default=Path("data/rscc/top_rscc.npy"), help="Top-group RSCC .npy dictionary"
    )
    parser.add_argument(
        "--bottom-npy", type=Path, default=Path("data/rscc/bottom_rscc.npy"), help="Bottom-group RSCC .npy dictionary"
    )
    parser.add_argument("--start", type=int, default=1, help="Start residue position (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End residue position (1-based; default full length)")
    parser.add_argument("--tick-step", type=int, default=5, help="Residue tick spacing (default: 5)")
    parser.add_argument("--top-label", type=str, default="Top", help="Legend label for top curve.")
    parser.add_argument("--bottom-label", type=str, default="Bottom", help="Legend label for bottom curve.")
    parser.add_argument("--focus-start", type=int, default=55, help="Focus region start residue (default: 55)")
    parser.add_argument("--focus-end", type=int, default=63, help="Focus region end residue (default: 63)")
    parser.add_argument(
        "--focus-label",
        type=str,
        default="Density\nguided",
        help="Label text for focus region annotation.",
    )
    parser.add_argument(
        "--no-focus",
        action="store_true",
        help="Disable focus-region highlighting.",
    )
    parser.add_argument("--context", default="poster", choices=["paper", "notebook", "talk", "poster"])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/figures/rscc_dssp_cartoon.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    setup_matplotlib_agg(cache_dir=Path("cache/mplcache"))
    import matplotlib.pyplot as plt

    set_theme(style="whitegrid", context=args.context)

    top_mat = load_group_matrix(args.top_npy)
    bottom_mat = load_group_matrix(args.bottom_npy)

    pdb_id = infer_pdb_id(args.pdb)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mmCIF file.*")
        ss_seq, _ = run_dssp(str(args.pdb), pdb_id, args.chain, dssp_exe="mkdssp")

    n_rscc = min(top_mat.shape[1], bottom_mat.shape[1])
    n = min(n_rscc, len(ss_seq))
    if n <= 0:
        raise SystemExit("No overlapping residues between RSCC arrays and DSSP sequence.")

    start = max(1, args.start)
    end = n if args.end is None else min(args.end, n)
    if end < start:
        raise SystemExit(f"Invalid range: start={start}, end={end}")

    top_mean, top_std = mean_std_by_position(top_mat[:, :n])
    bot_mean, bot_std = mean_std_by_position(bottom_mat[:, :n])

    idx = np.arange(start - 1, end)
    x = idx / 6.0

    fig = plt.figure(figsize=(16, 5.8), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 5.0], hspace=0.025)
    ax_ss = fig.add_subplot(gs[0, 0])
    ax_rscc = fig.add_subplot(gs[1, 0], sharex=ax_ss)

    draw_ssdraw_cartoon(ax_ss, ss_seq[:n], start, end)
    add_ss_background_shading(ax_rscc, ss_seq[:n], start, end)

    focus_lo = max(start, min(args.focus_start, args.focus_end))
    focus_hi = min(end, max(args.focus_start, args.focus_end))
    has_focus = (not args.no_focus) and (focus_lo <= focus_hi)
    if has_focus:
        fx0 = (focus_lo - 1) / 6.0
        fx1 = (focus_hi - 1) / 6.0
        focus_color = "#BFE7C8"
        ax_ss.axvspan(fx0, fx1, facecolor=focus_color, alpha=0.35, edgecolor="none", zorder=-1)
        ax_rscc.axvspan(fx0, fx1, facecolor=focus_color, alpha=0.20, edgecolor="none", zorder=0.8)

    top_color = "#1f77b4"
    bot_color = "#d62728"
    top_fill = "#1f77b4"
    bot_fill = "#d62728"
    ax_rscc.plot(x, top_mean[idx], color=top_color, lw=2.8, label=args.top_label)
    ax_rscc.fill_between(
        x,
        top_mean[idx] - top_std[idx],
        top_mean[idx] + top_std[idx],
        color=top_fill,
        alpha=0.16,
        zorder=1,
    )
    ax_rscc.plot(x, bot_mean[idx], color=bot_color, lw=2.8, label=args.bottom_label)
    ax_rscc.fill_between(
        x,
        bot_mean[idx] - bot_std[idx],
        bot_mean[idx] + bot_std[idx],
        color=bot_fill,
        alpha=0.16,
        zorder=1,
    )
    if has_focus:
        focus_mask = (idx + 1 >= focus_lo) & (idx + 1 <= focus_hi)
        ax_rscc.plot(x[focus_mask], top_mean[idx][focus_mask], color=top_color, lw=3.6, zorder=3)
        ax_rscc.plot(x[focus_mask], bot_mean[idx][focus_mask], color=bot_color, lw=3.6, zorder=3)

    tick_res = np.arange(start, end + 1, max(1, args.tick_step))
    tick_pos = (tick_res - 1) / 6.0
    tick_labels = [f"{r}" for r in tick_res]
    ax_rscc.set_xticks(tick_pos)
    ax_rscc.set_xticklabels(tick_labels)

    ax_rscc.set_xlim((start - 1) / 6.0, (end - 1) / 6.0)
    ax_rscc.set_ylim(0.5, 1.0)
    ax_rscc.set_ylabel("RSCC")
    ax_rscc.set_xlabel("Residue Number")
    if has_focus:
        yb = 0.505
        yt = 0.518
        bracket_color = "#2E7D32"
        ax_rscc.plot([fx0, fx1], [yb, yb], color=bracket_color, lw=1.8, zorder=4)
        ax_rscc.plot([fx0, fx0], [yb, yt], color=bracket_color, lw=1.8, zorder=4)
        ax_rscc.plot([fx1, fx1], [yb, yt], color=bracket_color, lw=1.8, zorder=4)
        ax_rscc.text(
            (fx0 + fx1) / 2,
            yt + 0.004,
            args.focus_label,
            ha="center",
            va="bottom",
            color=bracket_color,
            fontsize=16,
        )
    ax_rscc.legend(frameon=False, loc="lower right", prop={"family": "monospace"})
    ax_rscc.grid(alpha=0.25)

    fig.subplots_adjust(top=0.94, bottom=0.16, left=0.08, right=0.98, hspace=0.06)
    save_figure(fig, args.output, tight_layout=False)

    print(f"Saved: {args.output}")
    print(f"DSSP length: {len(ss_seq)}; RSCC length used: {n}; plotted range: {start}-{end}")
    print(f"Top entries: {top_mat.shape[0]}, Bottom entries: {bottom_mat.shape[0]}")


if __name__ == "__main__":
    main()
