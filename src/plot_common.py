#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import matplotlib
import seaborn as sns


def setup_matplotlib_agg(cache_dir: Path | None = None) -> None:
    if cache_dir is not None:
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")


def set_theme(*, style: str = "whitegrid", context: str = "poster", rc: dict[str, Any] | None = None) -> None:
    sns.set_theme(style=style, context=context, rc=rc)


def sanitize_filename(name: str) -> str:
    s = name.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def style_xticks_monospace(ax) -> None:
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("monospace")


def style_legend_monospace(ax, title: str | None = None) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    if title is not None:
        leg.set_title(title)
    for txt in leg.get_texts():
        txt.set_fontfamily("monospace")
    title_obj = leg.get_title()
    if title_obj is not None:
        title_obj.set_fontfamily("monospace")


def dedupe_legend(ax, ordered_labels: list[str], title: str = "") -> None:
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(
        [uniq[l] for l in ordered_labels if l in uniq],
        [l for l in ordered_labels if l in uniq],
        title=title,
        frameon=True,
        loc="best",
    )


def add_box_strip(
    ax,
    *,
    data,
    x: str,
    y: str,
    order: list[str],
    palette: dict[str, str],
    hue: str | None = None,
    hue_order: list[str] | None = None,
    box_width: float = 0.5,
    box_linewidth: float = 2.0,
    box_fliersize: float = 0.0,
    box_dodge: bool = False,
    box_saturation: float = 1.0,
    box_legend: bool = False,
    strip_color: str = "black",
    strip_jitter: float = 0.10,
    strip_size: float = 7.0,
    strip_alpha: float = 0.85,
    strip_dodge: bool = False,
) -> None:
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        saturation=box_saturation,
        width=box_width,
        fliersize=box_fliersize,
        linewidth=box_linewidth,
        dodge=box_dodge,
        legend=box_legend,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        dodge=strip_dodge,
        color=strip_color,
        jitter=strip_jitter,
        size=strip_size,
        alpha=strip_alpha,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    style_xticks_monospace(ax)


def save_figure(fig, output: Path, *, tight_layout: bool = True, layout_rect: tuple[float, float, float, float] | None = None) -> None:
    if tight_layout:
        if layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=layout_rect)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
