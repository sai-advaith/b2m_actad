#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from plot_common import save_figure, set_theme, setup_matplotlib_agg, style_legend_monospace

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None


def normalize_group_name(value: object) -> str:
    s = str(value).strip().replace(" ", "").upper()
    if s == "C121":
        return "C 121"
    if s == "I121":
        return "I 121"
    return str(value).strip()


def is_hydrogen(atom: gemmi.Atom) -> bool:
    return atom.element.name.upper() == "H"


def atom_site_key(chain: gemmi.Chain, residue: gemmi.Residue, atom: gemmi.Atom) -> tuple[str, int, str, str, str]:
    return (
        chain.name,
        int(residue.seqid.num),
        residue.seqid.icode.strip(),
        residue.name,
        atom.name.strip(),
    )


def atom_conf_key(chain: gemmi.Chain, residue: gemmi.Residue, atom: gemmi.Atom) -> tuple[str, int, str, str, str, str]:
    return (
        chain.name,
        int(residue.seqid.num),
        residue.seqid.icode.strip(),
        residue.name,
        atom.name.strip(),
        atom.altloc.strip(),
    )


def cra_site_key(cra: gemmi.CRA) -> tuple[str, int, str, str, str]:
    return (
        cra.chain.name,
        int(cra.residue.seqid.num),
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
        cra.atom.name.strip(),
    )


def cra_conf_key(cra: gemmi.CRA) -> tuple[str, int, str, str, str, str]:
    return (
        cra.chain.name,
        int(cra.residue.seqid.num),
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
        cra.atom.name.strip(),
        cra.atom.altloc.strip(),
    )


def residue_number(cra: gemmi.CRA) -> int:
    return int(cra.residue.seqid.num)


def occupancy_probabilities(model: gemmi.Model) -> dict[tuple[str, int, str, str, str, str], float]:
    site_to_conf_occ: dict[tuple[str, int, str, str, str], dict[tuple[str, int, str, str, str, str], float]] = {}
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            for atom in residue:
                if is_hydrogen(atom):
                    continue
                sk = atom_site_key(chain, residue, atom)
                ck = atom_conf_key(chain, residue, atom)
                site_to_conf_occ.setdefault(sk, {})
                site_to_conf_occ[sk][ck] = site_to_conf_occ[sk].get(ck, 0.0) + max(float(atom.occ), 0.0)

    conf_prob: dict[tuple[str, int, str, str, str, str], float] = {}
    for sk, conf_map in site_to_conf_occ.items():
        conf_keys = sorted(conf_map.keys())
        occ = np.array([float(conf_map[k]) for k in conf_keys], dtype=float)
        s = float(np.sum(occ))
        if s > 0:
            probs = occ / s
        else:
            probs = np.full(len(conf_keys), 1.0 / len(conf_keys), dtype=float)
        for ck, p in zip(conf_keys, probs):
            conf_prob[ck] = float(p)
    return conf_prob


def expected_contact_profile_by_residue(structure: gemmi.Structure, cutoff: float) -> dict[int, float]:
    if len(structure) == 0:
        return {}
    model = structure[0]
    cell = structure.cell
    if cell.a <= 0 or cell.b <= 0 or cell.c <= 0:
        return {}

    conf_prob = occupancy_probabilities(model)
    ns = gemmi.NeighborSearch(model, cell, cutoff + 1.0).populate(include_h=False)
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    contacts = cs.find_contacts(ns)

    profile = defaultdict(float)
    for c in contacts:
        p1 = c.partner1
        p2 = c.partner2
        if p1.residue.is_water() or p2.residue.is_water():
            continue
        if is_hydrogen(p1.atom) or is_hydrogen(p2.atom):
            continue

        ck1 = cra_conf_key(p1)
        ck2 = cra_conf_key(p2)
        sk1 = cra_site_key(p1)
        sk2 = cra_site_key(p2)
        q1 = float(conf_prob.get(ck1, 1.0))
        q2 = float(conf_prob.get(ck2, 1.0))
        if sk1 == sk2:
            if ck1 != ck2:
                continue
            q = q1
        else:
            q = q1 * q2
        if q <= 0:
            continue

        # Count contacts made by ASU/query atoms only (partner1 side).
        profile[residue_number(p1)] += float(q)
    return dict(profile)


def find_structure_file(pdb_id: str, structures_dir: Path) -> Path | None:
    pdb_id = pdb_id.lower()
    candidates = [
        structures_dir / f"{pdb_id}.cif",
        structures_dir / f"{pdb_id}.pdb",
        structures_dir / f"pdb{pdb_id}.ent",
        structures_dir / f"{pdb_id}.ent",
    ]
    for p in candidates:
        if p.exists():
            return p
    any_hits = sorted(structures_dir.glob(f"*{pdb_id}*"))
    for p in any_hits:
        if p.suffix.lower() in {".cif", ".pdb", ".ent"}:
            return p
    return None


def kde_curve_from_profile(
    residues: np.ndarray,
    counts: np.ndarray,
    x_grid: np.ndarray,
    bw_adjust: float,
) -> np.ndarray:
    if residues.size == 0 or counts.size == 0:
        return np.zeros_like(x_grid, dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros_like(x_grid, dtype=float)

    if gaussian_kde is None or residues.size < 2 or np.allclose(np.std(residues), 0.0):
        bw = max(1.0, bw_adjust * 2.0)
        y = np.zeros_like(x_grid, dtype=float)
        for r, c in zip(residues, counts):
            y += float(c) * np.exp(-0.5 * ((x_grid - float(r)) / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))
        return y

    kde = gaussian_kde(
        residues.astype(float),
        weights=counts.astype(float),
        bw_method=lambda obj: obj.scotts_factor() * float(bw_adjust),
    )
    return kde(x_grid) * total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot altloc-aware KDE-smoothed crystal-contact profiles (C121 vs I121)."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("data/tables/b2m_group_metrics.csv"))
    parser.add_argument("--structures-dir", type=Path, default=Path("data/pdb_downloads"))
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--bw-adjust", type=float, default=0.8)
    parser.add_argument("--res-min", type=int, default=None, help="Minimum residue number to display.")
    parser.add_argument("--res-max", type=int, default=None, help="Maximum residue number to display.")
    parser.add_argument("--show-individual", action="store_true", help="Overlay individual structure KDE curves.")
    parser.add_argument("--individual-alpha", type=float, default=0.30, help="Alpha for individual curves.")
    parser.add_argument("--individual-lw", type=float, default=1.2, help="Linewidth for individual curves.")
    parser.add_argument("--highlight-start", type=float, default=None, help="Start residue for highlighted window.")
    parser.add_argument("--highlight-end", type=float, default=None, help="End residue for highlighted window.")
    parser.add_argument("--highlight-color", default="#9acd8f", help="Color for highlighted residue window.")
    parser.add_argument("--highlight-alpha", type=float, default=0.20, help="Alpha for highlighted residue window.")
    parser.add_argument("--highlight-label", default="Guided", help="Text label for highlighted residue window.")
    parser.add_argument(
        "--show-highlight-label",
        action="store_true",
        help="Show a text label centered on the highlighted window.",
    )
    parser.add_argument("--n-grid", type=int, default=800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/figures/crystal_contact_profile_kde.png"),
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("out/tables/crystal_contact_profile.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"pdb_id", "space_group"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input CSV missing columns: {', '.join(missing)}")

    rows = []
    skipped = []
    for _, rec in df.iterrows():
        pdb_id = str(rec["pdb_id"]).strip().lower()
        group = normalize_group_name(rec["space_group"])
        if group not in {"C 121", "I 121"}:
            continue
        st_path = find_structure_file(pdb_id, args.structures_dir)
        if st_path is None:
            skipped.append((pdb_id, "missing_structure_file"))
            continue
        try:
            st = gemmi.read_structure(str(st_path))
            prof = expected_contact_profile_by_residue(st, cutoff=args.contact_cutoff)
        except Exception as exc:
            skipped.append((pdb_id, str(exc)))
            continue
        if not prof:
            skipped.append((pdb_id, "no_contacts_or_invalid_structure"))
            continue
        for resi, exp_contacts in prof.items():
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "group": group,
                    "residue": int(resi),
                    "expected_contacts": float(exp_contacts),
                }
            )

    prof_df = pd.DataFrame(rows)
    if prof_df.empty:
        raise SystemExit("No profiles generated.")
    args.profile_csv.parent.mkdir(parents=True, exist_ok=True)
    prof_df.to_csv(args.profile_csv, index=False)

    per_struct_curves: dict[str, list[np.ndarray]] = {"C 121": [], "I 121": []}
    all_res = prof_df["residue"].astype(int).to_numpy()
    xmin = float(all_res.min()) if args.res_min is None else float(args.res_min)
    xmax = float(all_res.max()) if args.res_max is None else float(args.res_max)
    if xmax <= xmin:
        raise SystemExit("--res-max must be greater than --res-min.")
    x_grid = np.linspace(xmin, xmax, int(args.n_grid))

    for (group, pdb_id), sub in prof_df.groupby(["group", "pdb_id"]):
        residues = sub["residue"].to_numpy(dtype=float)
        counts = sub["expected_contacts"].to_numpy(dtype=float)
        y = kde_curve_from_profile(residues, counts, x_grid, bw_adjust=args.bw_adjust)
        per_struct_curves[group].append(y)

    setup_matplotlib_agg()
    import matplotlib.pyplot as plt

    set_theme(style="whitegrid", context="poster")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5), dpi=220)
    palette = {"C 121": "#1f77b4", "I 121": "#d62728"}

    for group in ["C 121", "I 121"]:
        curves = per_struct_curves[group]
        if not curves:
            continue
        mat = np.vstack(curves)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        if args.show_individual:
            for y in curves:
                ax.plot(
                    x_grid,
                    y,
                    lw=args.individual_lw,
                    color=palette[group],
                    alpha=args.individual_alpha,
                    zorder=1,
                )
        ax.plot(x_grid, mean, lw=2.6, color=palette[group], label=f"{group} (n={len(curves)})")
        ax.fill_between(x_grid, np.clip(mean - std, 0.0, None), mean + std, color=palette[group], alpha=0.20)

    ax.set_xlim(xmin, xmax)
    if args.highlight_start is not None and args.highlight_end is not None:
        h0 = float(min(args.highlight_start, args.highlight_end))
        h1 = float(max(args.highlight_start, args.highlight_end))
        ax.axvspan(h0, h1, color=args.highlight_color, alpha=args.highlight_alpha, zorder=0)
        if args.show_highlight_label:
            y0, y1 = ax.get_ylim()
            ax.text(
                0.5 * (h0 + h1),
                y0 + 0.92 * (y1 - y0),
                args.highlight_label,
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
                color="#2f5d2f",
            )
    ax.set_xlabel("Residue number")
    ax.set_ylabel("Crystal contacts")
    ax.grid(axis="y", alpha=0.25)
    leg = ax.legend(frameon=True)
    if leg is not None:
        style_legend_monospace(ax, title="")
    save_figure(fig, args.output)
    plt.close(fig)

    print(f"Saved plot: {args.output}")
    print(f"Saved profile table: {args.profile_csv}")
    print(f"Processed structures: {prof_df['pdb_id'].nunique()}")
    print(f"C 121 structures: {prof_df.loc[prof_df['group'] == 'C 121', 'pdb_id'].nunique()}")
    print(f"I 121 structures: {prof_df.loc[prof_df['group'] == 'I 121', 'pdb_id'].nunique()}")
    if skipped:
        print(f"Skipped structures: {len(skipped)}")
        for pdb_id, reason in skipped:
            print(f"  {pdb_id}: {reason}")


if __name__ == "__main__":
    main()
