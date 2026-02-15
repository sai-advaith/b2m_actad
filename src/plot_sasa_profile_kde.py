#!/usr/bin/env python3
import argparse
import itertools
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

try:
    import freesasa
except Exception:
    freesasa = None


def normalize_group_name(value: object) -> str:
    s = str(value).strip().replace(" ", "").upper()
    if s == "C121":
        return "C 121"
    if s == "I121":
        return "I 121"
    return str(value).strip()


def is_hydrogen(atom: gemmi.Atom) -> bool:
    return atom.element.name.upper() == "H"


def collect_atom_rows(model: gemmi.Model) -> list[dict[str, object]]:
    rows = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            resnum_i = int(residue.seqid.num)
            icode = residue.seqid.icode.strip()
            resnum = f"{resnum_i}{icode}" if icode else str(resnum_i)
            for atom in residue:
                if is_hydrogen(atom):
                    continue
                rows.append(
                    {
                        "atom_name": atom.name.strip(),
                        "res_name": residue.name,
                        "res_num": resnum,
                        "res_num_i": resnum_i,
                        "icode": icode,
                        "chain_name": chain.name,
                        "pos": atom.pos,
                        "occ": max(float(atom.occ), 0.0),
                        "altloc": atom.altloc.strip(),
                    }
                )
    return rows


def occupancy_weights_for_rows(rows: list[dict[str, object]]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        sk = (
            str(r["chain_name"]),
            int(r["res_num_i"]),
            str(r["icode"]),
            str(r["res_name"]),
            str(r["atom_name"]),
        )
        groups.setdefault(sk, []).append(i)

    w = np.ones(len(rows), dtype=float)
    for idxs in groups.values():
        if len(idxs) == 1:
            w[idxs[0]] = 1.0
            continue
        occ = np.array([float(rows[i]["occ"]) for i in idxs], dtype=float)
        s = float(np.sum(occ))
        if s > 0:
            occ = occ / s
        else:
            occ = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
        for i, wi in zip(idxs, occ):
            w[i] = float(wi)
    return w


def build_sites(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    w = occupancy_weights_for_rows(rows)
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        sk = (
            str(r["chain_name"]),
            int(r["res_num_i"]),
            str(r["icode"]),
            str(r["res_name"]),
            str(r["atom_name"]),
        )
        groups.setdefault(sk, []).append(i)

    sites = []
    for sk, idxs in groups.items():
        probs = np.array([float(w[i]) for i in idxs], dtype=float)
        ps = float(np.sum(probs))
        if ps > 0:
            probs = probs / ps
        else:
            probs = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
        sites.append({"site_key": sk, "idxs": idxs, "probs": probs})
    return sites


def iter_state_indices(
    sites: list[dict[str, object]],
    max_exact_states: int = 256,
    mc_samples: int = 64,
    random_seed: int = 0,
) -> itertools.chain:
    if not sites:
        return iter(())
    sizes = [len(s["idxs"]) for s in sites]
    n_states = 1
    exact = True
    for k in sizes:
        n_states *= max(1, int(k))
        if n_states > max_exact_states:
            exact = False
            break

    if exact:
        option_ranges = [range(k) for k in sizes]

        def exact_iter():
            for choice_tuple in itertools.product(*option_ranges):
                prob = 1.0
                idxs = []
                for s, j in zip(sites, choice_tuple):
                    p = float(s["probs"][j])
                    prob *= p
                    if prob <= 0:
                        break
                    idxs.append(int(s["idxs"][j]))
                if prob > 0 and idxs:
                    yield idxs, float(prob)

        return exact_iter()

    rng = np.random.default_rng(random_seed)
    n = int(max(1, mc_samples))

    def mc_iter():
        for _ in range(n):
            idxs = []
            for s in sites:
                probs = np.asarray(s["probs"], dtype=float)
                j = int(rng.choice(probs.size, p=probs))
                idxs.append(int(s["idxs"][j]))
            yield idxs, 1.0 / float(n)

    return mc_iter()


def freesasa_areas_for_rows(rows: list[dict[str, object]]) -> np.ndarray:
    s = freesasa.Structure()
    for r in rows:
        p = r["pos"]
        s.addAtom(
            str(r["atom_name"]),
            str(r["res_name"]),
            str(r["res_num"]),
            str(r["chain_name"]),
            float(p.x),
            float(p.y),
            float(p.z),
        )
    result = freesasa.calc(s)
    return np.array([float(result.atomArea(i)) for i in range(len(rows))], dtype=float)


def expected_sasa_profile_by_residue(
    structure: gemmi.Structure,
    max_exact_states: int = 256,
    mc_samples: int = 64,
    random_seed: int = 0,
) -> dict[int, float]:
    if freesasa is None:
        raise RuntimeError("freesasa is not installed.")
    if len(structure) == 0:
        return {}
    model = structure[0]
    rows = collect_atom_rows(model)
    if not rows:
        return {}

    sites = build_sites(rows)
    profile = defaultdict(float)
    weight_sum = 0.0

    for idxs, w_state in iter_state_indices(
        sites, max_exact_states=max_exact_states, mc_samples=mc_samples, random_seed=random_seed
    ):
        if w_state <= 0 or not idxs:
            continue
        state_rows = [rows[i] for i in idxs]
        areas = freesasa_areas_for_rows(state_rows)
        if areas.size != len(state_rows):
            continue
        by_res = defaultdict(float)
        for r, a in zip(state_rows, areas):
            by_res[int(r["res_num_i"])] += float(a)
        for resi, area in by_res.items():
            profile[int(resi)] += float(w_state) * float(area)
        weight_sum += float(w_state)

    if weight_sum <= 0:
        return {}
    # Normalize in case state weights do not sum exactly to 1 due to truncation/numerics.
    return {resi: float(v / weight_sum) for resi, v in profile.items()}


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
    values: np.ndarray,
    x_grid: np.ndarray,
    bw_adjust: float,
) -> np.ndarray:
    if residues.size == 0 or values.size == 0:
        return np.zeros_like(x_grid, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return np.zeros_like(x_grid, dtype=float)

    if gaussian_kde is None or residues.size < 2 or np.allclose(np.std(residues), 0.0):
        bw = max(1.0, bw_adjust * 2.0)
        y = np.zeros_like(x_grid, dtype=float)
        for r, c in zip(residues, values):
            y += float(c) * np.exp(-0.5 * ((x_grid - float(r)) / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))
        return y

    kde = gaussian_kde(
        residues.astype(float),
        weights=values.astype(float),
        bw_method=lambda obj: obj.scotts_factor() * float(bw_adjust),
    )
    return kde(x_grid) * total


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot altloc-aware KDE-smoothed per-residue SASA profiles.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/tables/b2m_group_metrics.csv"))
    parser.add_argument("--structures-dir", type=Path, default=Path("data/pdb_downloads"))
    parser.add_argument("--bw-adjust", type=float, default=0.18)
    parser.add_argument("--res-min", type=int, default=30)
    parser.add_argument("--res-max", type=int, default=70)
    parser.add_argument("--show-individual", action="store_true")
    parser.add_argument("--individual-alpha", type=float, default=0.12)
    parser.add_argument("--individual-lw", type=float, default=1.0)
    parser.add_argument("--highlight-start", type=float, default=55.0)
    parser.add_argument("--highlight-end", type=float, default=63.0)
    parser.add_argument("--highlight-color", default="#9acd8f")
    parser.add_argument("--highlight-alpha", type=float, default=0.20)
    parser.add_argument("--n-grid", type=int, default=800)
    parser.add_argument("--max-exact-states", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/figures/sasa_profile_kde.png"),
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("out/tables/sasa_profile.csv"),
    )
    args = parser.parse_args()

    if freesasa is None:
        raise SystemExit("freesasa is required. Install with: conda run -n ansurr python -m pip install freesasa")
    freesasa.setVerbosity(freesasa.nowarnings)

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
            seed = sum(ord(c) for c in pdb_id) + 17
            prof = expected_sasa_profile_by_residue(
                st,
                max_exact_states=int(args.max_exact_states),
                mc_samples=int(args.mc_samples),
                random_seed=int(seed),
            )
        except Exception as exc:
            skipped.append((pdb_id, str(exc)))
            continue
        if not prof:
            skipped.append((pdb_id, "no_profile"))
            continue
        for resi, exp_sasa in prof.items():
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "group": group,
                    "residue": int(resi),
                    "expected_sasa": float(exp_sasa),
                }
            )

    prof_df = pd.DataFrame(rows)
    if prof_df.empty:
        raise SystemExit("No SASA profiles generated.")
    args.profile_csv.parent.mkdir(parents=True, exist_ok=True)
    prof_df.to_csv(args.profile_csv, index=False)

    per_struct_curves: dict[str, list[np.ndarray]] = {"C 121": [], "I 121": []}
    x_grid = np.linspace(float(args.res_min), float(args.res_max), int(args.n_grid))

    for (group, pdb_id), sub in prof_df.groupby(["group", "pdb_id"]):
        residues = sub["residue"].to_numpy(dtype=float)
        values = sub["expected_sasa"].to_numpy(dtype=float)
        y = kde_curve_from_profile(residues, values, x_grid, bw_adjust=args.bw_adjust)
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

    if args.highlight_start is not None and args.highlight_end is not None:
        h0 = float(min(args.highlight_start, args.highlight_end))
        h1 = float(max(args.highlight_start, args.highlight_end))
        ax.axvspan(h0, h1, color=args.highlight_color, alpha=args.highlight_alpha, zorder=0)

    ax.set_xlim(float(args.res_min), float(args.res_max))
    ax.set_xlabel("Residue number")
    ax.set_ylabel("SASA profile")
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
