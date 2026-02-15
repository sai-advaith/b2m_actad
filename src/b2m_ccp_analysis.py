#!/usr/bin/env python3
import argparse
import itertools
import math
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import gemmi
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

try:
    import freesasa
except Exception:
    freesasa = None


def normalize_group_name(name: str) -> str:
    raw = str(name).strip().replace(" ", "")
    if raw.upper() == "C121":
        return "C 121"
    if raw.upper() == "I121":
        return "I 121"
    return str(name).strip()


def download_structure(pdb_id: str, out_dir: Path, overwrite: bool = False) -> tuple[Path, str]:
    pdb_up = pdb_id.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_path = out_dir / f"{pdb_up}.cif"
    pdb_path = out_dir / f"{pdb_up}.pdb"

    if not overwrite and cif_path.exists():
        return cif_path, "cif"
    if not overwrite and pdb_path.exists():
        return pdb_path, "pdb"

    urls = [
        (f"https://files.rcsb.org/download/{pdb_up}.cif", cif_path, "cif"),
        (f"https://files.rcsb.org/download/{pdb_up}.pdb", pdb_path, "pdb"),
    ]
    last_err = None
    for url, path, fmt in urls:
        try:
            urllib.request.urlretrieve(url, path)
            return path, fmt
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Failed to download {pdb_up}: {last_err}")


def residue_id(cra: gemmi.CRA) -> tuple[str, int, str, str]:
    return (
        cra.chain.name,
        cra.residue.seqid.num,
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
    )


def atom_id(cra: gemmi.CRA) -> tuple[str, int, str, str, str, str]:
    return (
        cra.chain.name,
        cra.residue.seqid.num,
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
        cra.atom.name.strip(),
        cra.atom.altloc.strip(),
    )


def atom_site_id(cra: gemmi.CRA) -> tuple[str, int, str, str, str]:
    return (
        cra.chain.name,
        cra.residue.seqid.num,
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
        cra.atom.name.strip(),
    )


def atom_conf_id(cra: gemmi.CRA) -> tuple[str, int, str, str, str, str]:
    return (
        cra.chain.name,
        cra.residue.seqid.num,
        cra.residue.seqid.icode.strip(),
        cra.residue.name,
        cra.atom.name.strip(),
        cra.atom.altloc.strip(),
    )


def row_site_id(r: dict[str, object]) -> tuple[str, int, str, str, str]:
    return (
        str(r["chain_name"]),
        int(r["res_num_i"]),
        str(r["icode"]),
        str(r["res_name"]),
        str(r["atom_name"]),
    )


def row_conf_id(r: dict[str, object]) -> tuple[str, int, str, str, str, str]:
    return (
        str(r["chain_name"]),
        int(r["res_num_i"]),
        str(r["icode"]),
        str(r["res_name"]),
        str(r["atom_name"]),
        str(r["altloc"]).strip(),
    )


def is_hydrogen(atom: gemmi.Atom) -> bool:
    return atom.element.name.upper() == "H"


def residue_in_region(residue: gemmi.Residue, start: int, end: int) -> bool:
    n = int(residue.seqid.num)
    return start <= n <= end


def count_non_h_non_water_atoms(model: gemmi.Model) -> int:
    n = 0
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            for atom in residue:
                if is_hydrogen(atom):
                    continue
                n += 1
    return n


def model_mass_non_water_da(model: gemmi.Model) -> float:
    mass = 0.0
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            for atom in residue:
                mass += float(atom.element.weight)
    return mass


def count_non_h_non_water_atoms_in_region(model: gemmi.Model, start: int, end: int) -> int:
    n = 0
    for chain in model:
        for residue in chain:
            if residue.is_water() or not residue_in_region(residue, start, end):
                continue
            for atom in residue:
                if is_hydrogen(atom):
                    continue
                n += 1
    return n


def collect_asu_atoms(model: gemmi.Model) -> list[tuple[str, str, str, str, gemmi.Position]]:
    atoms = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            resnum = str(int(residue.seqid.num))
            icode = residue.seqid.icode.strip()
            if icode:
                resnum = f"{resnum}{icode}"
            for atom in residue:
                if is_hydrogen(atom):
                    continue
                atoms.append(
                    (
                        atom.name.strip(),
                        residue.name,
                        resnum,
                        chain.name,
                        atom.pos,
                    )
                )
    return atoms


def atom_altloc_is_set(atom: gemmi.Atom) -> bool:
    a = atom.altloc
    if a in {"", " ", ".", "?", "\x00"}:
        return False
    if a.strip() == "":
        return False
    return True


def collect_asu_atoms_detailed(
    model: gemmi.Model, region_start: int, region_end: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chain in model:
        for residue in chain:
            if residue.is_water():
                continue
            resnum_i = int(residue.seqid.num)
            resnum = str(resnum_i)
            icode = residue.seqid.icode.strip()
            if icode:
                resnum = f"{resnum}{icode}"
            in_region = region_start <= resnum_i <= region_end
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
                        "in_region": bool(in_region),
                        "b_iso": float(atom.b_iso),
                        "occ": float(atom.occ),
                        "altloc": atom.altloc,
                        "altloc_set": bool(atom_altloc_is_set(atom)),
                        "residue_key": (chain.name, resnum_i, icode, residue.name),
                    }
                )
    return rows


def chain_label_generator() -> Iterable[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for c in alphabet:
        yield c
    for i in itertools.count(1):
        for c in alphabet:
            yield f"{c}{i}"


def freesasa_total_area(atom_rows: list[tuple[str, str, str, str, gemmi.Position]]) -> float:
    if freesasa is None:
        return math.nan
    s = freesasa.Structure()
    for atom_name, res_name, res_num, chain_name, pos in atom_rows:
        s.addAtom(atom_name, res_name, res_num, chain_name, float(pos.x), float(pos.y), float(pos.z))
    return float(freesasa.calc(s).totalArea())


def freesasa_atom_areas_from_rows(rows: list[dict[str, object]]) -> tuple[np.ndarray, float]:
    if freesasa is None:
        return np.array([], dtype=float), math.nan
    s = freesasa.Structure()
    for r in rows:
        pos = r["pos"]
        s.addAtom(
            str(r["atom_name"]),
            str(r["res_name"]),
            str(r["res_num"]),
            str(r["chain_name"]),
            float(pos.x),
            float(pos.y),
            float(pos.z),
        )
    result = freesasa.calc(s)
    areas = np.array([float(result.atomArea(i)) for i in range(len(rows))], dtype=float)
    return areas, float(result.totalArea())


def occupancy_weights_for_rows(rows: list[dict[str, object]]) -> np.ndarray:
    """
    Compute per-atom occupancy weights for altloc handling in SASA/BSA:
    - For atom sites with altloc alternatives, normalize occupancies to sum to 1 within that site.
    - For sites without altloc alternatives, use weight 1.
    """
    if not rows:
        return np.array([], dtype=float)
    w = np.ones(len(rows), dtype=float)
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (
            str(r["chain_name"]),
            int(r["res_num_i"]),
            str(r["icode"]),
            str(r["res_name"]),
            str(r["atom_name"]),
        )
        groups.setdefault(key, []).append(i)

    for idxs in groups.values():
        if len(idxs) <= 1:
            w[idxs[0]] = 1.0
            continue
        has_alt = any(bool(rows[i].get("altloc_set", False)) for i in idxs)
        if not has_alt:
            # Rare duplicate atoms without altloc labels: split evenly.
            ww = 1.0 / len(idxs)
            for i in idxs:
                w[i] = ww
            continue
        occ = np.array([max(float(rows[i].get("occ", 0.0)), 0.0) for i in idxs], dtype=float)
        s = float(np.sum(occ))
        if s > 0:
            occ = occ / s
        else:
            occ = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
        for i, wi in zip(idxs, occ):
            w[i] = float(wi)
    return w


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.size == 0:
        return math.nan
    s = float(np.sum(weights))
    if s <= 0:
        return math.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cw, q, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def weighted_mean_and_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0 or weights.size == 0 or values.size != weights.size:
        return math.nan, math.nan
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return math.nan, math.nan
    v = values[mask]
    w = weights[mask]
    ws = float(np.sum(w))
    if ws <= 0:
        return math.nan, math.nan
    mu = float(np.sum(v * w) / ws)
    var = float(np.sum(w * (v - mu) ** 2) / ws)
    return mu, float(math.sqrt(max(0.0, var)))


def occupancy_weighted_local_density_around_region(
    asu_rows: list[dict[str, object]],
    radius: float = 6.0,
    max_exact_states: int = 20000,
    mc_samples: int = 8000,
    random_seed: int = 0,
) -> float:
    """
    Estimate local atom density around the region centroid with altloc-aware occupancy handling.
    For region atom sites with alternate conformers, evaluate conformer-specific centroids and
    average the resulting density by occupancy probabilities.
    """
    if not asu_rows:
        return math.nan

    # Group atom rows by atom site (same atom across altloc conformers).
    site_to_idxs: dict[tuple, list[int]] = {}
    for i, r in enumerate(asu_rows):
        key = (
            str(r["chain_name"]),
            int(r["res_num_i"]),
            str(r["icode"]),
            str(r["res_name"]),
            str(r["atom_name"]),
        )
        site_to_idxs.setdefault(key, []).append(i)

    if not site_to_idxs:
        return math.nan

    row_weights = occupancy_weights_for_rows(asu_rows)
    sites: list[dict[str, object]] = []
    for idxs in site_to_idxs.values():
        pos = np.array(
            [[float(asu_rows[i]["pos"].x), float(asu_rows[i]["pos"].y), float(asu_rows[i]["pos"].z)] for i in idxs],
            dtype=float,
        )
        probs = np.array([float(row_weights[i]) for i in idxs], dtype=float)
        psum = float(np.sum(probs))
        if psum > 0:
            probs = probs / psum
        else:
            probs = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
        sites.append(
            {
                "in_region": bool(asu_rows[idxs[0]]["in_region"]),
                "pos": pos,
                "probs": probs,
            }
        )

    region_site_idxs = [i for i, s in enumerate(sites) if bool(s["in_region"])]
    if not region_site_idxs:
        return math.nan

    vol_a3 = (4.0 / 3.0) * math.pi * (radius**3)

    def density_for_region_choice(choice_map: dict[int, int], ctr: np.ndarray) -> float:
        expected_count = 0.0
        for si, s in enumerate(sites):
            pos = s["pos"]
            if si in choice_map:
                p = pos[int(choice_map[si])]
                if float(np.linalg.norm(p - ctr)) <= radius:
                    expected_count += 1.0
                continue
            if pos.shape[0] == 1:
                if float(np.linalg.norm(pos[0] - ctr)) <= radius:
                    expected_count += 1.0
            else:
                d = np.linalg.norm(pos - ctr[None, :], axis=1)
                expected_count += float(np.sum(s["probs"] * (d <= radius)))
        return (expected_count / vol_a3) * 1000.0  # atoms per nm^3

    state_sizes = [int(sites[si]["pos"].shape[0]) for si in region_site_idxs]
    n_states = 1
    for k in state_sizes:
        n_states *= max(1, int(k))

    if n_states <= max_exact_states:
        weighted_density = 0.0
        weight_sum = 0.0
        option_ranges = [range(k) for k in state_sizes]
        for choice_tuple in itertools.product(*option_ranges):
            prob = 1.0
            region_pos = []
            choice_map: dict[int, int] = {}
            for si, opt_idx in zip(region_site_idxs, choice_tuple):
                p = float(sites[si]["probs"][opt_idx])
                prob *= p
                if prob <= 0.0:
                    break
                choice_map[si] = int(opt_idx)
                region_pos.append(sites[si]["pos"][opt_idx])
            if prob <= 0.0 or not region_pos:
                continue
            ctr = np.mean(np.array(region_pos, dtype=float), axis=0)
            dens = density_for_region_choice(choice_map, ctr)
            weighted_density += prob * dens
            weight_sum += prob
        if weight_sum > 0:
            return float(weighted_density / weight_sum)
        return math.nan

    # Fallback for many region altloc combinations.
    rng = np.random.default_rng(random_seed)
    dens_vals = []
    for _ in range(int(mc_samples)):
        choice_map: dict[int, int] = {}
        region_pos = []
        for si in region_site_idxs:
            probs = np.asarray(sites[si]["probs"], dtype=float)
            opt_idx = int(rng.choice(probs.size, p=probs))
            choice_map[si] = opt_idx
            region_pos.append(sites[si]["pos"][opt_idx])
        if not region_pos:
            continue
        ctr = np.mean(np.array(region_pos, dtype=float), axis=0)
        dens_vals.append(density_for_region_choice(choice_map, ctr))
    if dens_vals:
        return float(np.mean(dens_vals))
    return math.nan


def build_altloc_sites_for_sampling(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    site_to_idxs: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (
            str(r["chain_name"]),
            int(r["res_num_i"]),
            str(r["icode"]),
            str(r["res_name"]),
            str(r["atom_name"]),
        )
        site_to_idxs.setdefault(key, []).append(i)
    row_weights = occupancy_weights_for_rows(rows)
    sites: list[dict[str, object]] = []
    for idxs in site_to_idxs.values():
        probs = np.array([float(row_weights[i]) for i in idxs], dtype=float)
        s = float(np.sum(probs))
        if s > 0:
            probs = probs / s
        else:
            probs = np.full(len(idxs), 1.0 / len(idxs), dtype=float)
        sites.append(
            {
                "rows": [rows[i] for i in idxs],
                "probs": probs,
            }
        )
    return sites


def iter_conformer_state_rows(
    sites: list[dict[str, object]],
    max_exact_states: int = 256,
    mc_samples: int = 64,
    random_seed: int = 0,
) -> Iterable[tuple[list[dict[str, object]], float]]:
    if not sites:
        return
    state_sizes = [int(len(s["rows"])) for s in sites]
    n_states = 1
    exact = True
    for k in state_sizes:
        n_states *= max(1, int(k))
        if n_states > max_exact_states:
            exact = False
            break

    if exact:
        option_ranges = [range(k) for k in state_sizes]
        for choice_tuple in itertools.product(*option_ranges):
            prob = 1.0
            chosen_rows: list[dict[str, object]] = []
            for si, opt_idx in enumerate(choice_tuple):
                p = float(sites[si]["probs"][opt_idx])
                prob *= p
                if prob <= 0.0:
                    break
                chosen_rows.append(sites[si]["rows"][opt_idx])
            if prob <= 0.0 or not chosen_rows:
                continue
            yield chosen_rows, float(prob)
        return

    rng = np.random.default_rng(random_seed)
    for _ in range(int(max(1, mc_samples))):
        chosen_rows: list[dict[str, object]] = []
        for s in sites:
            probs = np.asarray(s["probs"], dtype=float)
            opt_idx = int(rng.choice(probs.size, p=probs))
            chosen_rows.append(s["rows"][opt_idx])
        if chosen_rows:
            yield chosen_rows, 1.0 / float(max(1, mc_samples))


def iter_site_assignments(
    site_options: list[dict[str, object]],
    max_exact_states: int = 20000,
    mc_samples: int = 2000,
    random_seed: int = 0,
) -> Iterable[tuple[dict[tuple, tuple], float]]:
    """
    Iterate occupancy-weighted altloc assignments for atom sites.
    Yields (site->selected_conformer, assignment_weight).
    """
    if not site_options:
        return

    fixed: dict[tuple, tuple] = {}
    variable: list[tuple[tuple, list[tuple], np.ndarray]] = []
    for s in site_options:
        site_key = s["site_key"]
        conf_keys = list(s["conf_keys"])
        probs = np.asarray(s["probs"], dtype=float)
        if len(conf_keys) <= 1:
            if conf_keys:
                fixed[site_key] = conf_keys[0]
            continue
        ps = float(np.sum(probs))
        if ps > 0:
            probs = probs / ps
        else:
            probs = np.full(len(conf_keys), 1.0 / len(conf_keys), dtype=float)
        variable.append((site_key, conf_keys, probs))

    if not variable:
        yield dict(fixed), 1.0
        return

    n_states = 1
    exact = True
    for _site_key, conf_keys, _probs in variable:
        n_states *= len(conf_keys)
        if n_states > max_exact_states:
            exact = False
            break

    if exact:
        option_ranges = [range(len(conf_keys)) for _site_key, conf_keys, _probs in variable]
        for choice_tuple in itertools.product(*option_ranges):
            assign = dict(fixed)
            prob = 1.0
            for (site_key, conf_keys, probs), idx in zip(variable, choice_tuple):
                assign[site_key] = conf_keys[idx]
                prob *= float(probs[idx])
                if prob <= 0.0:
                    break
            if prob > 0.0:
                yield assign, float(prob)
        return

    rng = np.random.default_rng(random_seed)
    n = int(max(1, mc_samples))
    for _ in range(n):
        assign = dict(fixed)
        for site_key, conf_keys, probs in variable:
            idx = int(rng.choice(len(conf_keys), p=probs))
            assign[site_key] = conf_keys[idx]
        yield assign, 1.0 / float(n)


def estimate_bsa_top_images_altloc_state_aware(
    structure: gemmi.Structure,
    asu_rows: list[dict[str, object]],
    image_counts: Counter,
    max_images: int,
    ns_ref: gemmi.NeighborSearch,
    max_exact_states: int = 256,
    mc_samples: int = 64,
    random_seed: int = 0,
) -> dict[str, float]:
    out = {
        "interface_images_used": 0,
        "interface_bsa_top_images_sum": math.nan,
        "interface_bsa_top_images_mean": math.nan,
        "interface_bsa_top_image_max": math.nan,
        "interface_region_images_used": 0,
        "interface_region_bsa_top_images_sum": math.nan,
        "interface_region_bsa_top_images_mean": math.nan,
        "interface_region_bsa_top_image_max": math.nan,
    }
    if freesasa is None or not image_counts or not asu_rows:
        return out

    sites = build_altloc_sites_for_sampling(asu_rows)
    if not sites:
        return out

    top_images = [img for img, _ in image_counts.most_common(max_images)]
    if not top_images:
        return out

    cell = structure.cell
    chain_ids = chain_label_generator()
    next(chain_ids)  # reserve 'A' for ASU
    tr_data = []
    for image_idx in top_images:
        tr_inv = ns_ref.get_image_transformation(int(image_idx)).inverse()
        tr_data.append((tr_inv, next(chain_ids)))

    acc_global = np.zeros(len(top_images), dtype=float)
    w_global = np.zeros(len(top_images), dtype=float)
    acc_region = np.zeros(len(top_images), dtype=float)
    w_region = np.zeros(len(top_images), dtype=float)

    for state_rows, w_state in iter_conformer_state_rows(
        sites, max_exact_states=max_exact_states, mc_samples=mc_samples, random_seed=random_seed
    ):
        if not state_rows or w_state <= 0.0:
            continue
        areas_asu, _ = freesasa_atom_areas_from_rows(state_rows)
        if areas_asu.size != len(state_rows):
            continue
        asu_sasa = float(np.sum(areas_asu))
        region_mask = np.array([bool(r["in_region"]) for r in state_rows], dtype=bool)
        has_region = bool(np.any(region_mask))
        asu_region_area = float(np.sum(areas_asu[region_mask])) if has_region else math.nan

        for j, (tr_inv, mate_chain) in enumerate(tr_data):
            complex_rows: list[dict[str, object]] = []
            for r in state_rows:
                rr = dict(r)
                rr["chain_name"] = "A"
                complex_rows.append(rr)
            for r in state_rows:
                p = r["pos"]
                f = cell.fractionalize(p)
                v = tr_inv.apply(f)
                fi = gemmi.Fractional(v.x, v.y, v.z)
                p_img = cell.orthogonalize(fi)
                rr = dict(r)
                rr["chain_name"] = mate_chain
                rr["pos"] = p_img
                rr["in_region"] = False
                complex_rows.append(rr)

            areas_complex, _ = freesasa_atom_areas_from_rows(complex_rows)
            if areas_complex.size != len(complex_rows):
                continue
            complex_sasa = float(np.sum(areas_complex))
            if np.isfinite(complex_sasa):
                bsa = ((2.0 * asu_sasa) - complex_sasa) / 2.0
                acc_global[j] += float(w_state) * float(max(0.0, bsa))
                w_global[j] += float(w_state)

            if has_region:
                complex_region_area = float(np.sum(areas_complex[: len(state_rows)][region_mask]))
                bsa_r = asu_region_area - complex_region_area
                acc_region[j] += float(w_state) * float(max(0.0, bsa_r))
                w_region[j] += float(w_state)

    global_vals = [float(acc_global[i] / w_global[i]) for i in range(len(top_images)) if w_global[i] > 0]
    region_vals = [float(acc_region[i] / w_region[i]) for i in range(len(top_images)) if w_region[i] > 0]

    if global_vals:
        out["interface_images_used"] = int(len(global_vals))
        out["interface_bsa_top_images_sum"] = float(np.sum(global_vals))
        out["interface_bsa_top_images_mean"] = float(np.mean(global_vals))
        out["interface_bsa_top_image_max"] = float(np.max(global_vals))
    if region_vals:
        out["interface_region_images_used"] = int(len(region_vals))
        out["interface_region_bsa_top_images_sum"] = float(np.sum(region_vals))
        out["interface_region_bsa_top_images_mean"] = float(np.mean(region_vals))
        out["interface_region_bsa_top_image_max"] = float(np.max(region_vals))
    return out


def estimate_interface_bsa_top_images(
    structure: gemmi.Structure,
    asu_rows: list[dict[str, object]],
    image_counts: Counter,
    max_images: int,
    ns_ref: gemmi.NeighborSearch,
) -> dict[str, float]:
    full = estimate_bsa_top_images_altloc_state_aware(
        structure=structure,
        asu_rows=asu_rows,
        image_counts=image_counts,
        max_images=max_images,
        ns_ref=ns_ref,
    )
    return {
        "interface_images_used": full["interface_images_used"],
        "interface_bsa_top_images_sum": full["interface_bsa_top_images_sum"],
        "interface_bsa_top_images_mean": full["interface_bsa_top_images_mean"],
        "interface_bsa_top_image_max": full["interface_bsa_top_image_max"],
    }


def estimate_region_interface_bsa_top_images(
    structure: gemmi.Structure,
    asu_rows: list[dict[str, object]],
    image_counts: Counter,
    max_images: int,
    ns_ref: gemmi.NeighborSearch,
) -> dict[str, float]:
    full = estimate_bsa_top_images_altloc_state_aware(
        structure=structure,
        asu_rows=asu_rows,
        image_counts=image_counts,
        max_images=max_images,
        ns_ref=ns_ref,
    )
    return {
        "interface_region_images_used": full["interface_region_images_used"],
        "interface_region_bsa_top_images_sum": full["interface_region_bsa_top_images_sum"],
        "interface_region_bsa_top_images_mean": full["interface_region_bsa_top_images_mean"],
        "interface_region_bsa_top_image_max": full["interface_region_bsa_top_image_max"],
    }


def local_region_metrics(
    structure: gemmi.Structure,
    model: gemmi.Model,
    asu_rows: list[dict[str, object]],
    region_start: int,
    region_end: int,
    ns_ref: gemmi.NeighborSearch,
    nearest_cutoff: float,
) -> dict[str, float]:
    out = {
        "region_atom_count": 0,
        "region_sasa_sum": math.nan,
        "region_sasa_mean_atom": math.nan,
        "region_sasa_median_atom": math.nan,
        "region_buried_atom_fraction_sasa_lt_5": math.nan,
        "region_buried_atom_fraction_sasa_lt_10": math.nan,
        "region_neighbor_contacts_4A": math.nan,
        "region_neighbor_unique_partner_atoms_4A": math.nan,
        "region_neighbor_contacts_6A": math.nan,
        "region_neighbor_unique_partner_atoms_6A": math.nan,
        "region_centroid_local_atom_density_6A_per_nm3": math.nan,
        "region_nearest_symmetry_atom_distance": math.nan,
        "region_median_symmetry_atom_distance": math.nan,
        "region_b_iso_mean": math.nan,
        "global_b_iso_mean": math.nan,
        "region_b_iso_ratio_to_global": math.nan,
        "region_b_iso_zscore_vs_global": math.nan,
        "region_occ_mean": math.nan,
        "region_occ_lt1_fraction": math.nan,
        "region_altloc_atom_fraction": math.nan,
        "region_altloc_residue_fraction": math.nan,
        # Placeholders for map-dependent compactness metrics (requires map files).
        "region_map_fraction_above_threshold": math.nan,
        "region_map_gradient_sharpness": math.nan,
        "region_map_local_variance": math.nan,
    }
    if not asu_rows:
        return out

    region_rows = [r for r in asu_rows if bool(r["in_region"])]
    region_mask = np.array([bool(r["in_region"]) for r in asu_rows], dtype=bool)
    w_asu = occupancy_weights_for_rows(asu_rows)
    w_region = w_asu[region_mask]
    out["region_atom_count"] = int(len(region_rows))
    if not region_rows:
        return out

    if freesasa is not None:
        areas_asu, _ = freesasa_atom_areas_from_rows(asu_rows)
        if areas_asu.size == len(asu_rows):
            region_areas = areas_asu[region_mask]
            wsum = float(np.sum(w_region))
            out["region_sasa_sum"] = float(np.sum(region_areas * w_region))
            if wsum > 0:
                out["region_sasa_mean_atom"] = float(np.sum(region_areas * w_region) / wsum)
                out["region_sasa_median_atom"] = weighted_quantile(region_areas, w_region, 0.5)
                out["region_buried_atom_fraction_sasa_lt_5"] = float(np.sum(w_region * (region_areas < 5.0)) / wsum)
                out["region_buried_atom_fraction_sasa_lt_10"] = float(np.sum(w_region * (region_areas < 10.0)) / wsum)

    # Neighbor-contact counts around region, including crystal mates, excluding same-residue contacts.
    # Altloc-aware: evaluate mutually exclusive conformer assignments and occupancy-weight average.
    site_to_conf_probs: dict[tuple, dict[tuple, float]] = {}
    for i, r in enumerate(asu_rows):
        sk = row_site_id(r)
        ck = row_conf_id(r)
        site_to_conf_probs.setdefault(sk, {})
        site_to_conf_probs[sk][ck] = site_to_conf_probs[sk].get(ck, 0.0) + float(w_asu[i])
    for sk, m in site_to_conf_probs.items():
        s = float(sum(m.values()))
        if s > 0:
            for ck in list(m.keys()):
                m[ck] = float(m[ck] / s)
        elif len(m) > 0:
            eq = 1.0 / float(len(m))
            for ck in list(m.keys()):
                m[ck] = eq

    cell = structure.cell
    for rad in (4.0, 6.0):
        ns_r = gemmi.NeighborSearch(model, cell, rad + 1.0).populate(include_h=False)
        cs_r = gemmi.ContactSearch(rad)
        cs_r.ignore = gemmi.ContactSearch.Ignore.SameResidue
        contacts_r = cs_r.find_contacts(ns_r)

        records = []
        sites_in_records = set()
        fallback_conf_by_site: dict[tuple, tuple] = {}
        for c in contacts_r:
            p1 = c.partner1
            p2 = c.partner2
            if p1.residue.is_water() or p2.residue.is_water():
                continue
            if is_hydrogen(p1.atom) or is_hydrogen(p2.atom):
                continue
            in_r1 = residue_in_region(p1.residue, region_start, region_end)
            in_r2 = residue_in_region(p2.residue, region_start, region_end)
            if not (in_r1 or in_r2):
                continue
            conf1 = atom_conf_id(p1)
            conf2 = atom_conf_id(p2)
            site1 = atom_site_id(p1)
            site2 = atom_site_id(p2)
            sites_in_records.add(site1)
            sites_in_records.add(site2)
            fallback_conf_by_site.setdefault(site1, conf1)
            fallback_conf_by_site.setdefault(site2, conf2)
            a1 = atom_id(p1)
            a2 = atom_id(p2)
            pair = tuple(sorted((a1, a2)))
            partner_key = None
            if in_r1 and not in_r2:
                partner_key = a2
            elif in_r2 and not in_r1:
                partner_key = a1
            records.append(
                {
                    "image_idx": int(c.image_idx),
                    "dist": float(c.dist),
                    "site1": site1,
                    "site2": site2,
                    "conf1": conf1,
                    "conf2": conf2,
                    "pair_key": pair,
                    "partner_key": partner_key,
                }
            )

        if not records:
            out[f"region_neighbor_contacts_{int(rad)}A"] = 0.0
            out[f"region_neighbor_unique_partner_atoms_{int(rad)}A"] = 0.0
            continue

        site_options = []
        for sk in sorted(sites_in_records):
            m = site_to_conf_probs.get(sk, {})
            if m:
                conf_keys = sorted(m.keys())
                probs = np.array([float(m[k]) for k in conf_keys], dtype=float)
            else:
                # Fallback for rare mismatch between row and contact identities.
                conf_keys = [fallback_conf_by_site[sk]]
                probs = np.array([1.0], dtype=float)
            site_options.append({"site_key": sk, "conf_keys": conf_keys, "probs": probs})

        acc_pair = 0.0
        acc_partner = 0.0
        w_pair = 0.0
        acc_nearest = 0.0
        w_nearest = 0.0
        acc_med = 0.0
        w_med = 0.0
        for assign, w_state in iter_site_assignments(
            site_options, max_exact_states=20000, mc_samples=2000, random_seed=0 + int(rad * 10)
        ):
            if w_state <= 0:
                continue
            pair_keys = set()
            partner_atoms = set()
            region_sym_dists = []
            for rec in records:
                if assign.get(rec["site1"]) != rec["conf1"]:
                    continue
                if assign.get(rec["site2"]) != rec["conf2"]:
                    continue
                pair_keys.add((rec["image_idx"], rec["pair_key"]))
                if rec["partner_key"] is not None:
                    partner_atoms.add((rec["image_idx"], rec["partner_key"]))
                if rec["image_idx"] != 0:
                    region_sym_dists.append(rec["dist"])

            acc_pair += float(w_state) * float(len(pair_keys))
            acc_partner += float(w_state) * float(len(partner_atoms))
            w_pair += float(w_state)
            if region_sym_dists:
                acc_nearest += float(w_state) * float(np.min(region_sym_dists))
                w_nearest += float(w_state)
                acc_med += float(w_state) * float(np.median(region_sym_dists))
                w_med += float(w_state)

        out[f"region_neighbor_contacts_{int(rad)}A"] = float(acc_pair / w_pair) if w_pair > 0 else math.nan
        out[f"region_neighbor_unique_partner_atoms_{int(rad)}A"] = float(acc_partner / w_pair) if w_pair > 0 else math.nan
        if rad == 6.0:
            # Region-specific symmetry-neighbor distances from altloc-state-aware contacts up to 6A.
            out["region_nearest_symmetry_atom_distance"] = float(acc_nearest / w_nearest) if w_nearest > 0 else math.nan
            out["region_median_symmetry_atom_distance"] = float(acc_med / w_med) if w_med > 0 else math.nan

    # Altloc-aware local atom density around region centroid in a 6A sphere.
    density_nm3 = occupancy_weighted_local_density_around_region(asu_rows, radius=6.0)
    out["region_centroid_local_atom_density_6A_per_nm3"] = float(density_nm3) if np.isfinite(density_nm3) else math.nan

    # B-factor and occupancy/disorder metrics.
    b_region = np.array([float(r["b_iso"]) for r in region_rows], dtype=float)
    b_global = np.array([float(r["b_iso"]) for r in asu_rows], dtype=float)
    occ_region = np.array([float(r["occ"]) for r in region_rows], dtype=float)
    alt_region = np.array([bool(r["altloc_set"]) for r in region_rows], dtype=bool)
    mu_region, _ = weighted_mean_and_std(b_region, w_region)
    mu_global, sd_global = weighted_mean_and_std(b_global, w_asu)
    if np.isfinite(mu_region):
        out["region_b_iso_mean"] = float(mu_region)
    if np.isfinite(mu_global):
        out["global_b_iso_mean"] = float(mu_global)
    if np.isfinite(mu_global) and mu_global > 0 and np.isfinite(mu_region):
        out["region_b_iso_ratio_to_global"] = float(mu_region / mu_global)
    if np.isfinite(sd_global) and sd_global > 0 and np.isfinite(mu_region) and np.isfinite(mu_global):
        out["region_b_iso_zscore_vs_global"] = float((mu_region - mu_global) / sd_global)
    out["region_occ_mean"] = float(np.mean(occ_region))
    out["region_occ_lt1_fraction"] = float(np.mean(occ_region < 0.999))
    out["region_altloc_atom_fraction"] = float(np.mean(alt_region))
    region_res_to_alt = {}
    for r in region_rows:
        rk = r["residue_key"]
        region_res_to_alt[rk] = region_res_to_alt.get(rk, False) or bool(r["altloc_set"])
    if region_res_to_alt:
        out["region_altloc_residue_fraction"] = float(np.mean(list(region_res_to_alt.values())))

    return out


def crystal_contact_metrics(
    structure: gemmi.Structure,
    cutoff: float,
    region_start: int,
    region_end: int,
    nearest_cutoff: float,
    max_interface_images: int,
) -> dict[str, float]:
    if len(structure) == 0:
        raise ValueError("Structure has no models.")
    model = structure[0]
    cell = structure.cell
    if cell.a <= 0 or cell.b <= 0 or cell.c <= 0:
        raise ValueError("Structure has invalid unit cell.")
    sg = structure.find_spacegroup()
    if sg is None:
        raise ValueError("Could not determine space group.")
    n_asu_per_cell = len(sg.operations())
    if n_asu_per_cell <= 0:
        raise ValueError("Invalid number of symmetry operations.")

    asu_rows = collect_asu_atoms_detailed(model, region_start, region_end)
    asu_mass_non_water = model_mass_non_water_da(model)
    matthews_vm_est = float(cell.volume / (asu_mass_non_water * n_asu_per_cell)) if asu_mass_non_water > 0 else math.nan
    solvent_fraction_est = float(1.0 - (1.23 / matthews_vm_est)) if np.isfinite(matthews_vm_est) and matthews_vm_est > 0 else math.nan

    ns = gemmi.NeighborSearch(model, cell, cutoff + 1.0).populate(include_h=False)
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameAsu
    contacts = cs.find_contacts(ns)

    atom_contact_keys: set[tuple[int, tuple[tuple, tuple]]] = set()
    residue_contact_keys: set[tuple[int, tuple[tuple, tuple]]] = set()
    atom_contact_keys_region_any: set[tuple[int, tuple[tuple, tuple]]] = set()
    residue_contact_keys_region_any: set[tuple[int, tuple[tuple, tuple]]] = set()
    image_count = Counter()

    for c in contacts:
        p1 = c.partner1
        p2 = c.partner2

        # Defensive guard even though SameAsu is ignored above.
        if c.image_idx == 0:
            continue

        if p1.residue.is_water() or p2.residue.is_water():
            continue
        if is_hydrogen(p1.atom) or is_hydrogen(p2.atom):
            continue

        a1 = atom_id(p1)
        a2 = atom_id(p2)
        pair_atoms = tuple(sorted((a1, a2)))
        atom_contact_keys.add((int(c.image_idx), pair_atoms))
        image_count[int(c.image_idx)] += 1

        r1 = residue_id(p1)
        r2 = residue_id(p2)
        pair_res = tuple(sorted((r1, r2)))
        residue_contact_keys.add((int(c.image_idx), pair_res))
        in_region_any = residue_in_region(p1.residue, region_start, region_end) or residue_in_region(
            p2.residue, region_start, region_end
        )
        if in_region_any:
            atom_contact_keys_region_any.add((int(c.image_idx), pair_atoms))
            residue_contact_keys_region_any.add((int(c.image_idx), pair_res))

    ns_nearest = gemmi.NeighborSearch(model, cell, nearest_cutoff + 1.0).populate(include_h=False)
    cs_nearest = gemmi.ContactSearch(nearest_cutoff)
    cs_nearest.ignore = gemmi.ContactSearch.Ignore.SameAsu
    near_contacts = cs_nearest.find_contacts(ns_nearest)
    near_distances = []
    near_atom_keys: set[tuple[int, tuple[tuple, tuple]]] = set()
    for c in near_contacts:
        if c.image_idx == 0:
            continue
        p1 = c.partner1
        p2 = c.partner2
        if p1.residue.is_water() or p2.residue.is_water():
            continue
        if is_hydrogen(p1.atom) or is_hydrogen(p2.atom):
            continue
        near_distances.append(float(c.dist))
        a1 = atom_id(p1)
        a2 = atom_id(p2)
        pair_atoms = tuple(sorted((a1, a2)))
        near_atom_keys.add((int(c.image_idx), pair_atoms))

    bsa_all_metrics = estimate_bsa_top_images_altloc_state_aware(
        structure=structure,
        asu_rows=asu_rows,
        image_counts=image_count,
        max_images=max_interface_images,
        ns_ref=ns,
    )
    bsa_metrics = {
        "interface_images_used": bsa_all_metrics["interface_images_used"],
        "interface_bsa_top_images_sum": bsa_all_metrics["interface_bsa_top_images_sum"],
        "interface_bsa_top_images_mean": bsa_all_metrics["interface_bsa_top_images_mean"],
        "interface_bsa_top_image_max": bsa_all_metrics["interface_bsa_top_image_max"],
    }
    bsa_region_metrics = {
        "interface_region_images_used": bsa_all_metrics["interface_region_images_used"],
        "interface_region_bsa_top_images_sum": bsa_all_metrics["interface_region_bsa_top_images_sum"],
        "interface_region_bsa_top_images_mean": bsa_all_metrics["interface_region_bsa_top_images_mean"],
        "interface_region_bsa_top_image_max": bsa_all_metrics["interface_region_bsa_top_image_max"],
    }
    region_local = local_region_metrics(
        structure=structure,
        model=model,
        asu_rows=asu_rows,
        region_start=region_start,
        region_end=region_end,
        ns_ref=ns,
        nearest_cutoff=nearest_cutoff,
    )

    n_atoms = count_non_h_non_water_atoms(model)
    n_atoms_region = count_non_h_non_water_atoms_in_region(model, region_start, region_end)
    atom_contacts = len(atom_contact_keys)
    residue_contacts = len(residue_contact_keys)
    atom_contacts_region_any = len(atom_contact_keys_region_any)
    residue_contacts_region_any = len(residue_contact_keys_region_any)
    scale = n_atoms / 1000.0 if n_atoms > 0 else math.nan
    scale_region = n_atoms_region / 1000.0 if n_atoms_region > 0 else math.nan

    return {
        "region_start": int(region_start),
        "region_end": int(region_end),
        "cell_a": float(cell.a),
        "cell_b": float(cell.b),
        "cell_c": float(cell.c),
        "cell_alpha": float(cell.alpha),
        "cell_beta": float(cell.beta),
        "cell_gamma": float(cell.gamma),
        "cell_volume": float(cell.volume),
        "spacegroup_hm": structure.spacegroup_hm,
        "n_asu_per_cell": int(n_asu_per_cell),
        "asu_mass_non_water_da": float(asu_mass_non_water),
        "matthews_vm_est": float(matthews_vm_est) if np.isfinite(matthews_vm_est) else math.nan,
        "solvent_fraction_est": float(solvent_fraction_est) if np.isfinite(solvent_fraction_est) else math.nan,
        "atoms_non_h_non_water": int(n_atoms),
        "atoms_non_h_non_water_region": int(n_atoms_region),
        "crystal_contacts_atom": int(atom_contacts),
        "crystal_contacts_residue": int(residue_contacts),
        "crystal_contacts_atom_region_any": int(atom_contacts_region_any),
        "crystal_contacts_residue_region_any": int(residue_contacts_region_any),
        "nearest_symmetry_atom_distance": float(np.min(near_distances)) if near_distances else math.nan,
        "median_symmetry_atom_distance": float(np.median(near_distances)) if near_distances else math.nan,
        "symmetry_contacts_within_nearest_cutoff": int(len(near_atom_keys)),
        "crystal_contacts_atom_per_1k_atoms": float(atom_contacts / scale) if n_atoms > 0 else math.nan,
        "crystal_contacts_residue_per_1k_atoms": float(residue_contacts / scale) if n_atoms > 0 else math.nan,
        "crystal_contacts_atom_region_any_per_1k_region_atoms": float(atom_contacts_region_any / scale_region)
        if n_atoms_region > 0
        else math.nan,
        "crystal_contacts_residue_region_any_per_1k_region_atoms": float(residue_contacts_region_any / scale_region)
        if n_atoms_region > 0
        else math.nan,
        **bsa_metrics,
        **bsa_region_metrics,
        **region_local,
    }


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size == 0 or ya.size == 0:
        return math.nan
    gt = 0
    lt = 0
    for xv in xa:
        gt += int(np.sum(xv > ya))
        lt += int(np.sum(xv < ya))
    return (gt - lt) / float(xa.size * ya.size)


def compare_groups(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out_rows = []
    g1 = "C 121"
    g2 = "I 121"
    for metric in metrics:
        s1 = pd.to_numeric(df.loc[df["group"] == g1, metric], errors="coerce").dropna()
        s2 = pd.to_numeric(df.loc[df["group"] == g2, metric], errors="coerce").dropna()
        row = {
            "metric": metric,
            "n_C121": int(s1.size),
            "n_I121": int(s2.size),
            "mean_C121": float(s1.mean()) if s1.size else math.nan,
            "mean_I121": float(s2.mean()) if s2.size else math.nan,
            "median_C121": float(s1.median()) if s1.size else math.nan,
            "median_I121": float(s2.median()) if s2.size else math.nan,
            "delta_mean_C121_minus_I121": float(s1.mean() - s2.mean()) if s1.size and s2.size else math.nan,
            "cliffs_delta_C121_vs_I121": cliffs_delta(s1, s2),
            "mannwhitneyu_pvalue": math.nan,
        }
        if mannwhitneyu is not None and s1.size > 0 and s2.size > 0:
            try:
                _, p = mannwhitneyu(s1, s2, alternative="two-sided")
                row["mannwhitneyu_pvalue"] = float(p)
            except Exception:
                row["mannwhitneyu_pvalue"] = math.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def compute_loose_packing_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Positive contributions indicate looser packing.
    components = [
        ("solvent_fraction_est", +1.0),
        ("matthews_vm_est", +1.0),
        ("region_nearest_symmetry_atom_distance", +1.0),
        ("region_sasa_mean_atom", +1.0),
        ("region_buried_atom_fraction_sasa_lt_10", -1.0),
        ("region_centroid_local_atom_density_6A_per_nm3", -1.0),
        ("crystal_contacts_atom_per_1k_atoms", -1.0),
        ("crystal_contacts_atom_region_any_per_1k_region_atoms", -1.0),
        ("interface_region_bsa_top_images_sum", -1.0),
        ("region_neighbor_contacts_6A", -1.0),
    ]
    zcols = []
    for metric, sign in components:
        if metric not in out.columns:
            continue
        vals = pd.to_numeric(out[metric], errors="coerce")
        mu = float(vals.mean(skipna=True))
        sd = float(vals.std(skipna=True))
        if not np.isfinite(sd) or sd <= 0:
            z = pd.Series(0.0, index=out.index, dtype=float)
            z[vals.isna()] = np.nan
        else:
            z = (vals - mu) / sd
        z *= sign
        zname = f"z_{metric}"
        out[zname] = z
        zcols.append(zname)
    out["loose_packing_score"] = out[zcols].mean(axis=1, skipna=True) if zcols else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PDB structures from IDs in data/tables/b2m_group_metrics.csv, compute "
            "unit-cell volume and crystal-contact metrics, and compare C121 vs I121."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=Path("data/tables/b2m_group_metrics.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("out/tables/b2m_ccp_cell_contacts.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("out/tables/b2m_ccp_group_comparison.csv"))
    parser.add_argument("--summary-txt", type=Path, default=Path("out/tables/b2m_ccp_group_comparison.txt"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/pdb_downloads"))
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--nearest-cutoff", type=float, default=20.0)
    parser.add_argument("--max-interface-images", type=int, default=3)
    parser.add_argument("--region-start", type=int, default=50)
    parser.add_argument("--region-end", type=int, default=65)
    parser.add_argument("--overwrite-downloads", action="store_true")
    args = parser.parse_args()
    region_lo = min(args.region_start, args.region_end)
    region_hi = max(args.region_start, args.region_end)
    if freesasa is not None:
        freesasa.setVerbosity(freesasa.nowarnings)

    inp = pd.read_csv(args.input_csv)
    required = {"pdb_id", "space_group"}
    missing = sorted(required - set(inp.columns))
    if missing:
        raise SystemExit(f"Input CSV missing columns: {', '.join(missing)}")

    rows = []
    for _, rec in inp.iterrows():
        pdb_id = str(rec["pdb_id"]).strip().lower()
        group = normalize_group_name(rec["space_group"])
        out_row = {
            "pdb_id": pdb_id,
            "group": group,
            "download_path": "",
            "download_format": "",
            "error": "",
        }
        try:
            path, fmt = download_structure(pdb_id, args.download_dir, overwrite=args.overwrite_downloads)
            out_row["download_path"] = str(path)
            out_row["download_format"] = fmt
            structure = gemmi.read_structure(str(path))
            metrics = crystal_contact_metrics(
                structure,
                cutoff=args.contact_cutoff,
                region_start=region_lo,
                region_end=region_hi,
                nearest_cutoff=args.nearest_cutoff,
                max_interface_images=args.max_interface_images,
            )
            out_row.update(metrics)
        except Exception as exc:
            out_row["error"] = str(exc)
        rows.append(out_row)

    result_df = pd.DataFrame(rows).sort_values(["group", "pdb_id"]).reset_index(drop=True)
    result_df = compute_loose_packing_score(result_df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)

    good = result_df[result_df["error"] == ""].copy()
    metric_cols = [
        "cell_volume",
        "matthews_vm_est",
        "solvent_fraction_est",
        "crystal_contacts_atom",
        "crystal_contacts_residue",
        "nearest_symmetry_atom_distance",
        "median_symmetry_atom_distance",
        "symmetry_contacts_within_nearest_cutoff",
        "crystal_contacts_atom_per_1k_atoms",
        "crystal_contacts_residue_per_1k_atoms",
        "crystal_contacts_atom_region_any",
        "crystal_contacts_residue_region_any",
        "crystal_contacts_atom_region_any_per_1k_region_atoms",
        "crystal_contacts_residue_region_any_per_1k_region_atoms",
        "interface_bsa_top_images_sum",
        "interface_bsa_top_images_mean",
        "interface_bsa_top_image_max",
        "interface_region_bsa_top_images_sum",
        "interface_region_bsa_top_images_mean",
        "interface_region_bsa_top_image_max",
        "region_sasa_sum",
        "region_sasa_mean_atom",
        "region_sasa_median_atom",
        "region_buried_atom_fraction_sasa_lt_5",
        "region_buried_atom_fraction_sasa_lt_10",
        "region_neighbor_contacts_4A",
        "region_neighbor_unique_partner_atoms_4A",
        "region_neighbor_contacts_6A",
        "region_neighbor_unique_partner_atoms_6A",
        "region_centroid_local_atom_density_6A_per_nm3",
        "region_nearest_symmetry_atom_distance",
        "region_median_symmetry_atom_distance",
        "region_b_iso_mean",
        "global_b_iso_mean",
        "region_b_iso_ratio_to_global",
        "region_b_iso_zscore_vs_global",
        "region_occ_mean",
        "region_occ_lt1_fraction",
        "region_altloc_atom_fraction",
        "region_altloc_residue_fraction",
        "region_map_fraction_above_threshold",
        "region_map_gradient_sharpness",
        "region_map_local_variance",
        "loose_packing_score",
    ]
    comp = compare_groups(good, metric_cols)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.summary_csv, index=False)

    lines = []
    lines.append("C121 vs I121 lattice comparison")
    lines.append(f"Input rows: {len(result_df)}")
    lines.append(f"Successful structures: {len(good)}")
    lines.append(f"Region-contact window: residues {region_lo}-{region_hi} (contact has at least one partner in window)")
    lines.append(f"FreeSASA available: {'yes' if freesasa is not None else 'no'}")
    lines.append("Map-local compactness metrics are placeholders (NaN) unless map data are provided.")
    lines.append("")

    # Flag possible systematic differences with a practical-effect + p-value rule.
    flagged = []
    for _, r in comp.iterrows():
        p = r["mannwhitneyu_pvalue"]
        d = abs(r["cliffs_delta_C121_vs_I121"]) if pd.notna(r["cliffs_delta_C121_vs_I121"]) else math.nan
        maybe_systematic = pd.notna(p) and p < 0.05 and pd.notna(d) and d >= 0.33
        direction = "C121 > I121" if r["delta_mean_C121_minus_I121"] > 0 else "C121 < I121"
        lines.append(
            f"{r['metric']}: mean_C121={r['mean_C121']:.4g}, "
            f"mean_I121={r['mean_I121']:.4g}, {direction}, "
            f"cliffs_delta={r['cliffs_delta_C121_vs_I121']:.3f}, p={r['mannwhitneyu_pvalue']:.3g}"
        )
        if maybe_systematic:
            flagged.append(str(r["metric"]))

    lines.append("")
    if flagged:
        lines.append("Potential systematic differences detected:")
        for m in flagged:
            lines.append(f"- {m}")
    else:
        lines.append("No strong systematic differences detected with the current rule (p < 0.05 and |Cliff's delta| >= 0.33).")

    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
    args.summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Saved per-structure metrics: {args.output_csv}")
    print(f"Saved group comparison CSV: {args.summary_csv}")
    print(f"Saved group comparison text: {args.summary_txt}")
    print("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
