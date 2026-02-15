# b2m Figure Appendix

This appendix supports reproduction of results presented in the paper **"Density-guided AlphaFold3 uncovers unmodelled conformations in** $\\mathbf{\\beta}_2$**-microglobulin"**.

All figures documented in this appendix are also part of reproducing **Figure 2** in the paper.

This appendix contains the non-primary figures and their reproduction commands.

Run commands from the repository root.

## A1) Packing Metric Boxplots (C121 vs I121)
Per-structure box+strip comparisons of packing/contact metrics.

Reproduce:
```bash
python3 src/plot_ccp_packing_boxplots.py
```

Outputs:
- `out/figures/loose_packing_score_box.png`
![loose_packing_score_box](out/figures/loose_packing_score_box.png)

- `out/figures/interface_region_bsa_top_images_sum_box.png`
![interface_region_bsa_top_images_sum_box](out/figures/interface_region_bsa_top_images_sum_box.png)

- `out/figures/crystal_contacts_atom_region_any_per_1k_region_atoms_box.png`
![crystal_contacts_atom_region_any_per_1k_region_atoms_box](out/figures/crystal_contacts_atom_region_any_per_1k_region_atoms_box.png)

- `out/figures/region_sasa_mean_atom_box.png`
![region_sasa_mean_atom_box](out/figures/region_sasa_mean_atom_box.png)

- `out/figures/region_neighbor_contacts_6a_box.png`
![region_neighbor_contacts_6a_box](out/figures/region_neighbor_contacts_6a_box.png)

- `out/figures/region_nearest_symmetry_atom_distance_box.png`
![region_nearest_symmetry_atom_distance_box](out/figures/region_nearest_symmetry_atom_distance_box.png)

- `out/figures/matthews_vm_est_box.png`
![matthews_vm_est_box](out/figures/matthews_vm_est_box.png)

- `out/figures/solvent_fraction_est_box.png`
![solvent_fraction_est_box](out/figures/solvent_fraction_est_box.png)

## A2) Altloc-Split Packing Boxplots
Same metrics as above, split by altloc modeled vs no-altloc.

Reproduce:
```bash
python3 src/plot_ccp_packing_boxplots_altloc_split.py
```

Outputs:
- `out/figures/loose_packing_score_box_altloc.png`
![loose_packing_score_box_altloc](out/figures/loose_packing_score_box_altloc.png)

- `out/figures/interface_region_bsa_top_images_sum_box_altloc.png`
![interface_region_bsa_top_images_sum_box_altloc](out/figures/interface_region_bsa_top_images_sum_box_altloc.png)

- `out/figures/crystal_contacts_atom_region_any_per_1k_region_atoms_box_altloc.png`
![crystal_contacts_atom_region_any_per_1k_region_atoms_box_altloc](out/figures/crystal_contacts_atom_region_any_per_1k_region_atoms_box_altloc.png)

- `out/figures/region_sasa_mean_atom_box_altloc.png`
![region_sasa_mean_atom_box_altloc](out/figures/region_sasa_mean_atom_box_altloc.png)

- `out/figures/region_neighbor_contacts_6a_box_altloc.png`
![region_neighbor_contacts_6a_box_altloc](out/figures/region_neighbor_contacts_6a_box_altloc.png)

- `out/figures/region_neighbor_contacts_4a_box_altloc.png`
![region_neighbor_contacts_4a_box_altloc](out/figures/region_neighbor_contacts_4a_box_altloc.png)

- `out/figures/region_nearest_symmetry_atom_distance_box_altloc.png`
![region_nearest_symmetry_atom_distance_box_altloc](out/figures/region_nearest_symmetry_atom_distance_box_altloc.png)

- `out/figures/region_b_iso_mean_box_altloc.png`
![region_b_iso_mean_box_altloc](out/figures/region_b_iso_mean_box_altloc.png)

- `out/figures/matthews_vm_est_box_altloc.png`
![matthews_vm_est_box_altloc](out/figures/matthews_vm_est_box_altloc.png)

- `out/figures/solvent_fraction_est_box_altloc.png`
![solvent_fraction_est_box_altloc](out/figures/solvent_fraction_est_box_altloc.png)

## A3) Additional Packing-Derived Plots
### A3.1) Extra metrics in the non-altloc style
Reproduce:
```bash
python3 src/plot_ccp_packing_boxplots.py \
  --metrics region_neighbor_contacts_4A region_b_iso_mean
```

Outputs:
- `out/figures/region_neighbor_contacts_4a_box.png`
![region_neighbor_contacts_4a_box](out/figures/region_neighbor_contacts_4a_box.png)

- `out/figures/region_b_iso_mean_box.png`
![region_b_iso_mean_box](out/figures/region_b_iso_mean_box.png)

### A3.2) Altloc-split cell volume
Reproduce:
```bash
python3 src/plot_ccp_packing_boxplots_altloc_split.py --metrics cell_volume
```

Output:
- `out/figures/cell_volume_box_altloc.png`
![cell_volume_box_altloc](out/figures/cell_volume_box_altloc.png)

## A4) Cell Volume Boxplot (Group-Level)
Reproduce:
```bash
python3 src/plot_cell_volume_boxplot.py
```

Output:
- `out/figures/cell_volume_box.png`
![cell_volume_box](out/figures/cell_volume_box.png)

## A5) Region Contact Count Plots
### A5.1) Atom-level region contact boxplot
Reproduce:
```bash
python3 src/plot_region_contacts_box.py \
  --metric crystal_contacts_atom_region_any \
  -o out/figures/region_contacts_atom_box.png
```

Output:
- `out/figures/region_contacts_atom_box.png`
![region_contacts_atom_box](out/figures/region_contacts_atom_box.png)

### A5.2) Residue-level region contact boxplot
Reproduce:
```bash
python3 src/plot_region_contacts_box.py \
  --metric crystal_contacts_residue_region_any \
  -o out/figures/region_contacts_residue_box.png
```

Output:
- `out/figures/region_contacts_residue_box.png`
![region_contacts_residue_box](out/figures/region_contacts_residue_box.png)

### A5.3) Atom-level region contact histogram
Reproduce:
```bash
python3 src/plot_region_contacts_hist.py \
  --metric crystal_contacts_atom_region_any \
  -o out/figures/region_contacts_atom_hist.png
```

Output:
- `out/figures/region_contacts_atom_hist.png`
![region_contacts_atom_hist](out/figures/region_contacts_atom_hist.png)

## A6) Relative Improvement Violin Plots
Reproduce:
```bash
python3 src/plot_relative_improvement_violins.py
```

Outputs:
- `out/figures/cosine_improvement_violin.png`
![cosine_improvement_violin](out/figures/cosine_improvement_violin.png)

- `out/figures/r_work_improvement_violin.png`
![r_work_improvement_violin](out/figures/r_work_improvement_violin.png)

- `out/figures/r_free_improvement_violin.png`
![r_free_improvement_violin](out/figures/r_free_improvement_violin.png)

## A7) Crystal Contact Profile KDE
Per-residue, altloc-aware expected crystal-contact profile.

Reproduce:
```bash
python3 src/plot_crystal_contact_profile_kde.py
```

Outputs:
- `out/figures/crystal_contact_profile_kde.png`
![crystal_contact_profile_kde](out/figures/crystal_contact_profile_kde.png)

- `out/tables/crystal_contact_profile.csv`

## A8) SASA Profile KDE
Per-residue, altloc-aware expected SASA profile.

Reproduce:
```bash
python3 src/plot_sasa_profile_kde.py
```

Outputs:
- `out/figures/sasa_profile_kde.png`
![sasa_profile_kde](out/figures/sasa_profile_kde.png)

- `out/tables/sasa_profile.csv`

## Legacy/Archived Output Variants
Some files in `out/figures/` with suffixes like `_2`, `_3`, etc. are archived variants from earlier parameter sweeps and are kept for reference.
