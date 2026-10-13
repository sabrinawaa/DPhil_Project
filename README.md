# VHEE dual-scattering design and dosimetry

Simulation and analysis code for designing dual-scattering systems (a flat primary
scatterer S1 and a Gaussian-profile secondary scatterer S2) that turn a very-high-energy
electron (VHEE) beam into a uniform field. It also covers comparing the designs with
film (RCF) and YAG screen measurements at CLEAR (CERN) and CLARA (Daresbury).

The workflow is:

1. **Track the beam** through the accelerator lattice with [RF-Track](https://gitlab.cern.ch/rf-track)
   (`CLEAR_line.py`, `CLEAR_2nd_line.py`).
2. **Optimise the scatterers** (S1 thickness, S2 radius and depth) in RF-Track against a
   uniformity merit function (`design_optimisation.py`, `flatness.py`).
3. **Check the design** with a full Monte Carlo simulation in TOPAS/Geant4
   (`partrec_gaussian_optimiser_utils.py`, `topasToDose.py`).
4. **Compare with measurements** on RCF films and YAG screens (`YAG_RCF_analysis.py` and the
   `*_processing` notebooks).

## Setup

Python 3.11 conda environment (`RFT_topas_bdsim`) with:

- `RF_Track` (tested with 2.7.3)
- `numpy`, `scipy` (≥ 1.14 for COBYQA in `optimiser_comparison.py`), `matplotlib`, `pandas`
- `scikit-learn` (nearest-neighbour merit), `cma` (CMA-ES)
- `astropy`, `topas2numpy` (TOPAS input/output)
- `tifffile`, `plotly`, `h5py` (film / screen analysis)
- `pybdsim`, `pyg4ometry` (BDSIM notebooks only), `scikit-optimize` (`Optimisation_funcs.py` only)

TOPAS must be installed under `/Applications/`. Set `TOPAS_G4_DATA_DIR=/Applications/G4Data`
before running it. Without this TOPAS exits silently and the old dose CSVs are left in place,
so they can be mistaken for new results.

TOPAS input and output files are written to `data/topas/`.

## Core modules

| File | Contents |
|---|---|
| `CLEAR_line.py` | Builds the CLEAR main beamline in RF-Track from `CLEAR_Beamline_Survey.txt`, with quadrupole current → k1 conversion. |
| `CLEAR_2nd_line.py` | Same for the CLEAR second beamline (`CLEAR_2nd_Beamline_Survey.txt`), including the two ±22° dipoles as `RBend`s. Dipoles are set for `P_ref`; a beam at a different momentum is offset by the ~2.7 m dispersion of the dogleg. |
| `design_optimisation.py` | Shared scatterer design code: S2 slice profile (`get_s2_params`), lattice builders (`add_s1`, `add_s2`, `build_scatterer_lattice`), the merit function `scatterer_merit` and the optimiser objective `loss`. |
| `flatness.py` | Merit function building blocks: projection-based uniform-disk merit (`merit_beam_Uniform`, `merit_uniform_xy`), k-th nearest-neighbour statistic (`nearest_neighbor_test`), double-Gaussian fit (`loss_2gauss`). |
| `uniformity_fit.py` | Super-Gaussian / double-Gaussian fits to dose maps, `flatness()` profile metric, `plot_phsp` phase-space plots. |
| `partrec_gaussian_optimiser_utils.py` | Writes TOPAS input files: beam sources, flat and Gaussian scatterers, cylinders/boxes, water tank scorers, phase-space import/export, `run_topas`. |
| `topasToDose.py` | Converts TOPAS dose CSVs into calibrated dose maps (`getDosemap`). |
| `partrec_foil_plotting.py` | Plots TOPAS phase-space output (transverse distributions, energy spectra). |
| `RF_track_utils.py` | RF-Track ↔ TOPAS helper class. |
| `YAG_RCF_analysis.py` | Film/screen analysis: beam centroid, masks, dose slices, `plot_dose1` (flatness, r90, ratios with errors). |
| `quad_scan_helper.py` | Thin- and thick-lens quadrupole-scan emittance fits. |
| `sabrina_rf_topas_conversion.py` | Standalone RF-Track → TOPAS phase-space export. |

### Merit functions (`design_optimisation.scatterer_merit`)

Choose one with `merit=`; lower is better. All are evaluated at the measurement plane.

| Name | What it measures |
|---|---|
| `'nn'` | CV of the distance from each particle to its k-th nearest neighbour, over the central 60% of particles, plus a size term. `nn_frac` sets k as a fraction of the particle number. The default k = 1 is dominated by Poisson noise; use `nn_frac=0.001`. |
| `'andrea'` | Percentiles of projections at 60 angles compared with a uniform disk of radius `target_size`, plus a transmission penalty set by `transmission_goal`. |
| `'2gauss'` | Double-Gaussian fit to central x/y slices; penalises the deviation of \|x0/σ\| from `gauss_ratio` (1.0 is exactly flat). |
| `'med_flatness'` | (max−min)/(max+min) of the smoothed central 80% of the central x/y slices. |

`loss(..., seed=s)` reseeds RF-Track before tracking (common random numbers), so the
objective is deterministic within an optimiser run.

## Main notebooks

### Scatterer design
- **`CLEAR_RD.ipynb`**: current CLEAR second-beamline design. It tracks through the beamline,
  optimises (s1_l, s2_width, s2_depth) for a 25 mm field with Powell + restarts + re-scoring on
  fresh seeds, checks the result, then sets up the TOPAS run.
- `CLARA_design.ipynb`, `CLARA_design_copy.ipynb`, `CLARA_design_quad.ipynb`: the same procedure
  for CLARA at 150/200/250 MeV (2- and 3-parameter versions, and a version optimising the S2
  position).
- `optimise_CLEAR.ipynb`, `optimise_CLEAR_twiss.ipynb`, `optimise_CLEAR_twiss_small.ipynb`,
  `optimise_CLEAR_phsp_quadscans.ipynb`: older CLEAR main-line optimisation of the quadrupole
  settings and scatterers, starting from Twiss parameters or from quad-scan phase spaces.
- `Flateness_params.ipynb`: exploration of the flatness/merit parameters.
- `gaussian_scatterer_linearity_test.ipynb`: TOPAS test of whether Gaussian scatterers superpose.

### Simulation vs measurement
- `simulate_PDD_CLEAR.ipynb`, `PDD_CLEAR.ipynb`: percentage depth dose, simulated and measured.
- `cam_data.ipynb` (+ `cam_discrepancy_checks.py`): water-depth scan, RF-Track + TOPAS vs film,
  and the investigation of the discrepancy between them.
- `compare_film_sim_slices.ipynb` (+ `.py`): overlays film and simulated dose slices at matching depths.
- `fake_quad_scan_CLEAR.ipynb`: simulated quadrupole scans for the screen-875 analysis.

### Measurement processing
- `CLEAR_RCF_processing.ipynb`, `RCF_processing.ipynb`, `CLEAR_rcf_calib.ipynb`: RCF film
  processing and calibration (`compare_computers_pixel_values.py` checks scans of the same films
  made on two computers).
- `YAG_processing.ipynb`, `YAG_processing_CLEAR.ipynb`: YAG screen images.
- `TIMBER_charge_plot.ipynb`: per-pulse charge from CERN TIMBER logging.

### BDSIM (alternative tracking)
- `pylattice.ipynb`, `rft_bdsim.ipynb`, `test_bdsim.ipynb`, `RFT_bdsim.py`, `bdsim_utils.py`.

## Reproducing the optimiser comparison

`optimiser_comparison.py` reproduces the optimiser comparison in the paper. It runs Powell
(5 restarts), CMA-ES (3 runs) and COBYQA (5 restarts) on the CLEAR_RD setup, then re-scores
every result on 10⁵ particles with the same 5 fresh seeds:

```bash
python optimiser_comparison.py              # full run, ~20-25 min on 4 workers
python optimiser_comparison.py --quick      # smoke test (~15 s)
python optimiser_comparison.py --tag mytag  # write results to optimiser_comparison/mytag/
```

All settings are in the `CONFIG` dictionary at the top of the script. Results
(`results.json`, `summary.md` and one log per run) go to `optimiser_comparison/`, which is not
tracked by git.

## Older scripts

These are kept for reference and have been superseded by the modules and notebooks above:
`optimise_CLEAR.py`, `Optimisation_funcs.py`, `CLEAR_scatterer.py`, `CLEAR_scatterer1.py`,
`RF_track_4quad.py`, `RF_track_4quad_optimiser.py`, `RF_track_trial.py`, `combined_trial.py`,
`find_ks.py`, `partrec_main.py`, `step1.py`, `Cameron_step2.py`, `search_out.py`,
`export_csvcombined.py` (merges TOPAS CSVs), and `submitFileScripter.py` (batch job
submission files).

## Known modelling differences between RF-Track and TOPAS

These cause the RF-Track optimum to disagree with the TOPAS field:

- **Air**: RF-Track drifts are vacuum, while the TOPAS world is `Air`. Over the ~3 m from S1 to
  the tank this makes the field about 10 mm larger at r90 and fills in the centre. Model the
  drifts to match the real vacuum/air layout.
- **S2 material**: S2 is built in RF-Track with Z = 37, A = 288.31 (Z/A = 0.13). PEEK has
  Z/A ≈ 0.52.
- **S2 orientation**: in TOPAS the tip faces upstream; in RF-Track the base (largest slice) does.
  The two codes also measure S2 positions from different faces.
- **Physics**: RF-Track has no bremsstrahlung, no secondary particles and no single-scattering
  tails; TOPAS (Geant4 `g4em-standard_opt4`) includes all three.
