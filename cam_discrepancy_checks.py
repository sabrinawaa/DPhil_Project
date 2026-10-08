"""Step-by-step hunt for the film-vs-simulation dose discrepancy in cam_data.ipynb.

One config (Quads=4447, collimated) is tracked once with RF_Track and the same phase space is
fed to a series of TOPAS geometries, each changing one thing towards the sab-sim/partrec_test
setup:

  V0_old     cam_data geometry exactly as previously run (20 mm tank, 1 z-bin, Surface line)
  V1_fixed   + fixes: no Surface, 21 mm tank in 1 mm depth slices, Kapton spelling, S2 indexing
  V2_noYAG   + YAG, air gap, vacuum window and tank window removed
  V3_col     + collimator matched to sab-sim (49 mm steel + 47 mm lead, R5-500)
  V4_dist    + S1/S2/collimator/water distances matched to sab-sim
  V4a_s1s2   V3 + only the S1 -> S2 gap matched to sab-sim (532 -> 445.65 mm)
  V4b_s2water V3 + only the S2 -> collimator / water distances matched to sab-sim

Usage:  python cam_discrepancy_checks.py [variant ...]    (default: all; finished runs are reused)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import RF_Track  # must come before anything importing sklearn (see cam_data.ipynb cell 1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from topas2numpy import BinnedResult

from CLEAR_line import get_beamline
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from YAG_RCF_analysis import plot_dose1
from topasToDose import getDosemap

project = Path("/Users/sabrinawang/Desktop/DPhil_Project")
out_dir = project / "Cam_08_23" / "Simulations" / "discrepancy_checks"
out_dir.mkdir(parents=True, exist_ok=True)

quads, col = 4447, True
n_particles = int(5e5)
phsp_name = str(out_dir / f"Cam0823_Q{quads}")
tank_depth = 21          # water only to just past the 20 mm film, for speed; slice [20, 21] mm is read
film_depths = [20]

# cam_data S1/S2 ("large" option)
s1_l = 0.1
s2_thickness = [0.688, 0.778, 0.581, 0.386]
s2_radii = [0.4, 0.8, 1.2, 1.6]
s2_depth = 2.43


def track_beam():
    if os.path.exists(phsp_name + ".phsp"):
        return
    Twiss = RF_Track.Bunch6d_twiss()
    Twiss.beta_x, Twiss.beta_y = 5.45, 22.54
    Twiss.alpha_x, Twiss.alpha_y = 0.18, 0.58
    Twiss.emitt_x, Twiss.emitt_y = 12.9, 8.5
    Twiss.mean_xp = Twiss.mean_yp = 0.0
    quad_currents = [0.0] * 11
    quad_currents[6], quad_currents[7] = float(str(quads)[:2]), float(str(quads)[2:])
    lattice = get_beamline(str(project / "CLEAR_Beamline_Survey.txt"), "CA.QFD0760", "CA.DHJ0840",
                           200.0, quad_currents, Q=-1)
    B0 = RF_Track.Bunch6d_QR(RF_Track.electronmass, 10 * RF_Track.nC, -1, 200.0, Twiss, n_particles)
    R = lattice.track(B0).get_phase_space('%x %xp %y %yp %E %z')
    setup = partrec_gaussian_optimiser_utils(file_directory=str(out_dir) + "/", input_filename="_phsp_tmp.txt")
    setup.export_phsp(R, phsp_name + ".phsp")
    setup.write_header(R, phsp_name + ".header")
    setup.file.close()
    print(f"tracked beam: sx={R[:,0].std():.3f} mm sy={R[:,2].std():.3f} mm "
          f"sx'={R[:,1].std():.3f} sy'={R[:,3].std():.3f} mrad E={R[:,4].mean():.2f} MeV")


def build(variant):
    """Write the TOPAS file for one variant; returns (setup, dose csv name, tank depth, z bins)."""
    setup = partrec_gaussian_optimiser_utils(file_directory=str(out_dir) + "/",
                                             input_filename=f"topas_{variant}.txt")
    setup.import_beam_topas(phsp_name, position=0)

    if variant == "V4_dist":
        # sab-sim positions, measured from its source plane (1 mm upstream of S1)
        s1_pos, s2_front, col_front, tank_front = 1.0, 446.75, 1974.0, 2244.0
    else:
        s1_pos = 84
        s2_front = s1_pos + s1_l + (445.65 if variant == "V4a_s1s2" else 532)
        s2_back = s2_front + sum(s2_thickness)
        if variant == "V4b_s2water":
            # sab-sim distances downstream of S2's back face
            col_front, tank_front = s2_back + 1524.82, s2_back + 1794.82
        else:
            col_front = s2_front + s2_depth + 1585 - 25   # add_collimator takes the centre
            tank_front = s2_back + 2025

    setup.add_flat_scatterer(s1_l, "Aluminum", s1_pos)
    if variant == "V0_old":
        # reproduce the old off-by-one loop (same geometry, rotated names)
        for i in range(len(s2_thickness)):
            setup.add_cylinder("S2_slice_" + str(i), s2_thickness[i - 1], 0, s2_radii[i - 1], "Peek",
                               s2_front + sum(s2_thickness[:i - 1]))
    else:
        for i in range(len(s2_thickness)):
            setup.add_cylinder("S2_slice_" + str(i), s2_thickness[i], 0, s2_radii[i], "Peek",
                               s2_front + sum(s2_thickness[:i]))
    kapton = "kapton" if variant == "V0_old" else "Kapton"
    setup.add_cylinder("kapton_holder", 0.025, 0, 33, kapton, s2_front + sum(s2_thickness) + 0.025)

    with_screens = variant in ("V0_old", "V1_fixed")
    if with_screens:
        setup.add_cylinder("vacuum_window", 0.075, 0, 500, kapton, s2_front + sum(s2_thickness) + 1489)

    if variant.startswith(("V3", "V4")):
        setup.add_cylinder("col1", 49, 5, 500, "Steel", col_front)
        setup.add_cylinder("col2", 47, 5, 500, "Lead", col_front + 49)
    else:
        setup.add_collimator(50, 15, 5, 50, col_front)

    if with_screens:
        setup.add_box("YAG", 0.55, 35, 40, "YAG", s2_front + sum(s2_thickness) + 1939, rotation=45)
        setup.add_box("air2", 70, 100, 100, "Air", s2_front + sum(s2_thickness) + 1953)
        setup.add_cylinder("tank_window", 0.075, 0, 100, kapton, s2_front + sum(s2_thickness) + 2024)

    if variant == "V0_old":
        depth, z_bins = 20, 1
        setup.add_tank_bins(tank_front, depth, 100, 100, z_bins, variant, width=30,
                            surface="Tank/ZPlusSurface")
    else:
        depth, z_bins = tank_depth, tank_depth
        setup.add_tank_bins(tank_front, depth, 100, 100, z_bins, variant, width=30)
    return setup, out_dir / f"DoseAtTank{depth}_{variant}.csv", depth, z_bins


def dose_per_nC(raw2d, x, y):
    """Centre dose (Gy per nC of THz1) from plot_dose1, as in cam_data.ipynb."""
    sim_charge_nC = n_particles * 1.60217663e-19 * 1e9
    dmap = np.rot90(raw2d / sim_charge_nC)
    res = plot_dose1(dmap, "RCF", x, y, strip_width=2)
    plt.close(res[0])
    return res[3], res[4], res[5], dmap


profiles = {}
film_rows = []


def load_film_shots():
    """Every ININWATER shot of this config with its THz1, plus the film's measured centre dose."""
    cond = pd.read_csv(project / "Cam_08_23" / "08_23_conditions").dropna(subset=["Film"])
    shots = cond[cond["ScatFlag"].str.startswith("ININWATER") & (cond["Quads"] == quads) & (cond["Col"] == col)].copy()
    shots["depth_mm"] = shots["ScatFlag"].str.extract(r"ININWATER([+-]\d+)")[0].astype(int) + 24
    film = pd.read_csv(project / "Cam_08_23" / "water_depth_dose.csv")
    return shots.merge(film[["film", "centre"]].rename(columns={"film": "Film", "centre": "film_centre_Gy"}),
                       on="Film", how="left")


film_shots = load_film_shots()


def plot_profiles(film_mean):
    """Overlay central x/y profiles (2 mm strips) of every analysed variant at 20 mm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for (v, d), (xc, yc, dmap) in profiles.items():
        if d != 20:
            continue
        # centre on the dose-weighted centroid of the high-dose region rather than a single noisy max
        w = np.where(dmap > 0.5 * np.nanmax(dmap), dmap, 0)
        iy = int(round((w.sum(axis=1) * np.arange(w.shape[0])).sum() / w.sum()))
        ix = int(round((w.sum(axis=0) * np.arange(w.shape[1])).sum() / w.sum()))
        half = int(round(1.0 / np.mean(np.diff(xc))))  # +-1 mm strip
        axes[0].plot(xc - xc[ix], dmap[iy - half:iy + half + 1, :].mean(axis=0), label=v)
        axes[1].plot(yc - yc[iy], dmap[:, ix - half:ix + half + 1].mean(axis=1), label=v)
    for ax, lab in zip(axes, ["x (mm)", "y (mm)"]):
        if 20 in film_mean.index:
            ax.axhline(film_mean.loc[20, "mean"], color="k", ls="--", lw=1, label="film centre (mean)")
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Dose per THz1 charge (Gy/nC), 20 mm depth")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Q{quads} collimated: central profiles per check")
    fig.savefig(out_dir / "profiles_d20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse(variant, csv, depth, z_bins):
    r = BinnedResult(str(csv))
    raw = r.data["Sum"]
    x = r.dimensions[0].get_bin_centers() * 10
    y = r.dimensions[1].get_bin_centers() * 10
    rows = []
    depths = [20] if z_bins == 1 else film_depths
    for d in depths:
        sl = raw[:, :, 0] if z_bins == 1 else raw[:, :, z_bins - 1 - d]
        centre, Px, Py, dmap = dose_per_nC(sl, x, y)
        rows.append(dict(variant=variant, depth_mm=d, sim_Gy_per_nC=centre, P_x=Px, P_y=Py))
        profiles[(variant, d)] = (x - x.mean(), y - y.mean(), dmap)

        # one dose map per matching film shot, scaled by that shot's THz1, as in cam_data.ipynb
        fig_dir = out_dir / "dose_maps" / variant
        fig_dir.mkdir(parents=True, exist_ok=True)
        for _, f in film_shots[film_shots["depth_mm"] == d].iterrows():
            xs, ys, doseMap = getDosemap(str(csv), n_particles, dose_depth=d, outputFileName=variant,
                                         acChargenC=f["THz1"], plot=False,
                                         z_index=None if z_bins == 1 else z_bins - 1 - d)
            res = plot_dose1(doseMap, "RCF", xs, ys, strip_width=2)
            res[0].suptitle(f"{variant}: {f['Film']} (THz1 = {f['THz1']} nC), {d} mm")
            res[0].savefig(fig_dir / f"{f['Film']}_d{d}_dose_map.png", dpi=200, bbox_inches="tight")
            plt.close(res[0])
            film_rows.append(dict(variant=variant, film=f["Film"], depth_mm=d, THz1_nC=f["THz1"],
                                  sim_centre_Gy=res[3], film_centre_Gy=f["film_centre_Gy"],
                                  film_over_sim=f["film_centre_Gy"] / res[3]))
    if z_bins > 1:
        # central-axis depth dose (r < 2 mm) to confirm the z orientation
        xx, yy = np.meshgrid(x - x.mean(), y - y.mean(), indexing="ij")
        m = xx**2 + yy**2 <= 4
        pdd = raw[m].mean(axis=0)[::-1] / (n_particles * 1.60217663e-10)   # index = depth in mm
        np.savetxt(out_dir / f"pdd_{variant}.txt", pdd)
    return rows


def film_reference():
    cond = pd.read_csv(project / "Cam_08_23" / "08_23_conditions").dropna(subset=["Film"])
    film = pd.read_csv(project / "Cam_08_23" / "water_depth_dose.csv")
    f = film.merge(cond[["Film", "THz1", "Quads"]], left_on="film", right_on="Film")
    f = f[(f["Quads"] == quads) & (f["col"] == col)]
    f["film_Gy_per_nC"] = f["centre"] / f["THz1"]
    return f[["film", "type", "depth_mm", "THz1", "centre", "film_Gy_per_nC"]]


if __name__ == "__main__":
    variants = sys.argv[1:] or ["V0_old", "V1_fixed", "V2_noYAG", "V3_col", "V4_dist"]
    os.chdir(out_dir)  # TOPAS writes its output into the working directory
    track_beam()
    results = []
    for v in variants:
        setup, csv, depth, z_bins = build(v)
        if csv.exists() and csv.stat().st_size > 0:  # TOPAS creates the file empty at run start
            setup.file.close()
            print(f"{v}: reusing {csv.name}")
        else:
            print(f"{v}: running TOPAS", flush=True)
            setup.run_topas()
            if not csv.exists() or csv.stat().st_size == 0:
                sys.exit(f"{v}: TOPAS produced no {csv.name}")
        results += analyse(v, csv, depth, z_bins)

    films = film_reference()
    print("\nFilms (Q%d, collimated):" % quads)
    print(films.to_string(index=False))
    film_mean = films.groupby("depth_mm")["film_Gy_per_nC"].agg(["mean", "std", "count"])

    plot_profiles(film_mean)
    res = pd.DataFrame(results)
    res["film_Gy_per_nC"] = res["depth_mm"].map(film_mean["mean"])
    res["film/sim"] = res["film_Gy_per_nC"] / res["sim_Gy_per_nC"]
    print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    per_film = pd.DataFrame(film_rows)
    print("\nPer film shot (sim scaled by that shot's THz1):")
    print(per_film.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    per_film_csv = out_dir / "per_film.csv"
    if per_film_csv.exists():
        per_film = pd.concat([pd.read_csv(per_film_csv), per_film]).drop_duplicates(["variant", "film"], keep="last")
    per_film.to_csv(per_film_csv, index=False)
    summary = out_dir / "summary.csv"
    if summary.exists():
        res = pd.concat([pd.read_csv(summary), res]).drop_duplicates(["variant", "depth_mm"], keep="last")
    res.to_csv(summary, index=False)
