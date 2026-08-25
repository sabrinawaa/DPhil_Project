"""
Overlay film (RCF) and simulated dose slices at a matching depth.

Loads a film TIFF the same way CLEAR_RCF_processing.ipynb does (background
subtraction -> dose_CLEAR conversion -> marking removal), loads the matching
simulated depth slice the same way simulate_PDD_CLEAR.ipynb does (note: the
sim depths array is stored back-to-front, i.e. depths[i] pairs with
z_index=i via the same [::-1] reversal used there), fits both with the same
SuperGaussian/2-Gaussian slice fits as plot_dose1, and plots them overlaid
so the shapes and magnitudes can be compared directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from topas2numpy import BinnedResult

from YAG_RCF_analysis import get_dose_slices
from topasToDose import getDosemap
from uniformity_fit import supergaussian1D_skewed
from flatness import sum_2gaussians_skewed


def get_bin_edges(dim):
    return np.linspace(0, dim.n_bins * dim.bin_width, dim.n_bins + 1)


# ---- film-side helpers, copied from CLEAR_RCF_processing.ipynb (not in any shared module) ----

def dose_CLEAR(PV, nOD0, channel):
    if channel == 0:
        a, b, c = 4.6093, 0.1574, 4.6606
    elif channel == 1:
        a, b, c = 7.9162, 0.0418, 7.9213
    elif channel == 2:
        a, b, c = 23.5294, 0.1614, 23.2037

    OD = -np.log(PV / 65535)
    nOD = OD - nOD0
    exp_term = np.exp(-nOD)
    return (a - c * exp_term) / (exp_term - b)


def get_calibration(im_type):
    if im_type == "RCF":
        return 0.0846667  # mm/pixel


def remove_marking(film):
    h, w = film.shape
    mask = np.ones((h, w), dtype=bool)
    mask[50:-50, 50:-50] = False
    film[(mask) & (film < 30000)] = 40000
    return film


def load_film_dosemap(tif_path, channel, OD0, crop=(25, -25, 25, -25), clean_marking=True, dose_clip=25, dose_clip_to=4):
    film = tiff.imread(tif_path)[crop[0]:crop[1], crop[2]:crop[3], channel]
    if clean_marking:
        film = remove_marking(film)
    film_flipped = np.flipud(film)

    h, w = film.shape
    pixel_calibration = get_calibration("RCF")
    x = np.arange(w) * pixel_calibration
    y = np.arange(h) * pixel_calibration

    dose = dose_CLEAR(film_flipped, OD0, channel)
    dose[dose > dose_clip] = dose_clip_to
    return x, y, dose


def load_sim_dosemap(dose_file, n_particles, output_filename, acChargenC, target_depth_mm):
    z_dim = BinnedResult(dose_file).dimensions[2]
    depths_mm = (get_bin_edges(z_dim)[1:] * 10)[::-1]  # same reversal as simulate_PDD_CLEAR.ipynb
    z_index = int(np.argmin(np.abs(depths_mm - target_depth_mm)))
    matched_depth = float(depths_mm[z_index])

    x, y, doseMap = getDosemap(
        dose_file, n_particles, dose_depth=matched_depth,
        outputFileName=output_filename, acChargenC=acChargenC,
        plot=False, z_index=z_index,
    )
    return x, y, doseMap, matched_depth


def plot_slice_overlay(film_slices, sim_slices, depth_mm, film_label="Film", sim_label="Simulation"):
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Film vs simulation slices at depth = {depth_mm:.1f} mm")

    for ax, coord_key, slice_key, params_key, params2g_key, curve_fn, xlabel in [
        (ax_x, "new_x", "slice_row", "params_x", "params_xx", supergaussian1D_skewed, "X (mm)"),
        (ax_y, "new_y", "slice_col", "params_y", "params_yy", supergaussian1D_skewed, "Y (mm)"),
    ]:
        for slices, label, data_color, fit_color in [
            (film_slices, film_label, "tab:orange", "darkred"),
            (sim_slices, sim_label, "tab:blue", "navy"),
        ]:
            coord = slices[coord_key]
            data = slices[slice_key]
            ax.plot(coord, data, "o", color=data_color, alpha=0.5, markersize=3, label=f"{label} data")

            params = slices[params_key]
            fit_curve = curve_fn(coord, *params)
            ax.plot(coord, fit_curve, "-", color=fit_color, linewidth=1.8,
                     label=f"{label} SuperGaussian fit (P={params[3]:.2f})")

            params_2g = slices[params2g_key]
            if params_2g is not None:
                ax.plot(coord, sum_2gaussians_skewed(coord, *params_2g), "--", color=fit_color, linewidth=1.2,
                         label=f"{label} 2-Gaussian fit")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Dose (Gy)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # ---- film config (mirrors CLEAR_RCF_processing.ipynb) ----
    channel = 0
    film_dir = "CLEAR_experiments/May2026/scatterer_15-05_2026_21h/"
    bkgd1 = tiff.imread(film_dir + "G_025.tif")[50:-25, 25:-25, channel]
    bkgd2 = tiff.imread(film_dir + "G_026.tif")[50:-25, 25:-25, channel]
    OD0 = -np.log(np.mean([bkgd1, bkgd2]) / 65535)

    film_filename = "G_008"
    film_depth_mm = 20.0  # known depth for this film, e.g. from depth_map in PDD_CLEAR.ipynb

    # ---- sim config (mirrors simulate_PDD_CLEAR.ipynb) ----
    dose_file = "CLEAR_experiments/May2026/DoseAtTank256_CLEAR_dual_scatterer_0515_small_YAG_875_full.csv"
    output_filename = "CLEAR_dual_scatterer_0515_small_YAG_875_full"
    chargenC = 10.31
    n_particles = int(3e6)

    # ---- load both ----
    fx, fy, film_dose = load_film_dosemap(film_dir + film_filename + ".tif", channel, OD0)
    sx, sy, sim_dose, matched_depth = load_sim_dosemap(dose_file, n_particles, output_filename, chargenC, film_depth_mm)

    print(f"Film {film_filename} (depth {film_depth_mm} mm) vs sim z-slice at matched depth {matched_depth:.1f} mm")

    film_slices = get_dose_slices(film_dose, fx, fy, strip_width=2)
    sim_slices = get_dose_slices(sim_dose, sx, sy, strip_width=2)

    fig = plot_slice_overlay(film_slices, sim_slices, matched_depth)
    plt.show()
