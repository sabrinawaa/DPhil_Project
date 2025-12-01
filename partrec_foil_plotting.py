import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uniformity_fit import supergaussian1D, r90
from scipy.optimize import curve_fit


class partrec_foil_plotting:
    # import phase space from defined file
    def __init__(self, path_to_phsp_file):
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        # read Topas ASCII output file
        phase_space = pd.read_csv(
            path_to_phsp_file,
            names=["X", "Y", "Z", "PX", "PY", "E", "Weight", "PDG", "9", "10"],
            delim_whitespace=True,
        )
        phase_space["X"] = phase_space["X"] * 10
        phase_space["Y"] = phase_space["Y"] * 10
        # add "R" column for radial distance from origin in cm
        phase_space["R"] = np.sqrt(
            np.square(phase_space["X"]) + np.square(phase_space["Y"])
        )

        gamma_phase_space = phase_space.copy()
        # create DataFrame containing only electron data at patient
        electron_phase_space = phase_space.drop(
            phase_space[phase_space["PDG"] != 11].index
        )
        # create DataFrame containing only gamma data at patient
        gamma_phase_space = gamma_phase_space.drop(
            phase_space[phase_space["PDG"] != 22].index
        )
        phsp_dict = {
            "all": phase_space,
            "e": electron_phase_space,
            "y": gamma_phase_space,
        }

        self.phsp_dict = phsp_dict
    # show transverse beam profile and energy spectrum
    # fov is field of view of profile graphs
    # col is the virtual collimator radius for calculation of transmission
    # both in millimetres
    # mean energy is calculated within collimator radius only

    def show_transverse_beam(self, out_filename, s1_depth, s2_depth, s2_r,
                          particle='e', fov=50, col=50):

        n_bins = 50

        def get_slices(phsp, slice_width=3):
            phsp_xslice = phsp[(phsp["Y"].abs() < slice_width)]
            phsp_yslice = phsp[(phsp["X"].abs() < slice_width)]
            return phsp_xslice, phsp_yslice

        try:
            phsp = self.phsp_dict[particle]
        except KeyError:
            print('Beam type not found - should be "all", "e", or "y"')
            return

        phsp_xslice, phsp_yslice = get_slices(phsp)

        # Prepare histograms
        hist_x, edges_x = np.histogram(phsp_xslice["X"], bins=n_bins, range=[-fov, fov])
        hist_y, edges_y = np.histogram(phsp_yslice["Y"], bins=n_bins, range=[-fov, fov])
        centers_x = (edges_x[:-1] + edges_x[1:]) / 2
        centers_y = (edges_y[:-1] + edges_y[1:]) / 2

        # Fit SuperGaussian
        p0 = [np.max(hist_x), np.mean(phsp_xslice["X"]), np.std(phsp_xslice["X"]), 4]
        params_x, _ = curve_fit(supergaussian1D, centers_x, hist_x, p0=p0)
        params_y, _ = curve_fit(supergaussian1D, centers_y, hist_y, p0=p0)

        xy_fit_curve = np.linspace(-fov, fov, 500)

        sig_x, sig_y = params_x[2], params_y[2]
        P_x, P_y = params_x[3], params_y[3]
        r90_x, r90_y = r90(sig_x, P_x), r90(sig_y, P_y)

        # --- FIGURE LAYOUT (scatter + hist X + hist Y) ---
        fig, axs = plt.subplot_mosaic(
            [['histx', '.'],
            ['scatter', 'histy']],
            figsize=(10, 8),
            width_ratios=(4, 1),
            height_ratios=(1, 4),
            layout='constrained'
        )

        ax_scatter = axs['scatter']
        ax_histx   = axs['histx']
        ax_histy   = axs['histy']

        # ------------------- SCATTER PLOT --------------------
        ax_scatter.scatter(phsp["X"], phsp["Y"], s=1, alpha=0.4)
        ax_scatter.set_xlim(-fov, fov)
        ax_scatter.set_ylim(-fov, fov)
        ax_scatter.set_xlabel("X [mm]")
        ax_scatter.set_ylabel("Y [mm]")
        ax_scatter.set_title("Beam Profile (X–Y Scatter)")

        # ------------------- X HIST -------------------
        ax_histx.hist(phsp_xslice["X"], bins=n_bins, range=[-fov, fov],
                    color="b", alpha=0.6)
        ax_histx.plot(xy_fit_curve, supergaussian1D(xy_fit_curve, *params_x),
                    'r-', label=f"P={P_x:.2f}, r90={r90_x:.2f}")
        ax_histx.set_ylabel("Counts")
        ax_histx.legend()

        # ------------------- Y HIST -------------------
        ax_histy.hist(phsp_yslice["Y"], bins=n_bins, range=[-fov, fov],
                    orientation="horizontal",
                    color="b", alpha=0.6)
        ax_histy.plot(supergaussian1D(xy_fit_curve, *params_y), xy_fit_curve,
                    'r-', label=f"P={P_y:.2f}, r90={r90_y:.2f}")
        ax_histy.set_xlabel("Counts")
        ax_histy.legend()

        # ------------------- SAVE FIG -------------------
        fig.savefig(
            f"Output_figs/{out_filename}_s1depth={s1_depth}"
            f"s2depth={s2_depth}s2r={s2_r}{particle}_scatter_hist.png"
        )

        # ------------------- PRINT OUTPUTS -------------------
        col_phsp = phsp[(phsp.R < col)].dropna()

        if particle == 'e':
            print("Electron number inside r90 =", len(col_phsp["X"]))
            print("Electron Transmission =", len(col_phsp["X"]) / len(phsp["X"]) * 100, "%")
            print("Mean Energy at Dump:", np.mean(col_phsp["E"]))
            print("Energy Spread at Dump:",
                (np.max(col_phsp["E"]) - np.min(col_phsp["E"])) /
                np.mean(col_phsp["E"]) * 100, "%")

        if particle == 'y':
            print("Photon number inside r90 =", len(col_phsp["X"]))
            print("Energy Spread at Dump:",
                (np.max(col_phsp["E"]) - np.min(col_phsp["E"])) /
                np.mean(col_phsp["E"]) * 100, "%")
