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

    def show_transverse_beam(self, out_filename, s1_depth, s2_depth, s2_r, particle='e', fov=50, col=50):
        n_bins = 50

        def get_slices(phsp, slice_width=3): #was 1
            #Instead of plotting all particles in 2D (which can be slow),
            #  extracts a thin 1D slice around x,y=0 to represent the beam profile.
            phsp_xslice = phsp[(phsp["Y"] < slice_width)]
            phsp_xslice = phsp_xslice[(phsp_xslice["Y"] > -slice_width)]
            phsp_yslice = phsp[(phsp["X"] < slice_width)]
            phsp_yslice = phsp_yslice[(phsp_yslice["X"] > -slice_width)]
            print(max(phsp_xslice),max(phsp_yslice))
            return phsp_xslice, phsp_yslice

        try:
            phsp = self.phsp_dict[particle]
        except KeyError:
            print('Beam type not found - should be one of "all", "e", "y"')
        phsp_xslice, phsp_yslice = get_slices(phsp)

        hist_x, bin_edges_x = np.histogram(phsp_xslice["X"], bins=n_bins, range=[-fov, fov])
        bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        p0=[np.max(hist_x),  np.mean(phsp_xslice["X"]), np.std(phsp_xslice["X"]), 4]
        #curvefit 2d histogram to supergaussian
        # Perform the optimization
        params_x, _ = curve_fit(supergaussian1D, bin_centers_x, hist_x, p0=p0)

        hist_y, bin_edges_y = np.histogram(phsp_yslice["Y"], bins=n_bins, range=[-fov, fov])
        bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

        params_y, _ = curve_fit(supergaussian1D, bin_centers_y, hist_y, p0=p0)
        xy_fit_curve = np.linspace(-fov, fov, 500)
       
        sig_x,sig_y, P_x, P_y = params_x[2], params_y[2], params_x[3], params_y[3]
        r90_x, r90_y = r90(sig_x, P_x), r90(sig_y, P_y)

        font_size = 18
        plt.rcParams.update({
            'font.size': font_size,          # Default font size for titles, labels, etc.
            'axes.titlesize': font_size,     # Axis title size
            'axes.labelsize': font_size,     # Axis labels (x and y)
            'xtick.labelsize': font_size,    # X-axis tick labels
            'ytick.labelsize': font_size,    # Y-axis tick labels
        })

        fig, ax = plt.subplots(1, 2, figsize=(10, 6)) 
        ax[0].hist(phsp_xslice["X"], bins=n_bins, range=[-fov, fov], color="b",alpha=0.6,label= particle+ ' X-Intensity')
        ax[0].plot(xy_fit_curve, supergaussian1D(xy_fit_curve, *params_x), 'r-', label=f"SuperGaussian Fit (P={params_x[3]:.2f})")
        ax[0].set_xlabel("X [mm]")
        ax[0].set_ylabel("N")
        ax[0].set_title(f"X fit sigma = {sig_x:.2f}, r90 = {r90_x:.2f}")
        ax[0].legend()
        ax[1].hist(phsp_yslice["Y"], bins=n_bins, range=[-fov, fov], color="b",alpha=0.6,label= particle+ ' Y-Intensity')
        ax[1].plot(xy_fit_curve, supergaussian1D(xy_fit_curve, *params_y), 'r-', label=f"SuperGaussian Fit (P={params_y[3]:.2f})")
        ax[1].set_xlabel("Y [mm]")
        ax[1].set_ylabel("N")
        ax[1].set_title(f"Y fit sigma = {sig_y:.2f}, r90 = {r90_y:.2f}")
        ax[1].legend()
        # fig1, ax1 = plt.subplots(1, 2, figsize=(10,5))
        # ax1[0].hist2d(
        #     phsp["X"], phsp["Y"], bins=100, range=[[-fov, fov], [-fov, fov]], cmap="jet"
        # )
        # ax1[0].set_xlabel("X [mm]")
        # ax1[0].set_ylabel("Y [mm]")
        # ax1[0].set_title("XY Distribution")
        # ax1[1].hist(phsp["E"], bins=100, color="k")
        # ax1[1].set_xlabel("E [MeV]")
        # ax1[1].set_ylabel("N")
        # ax1[1].set_yscale('log')
        # ax1[1].set_title("Energy Spectrum")

        col_phsp = phsp[(phsp.R < col)]
        col_phsp = col_phsp.dropna()

       
        

        fig.savefig("Output_figs/"+out_filename+"_s1depth="+str(s1_depth)+"s2depth=" + str(s2_depth) + "s2r=" + str(s2_r) + particle+ "Intensity.png")
        #fig1.savefig("Output_figs/s2depth=" + str(s2_depth) + "s2r=" + str(s2_r) +"XY_Energy.png")
        #plt.show()

        if particle == 'e':
            print("Electron number inside r90= "+ str(len(col_phsp["X"])))
            print(
                "Electron Transmission  = " +
                str(len(col_phsp["X"]) / len(phsp["X"]) * 100) +" %"
            )
            print("Mean Energy at Dump:" + str(np.mean(col_phsp["E"])))
            print("Energy Spread at Dump:" + str((np.max(col_phsp["E"])-np.min(col_phsp["E"]))/np.mean(col_phsp["E"])*100)+" %")

        if particle == 'y':
            print("Photon number inside r90= "+ str(len(col_phsp["X"])))
            print("Energy Spread at Dump:" + str((np.max(col_phsp["E"])-np.min(col_phsp["E"]))/np.mean(col_phsp["E"])*100)+" %")
#
