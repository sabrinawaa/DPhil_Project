import RF_Track as RFT
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils

class RF_track_utils():
    # set filepaths and number of threads available
    # input file is the topas script being generated and run here
    # output file is 'patient_beam'
    def __init__(
        self,
        energy,
        k1s,
        
    ):
        #set constants
        quad_length = float(0.3)  # m
        drift_length = float(0.2)
        aperture = float(0.019)    # m

        self.energy = float(energy)
        self.k1s = k1s

        RFT.cvar.number_of_threads = 8
        self.lattice = RFT.Lattice()
        
        Drift = RFT.Drift(drift_length)
        Drift.set_aperture(aperture, aperture, 'circular')  # set aperture for drift PUT BACK!!!!!!!!!!!!!!!!!!!!
        for k1 in k1s:
            quad = RFT.Quadrupole(quad_length, -self.energy, k1)
            quad.set_aperture(aperture, aperture, 'circular')  # set aperture for quadrupole.  PUT BACK!!!!!!!!!!!!!!!!!!!!
            self.lattice.append(quad)
            self.lattice.append(Drift)
        Drift.set_tt_nsteps(50)  # set number of steps for transport table

    
        
    # define Gaussian beam in terms of phase space parameters
    # sigma, sigmap, E in mm, mrad, MeV respectively


#rewrite into class structure
# '''
    def gaussian(x,A,mu,sig):
        return A * np.exp(- (x-mu)**2 /(2*sig**2))

    def add_quadrupole(self, Lquad, k1):
        self.lattice.append(RFT.Quadrupole(Lquad, self.energy, k1))
    def add_drift(self, Ldrift):
        Drift = RFT.Drift(Ldrift)
        self.lattice.append(Drift)
        Drift.set_tt_nsteps(50) 

    def set_aperture(self):
        for element in self.lattice:
            element.set_aperture(self.aperture, self.aperture, 'circular')


    def track_bunch(self,N_particles, saveparams = True, E_deviation=0.5, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1):
        N_particles = int(N_particles)
        mass = RFT.electronmass
        charge = -1
        x = np.random.normal(0, sigma_x, N_particles)
        xp = np.random.normal(0, sigma_xp, N_particles)
        y = np.random.normal(0, sigma_y, N_particles)
        yp = np.random.normal(0, sigma_yp, N_particles)
        P = self.energy * (1 + np.random.normal(0, E_deviation/100, N_particles))  # 200 MeV ± 0.5%
        T = np.zeros(N_particles)
        matrix = np.column_stack((x, xp, y, yp, T, P)) #transpose to match Bunch6d format

        bunch = RFT.Bunch6d(mass, N_particles, charge, matrix )
        self.trackedBunch = self.lattice.track(bunch)
        self.transport_table = self.lattice.get_transport_table(
        '%S %beta_x %beta_y %alpha_x %alpha_y %sigma_x %sigma_y %sigma_px %sigma_py')
        self.phsp = self.trackedBunch.get_phase_space('%x %xp %y %yp %E %z')
        if saveparams:
            k1s_str = '_'.join(str(k) for k in self.k1s)  # Convert each element to string and join with '_'
            np.savetxt(f"RFT_k1s={k1s_str}_N={int(N_particles )}.txt", self.phsp)

        return self.phsp
    
    # def track_bunch(self, saveparams=True):
    #     B1 = self.lattice.track(self.bunch)
        
       
        
    
    def plot_phsp(self):
        T = self.transport_table
        M = self.phsp
        # plt.figure(1)
        # plt.plot(T[:,0], T[:,1], 'b-', label=r'$\beta_x$')
        # plt.plot(T[:,0], T[:,2], 'r-', label=r'$\beta_y$')
        # plt.plot(T[:,0], T[:,3], 'g-', label=r'$\alpha_x$')
        # plt.plot(T[:,0], T[:,4], 'b-', label=r'$\alpha_y$')
        # plt.legend()
        # plt.xlabel('S [m]')
        # plt.ylabel(r'$\beta$ [m]')

        def scatter_hist(x, y, ax, ax_histx, ax_histy):

            ax_histx.tick_params(axis="x", labelbottom=True)
            ax_histy.tick_params(axis="y", labelleft=True)

            # the scatter plot:
            ax.scatter(x, y,s=5)

            ax.set_xlabel('X (mm)')  # Set x-axis label for scatter plot
            ax.set_ylabel('Y (mm)')

            # now determine nice limits by hand:
            binwidth = 0.25
            xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
            lim = (int(xymax/binwidth) + 1) * binwidth

            bins = np.arange(-lim, lim + binwidth, binwidth)
            # ax_histx.hist(x, bins=bins)
            # ax_histy.hist(y, bins=bins, orientation='horizontal')
            n_x, bins_x, _ = ax_histx.hist(x, bins=bins, alpha=0.6)
            n_y, bins_y, _ = ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.6)

            mu_x, sigma_x = np.mean(x), np.std(x)
            mu_y, sigma_y = np.mean(y), np.std(y)
            
            # Scale factor to match histogram counts
            scale_x = len(x) * binwidth
            scale_y = len(y) * binwidth
            
            # Plot Gaussian fits
            pdf_x = norm.pdf(bins_x, mu_x, sigma_x) * scale_x
            ax_histx.plot(bins_x, pdf_x, 'k-', linewidth=2, label= f"sig_x = {sigma_x:.2f}")
            ax_histx.legend()
            
            pdf_y = norm.pdf(bins_y, mu_y, sigma_y) * scale_y
            ax_histy.plot(pdf_y, bins_y, 'k-', linewidth=2, label= f"sig_y = {sigma_y:.2f}")
            ax_histy.legend()

        fig, axs = plt.subplot_mosaic([['histx', '.'],
                                    ['scatter', 'histy']],
                                    figsize=(6, 6),
                                    width_ratios=(4, 1), height_ratios=(1, 4),
                                    layout='constrained')
        scatter_hist(M[:,0], M[:,2], axs['scatter'], axs['histx'], axs['histy'])

        k1s_str = '_'.join(str(k) for k in self.k1s)
        fig.savefig(f"Output_figs/RFT_k1s={k1s_str}.png")


