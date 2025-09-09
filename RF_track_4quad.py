import RF_Track as RFT
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize


quad_length=0.3,  # m
drift_l=0.2,
aperture=0.019 

def gaussian(x,A,mu,sig):
    return A * np.exp(- (x-mu)**2 /(2*sig**2))



def four_quads(Lquad, k11, k12, k13, k14, Ldrift, N_particles, Energy ,L_drift_after=0, plot=True, saveparams=True):
    Q1 =RFT.Quadrupole(Lquad, -Energy, k11) 
    Q2 = RFT.Quadrupole(Lquad, -Energy, k12)
    Q3 = RFT.Quadrupole(Lquad, -Energy, k13)
    Q4 = RFT.Quadrupole(Lquad, -Energy, k14)
    Drift = RFT.Drift(Ldrift)
    Drift_after = RFT.Drift(L_drift_after) if L_drift_after > 0 else None
    for i in [Q1, Q2, Q3, Q4, Drift]:
        i.set_aperture(aperture,aperture,'circular')  # m
    Drift.set_tt_nsteps(50)  

    #lattice
    lattice = RFT.Lattice()
    lattice.append(Q1)
    lattice.append(Drift)
    lattice.append(Q2)
    lattice.append(Drift)
    lattice.append(Q3)
    lattice.append(Drift)
    lattice.append(Q4)
    lattice.append(Drift)
    if Drift_after is not None:
        lattice.append(Drift_after)

    # ttable = lattice.get_transport_table("%beta_x %beta_y")
    # beta_x, beta_y = ttable[:, 0], ttable[:, 1]



    N_particles = int(N_particles)
    charge = -1
    Q = np.full(N_particles,charge)
    mass = RFT.electronmass
    rel_gamma = Energy/mass
    rel_beta = np.sqrt(1-1/(rel_gamma**2))

    MASS = np.full(N_particles,mass)
    sigma_x, sigma_y = 1,1
    sigma_xp, sigma_yp = 1,1
    Pref = Energy
    '''
    matrix (np.ndarray): Phase space matrix of shape (n, 6).
                Can have n rows, representing n macroparticles.
                Must be exactly 6 columns, representing:
                    Column 0: Horizontal coordinate X [mm]
                    Column 1: Horizontal angle PX [mrad]
                    Column 2: Vertical coordinate Y [mm]
                    Column 3: Vertical angle PY [mrad]
                    Column 4: Arrival time T [mm/c]
                    Column 5: Total momenta P [MeV/c]
    '''

    x = np.random.normal(0, sigma_x, N_particles)
    xp = np.random.normal(0, sigma_xp, N_particles)
    y = np.random.normal(0, sigma_y, N_particles)
    yp = np.random.normal(0, sigma_yp, N_particles)
    P = Energy * (1 + np.random.normal(0, 0.005, N_particles))  # 200 MeV ± 0.5%
    T = np.zeros(N_particles)
    matrix = np.column_stack((x, xp, y, yp, T, P)) #transpose to match Bunch6d format

    # Twiss = RFT.Bunch6d_twiss() #maybe need to set emittance?
    # Twiss.emitt_x = 10 #mm mrad normalised
    # Twiss.emitt_y = 10
    # geo_emm = Twiss.emitt_x / (rel_beta * rel_gamma)

    # Twiss.beta_x = 1/geo_emm #m to get 1 mm sigma x
    # Twiss.beta_y = 1/geo_emm
    # Twiss.alpha_x = 0.0
    # Twiss.alpha_y = 0.0 #at symmetry points (inside quads)
    # bunch = RFT.Bunch6d(mass, N_particles, charge, Pref, Twiss, N_particles)
    bunch = RFT.Bunch6d(mass, N_particles, charge, matrix )
    
    B1 = lattice.track(bunch)
    T = lattice.get_transport_table('%S %beta_x %beta_y %alpha_x %alpha_y')
    # print(B1.get_info().beta_x)
    M = B1.get_phase_space('%x %xp %y %yp %E %z')

    if plot:
        # Make plots
        plt.figure(1)
        plt.plot(T[:,0], T[:,1], 'b-', label=r'$\beta_x$')
        plt.plot(T[:,0], T[:,2], 'r-', label=r'$\beta_y$')
        plt.plot(T[:,0], T[:,3], 'g-', label=r'$\alpha_x$')
        plt.plot(T[:,0], T[:,4], 'b-', label=r'$\alpha_y$')
        plt.legend()
        plt.xlabel('S [m]')
        plt.ylabel(r'$\beta$ [m]')

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

        fig.savefig(f"Output_figs/RFT_k1s={k11}_{k12}_{k13}_{k14}.png")
    if saveparams:
        np.savetxt(f"RFT_k1s={k11}_{k12}_{k13}_{k14}_N={N_particles}.txt",M) 
    
    return M
'''
def optimize_four_quads(N): #N for number of random initialisations
    def merit_function(params):
       
        sx = T[-1, 0]
        sy = T[-1, 1]
        ax = T[-1, 2]
        ay = T[-1, 3]

        # sGoal is the desired beamsize taken at the measurement point
        # with finalDrift=True, the measurement point is at the end of the total
        # beamline with the scatterers removed (but including the 2.5m drift where they would be placed)
        # sGoal=75 thus would not be expected to require further magnification from the scatterers
        # so sGoal=75 may be useful for looking for a solution with no S1
        sGoal = 40

        # M is the merit function to be minimised (we want each term in the sum to end up small)
        # (sx/sGoal-1)**2 is minimised when the x beamsize=sGoal (same with y)
        # np.exp(ax) is minimised when alpha x is negative after the lattice (the beam is diverging rather than converging)
        # (ax/ay - 1)**2 is minimised when ax=ay, so the divergence is symmeterical

        M = 1000*(sx/sGoal-1)**2+1000*(sy/sGoal-1)**2 + \
            + np.exp(ax) + 10000*(ax/ay-1)**2

        return M

     # initialise randomiser
    rng = np.random.default_rng()
    minResult = np.inf
    # loop to allow random initialisations of quad values in optimiser
    for i in range(N):
        # randomised 4 quad input, within k1 limits
        opt_input = rng.uniform(low=-75, high=75, size=4)
        # scipy.optimize.minimise implemented to minimise four_quads_merit from randomised quad input with finalDrift=True
        # run with 1000 particles for the sake of time (each minimiser runs is running four_quads_merit thousands of times
        res = minimize(merit_function, x0=opt_input, args=(0.3, 0.2, 1000, 200, True), bounds=(
            (-75, 75), (-75, 75), (-75, 75), (-75, 75)), method='Nelder-Mead')
        print('M = '+str(res.fun)+', X0='+str(res.x))
        # after each minimisation in loop, check to see if this is the best solution and update minResul
        if res.fun < minResult:
            minResult = res.fun
            minInput = res.x
    # print best result and run four_quads using these strengths to investigate evolution of twiss parameters and beamsize
    print('Optimization Result: X=')
    print(minInput)
    four_quads(minInput, 0.3, 0.2, 100000, 200, True)
    ...


'''