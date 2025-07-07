import RF_Track as RFT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

RFT.cvar.number_of_threads = 8
quad_length = 0.3
q1k1 = 0
q2k1 = 0
q3k1 = 0
q4k1 = 0
drift_l = 0.2


def four_quads(strengths, Lquad, Ldrift, N_particles, Energy, finalDrift=False):

    Q1 = RFT.Quadrupole(Lquad, -200, strengths[0])
    Q2 = RFT.Quadrupole(Lquad, -200, strengths[1])
    Q3 = RFT.Quadrupole(Lquad, -200, strengths[2])
    Q4 = RFT.Quadrupole(Lquad, -200, strengths[3])
    Drift = RFT.Drift(Ldrift)
    fDrift = RFT.Drift(2.5)
    Drift.set_tt_nsteps(50)

    # lattice
    lattice = RFT.Lattice()
    lattice.append(Q1)
    lattice.append(Drift)
    lattice.append(Q2)
    lattice.append(Drift)
    lattice.append(Q3)
    lattice.append(Drift)
    lattice.append(Q4)
    lattice.append(Drift)
    # this effectively adds the TOPAS component without the scatterers
    # useful for enforcing a certain beam without scatterers before introducing
    # them
    if finalDrift:
        lattice.append(fDrift)

    # ttable = lattice.get_transport_table("%beta_x %beta_y")
    # beta_x, beta_y = ttable[:, 0], ttable[:, 1]

    E = 200  # MeV
    #N_particles = int(N_particles)
    charge = -1
    Q = np.full(N_particles, charge)
    mass = RFT.electronmass
    rel_gamma = E/mass
    rel_beta = np.sqrt(1-1/(rel_gamma**2))

    MASS = np.full(N_particles, mass)
    sigma_x, sigma_y = 1, 1
    sigma_xp, sigma_yp = 1, 1
    Pref = Energy
    x = np.random.normal(0, sigma_x, N_particles)
    xp = np.random.normal(0, sigma_xp, N_particles)
    y = np.random.normal(0, sigma_y, N_particles)
    yp = np.random.normal(0, sigma_yp, N_particles)
    P = E * (1 + np.random.normal(0, 0.005, N_particles))  # 200 MeV ± 0.5%
    T = np.zeros(N_particles)

    Twiss = RFT.Bunch6d_twiss()  # maybe need to set emittance?
    Twiss.emitt_x = 10  # mm mrad normalised
    Twiss.emitt_y = 10
    geo_emm = Twiss.emitt_x / (rel_beta * rel_gamma)

    Twiss.beta_x = (1/geo_emm)  # m
    Twiss.beta_y = (1/geo_emm)

    Twiss.alpha_x = 0.0
    Twiss.alpha_y = 0.0  # at symmetry points (inside quads)

    bunch = RFT.Bunch6d(mass, N_particles, charge, np.array([ x, xp, y, yp, T, P ]) )
    # bunch= RFT.Bunch6d( np.array([ x, xp, y, yp, T, P,  MASS, Q, np.ones(N_particles) ]) )
    # bunch = RFT.Bunch6d(mass, 1, charge, Pref, Twiss, N_particles)
    B1 = lattice.track(bunch)
    T = lattice.get_transport_table(
        '%S %beta_x %beta_y %alpha_x %alpha_y %sigma_x %sigma_y %sigma_px %sigma_py')
    if finalDrift:
        print('Beam size at end of TOPAS geometry, without scatterers')
    else:
        print('Beam size at end of lattice:')
    print(T[-1, 5])
    print(T[-1, 6])
    # print(B1.get_info().beta_x)
    M = B1.get_phase_space('%x %xp %y %yp %E %z')
    # Make plots
    plt.figure(1)
    plt.plot(T[:, 0], T[:, 1], 'b-', label=r'$\beta_x$')
    plt.plot(T[:, 0], T[:, 2], 'r-', label=r'$\beta_y$')
    plt.plot(T[:, 0], T[:, 3], 'g-', label=r'$\alpha_x$')
    plt.plot(T[:, 0], T[:, 4], 'k-', label=r'$\alpha_y$')
    plt.legend()
    plt.xlabel('S [m]')
    plt.ylabel(r'$\beta$ [m]')

    plt.figure(2)
    plt.plot(T[:, 0], T[:, 5], 'b-', label=r'$\sigma_x$')
    plt.plot(T[:, 0], T[:, 6], 'r-', label=r'$\sigma_y$')

    plt.plot(T[:, 0], T[:, 7], 'g-', label=r'$\sigma_px$')
    plt.plot(T[:, 0], T[:, 8], 'k-', label=r'$\sigma_py$')
    plt.legend()
    plt.xlabel('S [m]')
    plt.ylabel(r'$\sigma$ [m]')

    def scatter_hist(x, y, ax, ax_histx, ax_histy):

        ax_histx.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=True)

        # the scatter plot:
        ax.scatter(x, y, s=5)

        ax.set_xlabel('X')  # Set x-axis label for scatter plot
        ax.set_ylabel('Y')

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    fig, axs = plt.subplot_mosaic([['histx', '.'],
                                   ['scatter', 'histy']],
                                  figsize=(6, 6),
                                  width_ratios=(4, 1), height_ratios=(1, 4),
                                  layout='constrained')
    scatter_hist(M[:, 0], M[:, 2], axs['scatter'],
                 axs['histx'], axs['histy'])
    # plt.show()

    return M


# NM optimiser to produce beams which symmetrically magnify the profile
def optimize_four_quads(N):
    # merit function for optimiser. Returns a value which is minimised when
    # the solution is optimal
    # this is just the typical RF-Track function Sabrina wrote, but with
    # plotting removed to improve runtime. Quad strengths are incuded in first function
    # argument as an array.
    def four_quads_merit(strengths, Lquad, Ldrift, N_particles, Energy, finalDrift):

        Q1 = RFT.Quadrupole(Lquad, -200, strengths[0])
        Q2 = RFT.Quadrupole(Lquad, -200, strengths[1])
        Q3 = RFT.Quadrupole(Lquad, -200, strengths[2])
        Q4 = RFT.Quadrupole(Lquad, -200, strengths[3])
        fDrift = RFT.Drift(2.5)
        Drift = RFT.Drift(Ldrift)
        Drift.set_tt_nsteps(50)

        # lattice
        lattice = RFT.Lattice()
        lattice.append(Q1)
        lattice.append(Drift)
        lattice.append(Q2)
        lattice.append(Drift)
        lattice.append(Q3)
        lattice.append(Drift)
        lattice.append(Q4)
        lattice.append(Drift)
        if finalDrift:
            lattice.append(fDrift)

        # ttable = lattice.get_transport_table("%beta_x %beta_y")
        # beta_x, beta_y = ttable[:, 0], ttable[:, 1]

        E = Energy  # MeV
        #N_particles = int(N_particles)
        charge = -1

        mass = RFT.electronmass
        rel_gamma = E/mass
        rel_beta = np.sqrt(1-1/(rel_gamma**2))

        Pref = Energy
      # 200 MeV ± 0.5%

        Twiss = RFT.Bunch6d_twiss()  # maybe need to set emittance?
        Twiss.emitt_x = 10  # mm mrad normalised
        Twiss.emitt_y = 10
        geo_emm = Twiss.emitt_x / (rel_beta * rel_gamma)

        Twiss.beta_x = (1/geo_emm)  # m
        Twiss.beta_y = (1/geo_emm)
        # Twiss.beta_x = 1 * (1 + np.sin(np.radians(mu/2))) / np.sin(np.radians(mu))  # m
        # Twiss.beta_y = 1 * (1 - np.sin(np.radians(mu/2))) / np.sin(np.radians(mu))
        Twiss.alpha_x = 0.0
        Twiss.alpha_y = 0.0  # at symmetry points (inside quads)

        # bunch = RFT.Bunch6d(mass, N_particles, charge, np.array([ x, xp, y, yp, T, P ]) )
        # bunch= RFT.Bunch6d( np.array([ x, xp, y, yp, T, P,  MASS, Q, np.ones(N_particles) ]) )
        bunch = RFT.Bunch6d(mass, 1, charge, Pref, Twiss, N_particles)
        B1 = lattice.track(bunch)
        # required terms taken from TT
        T = lattice.get_transport_table(
            '%sigma_x %sigma_y %alpha_x %alpha_y %sigma_px %sigma_py')

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
        res = minimize(four_quads_merit, x0=opt_input, args=(0.3, 0.2, 1000, 200, True), bounds=(
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


# run optimiser 20 times and print results. Small value of M is best! Under 0.1 is very good
# even with 1000 particles, each minimisation takes up to a minute, so running the loop 20 times like here may take 20 mins
optimize_four_quads(20)


# this is a solutino which produces a mostly symmetrical 75x75mm beam at the end of the TOPAS geometry, with no scatterers
# good starting point for a solution with no S1
# four_quads([8.23470106,  -5.77519295,  12.30291714, -70.89257372],
#           0.3, 0.2, 100000, 200, True)
