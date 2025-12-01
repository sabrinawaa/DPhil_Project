from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from RF_track_utils import RF_track_utils
from uniformity_fit import *
from scipy.stats import qmc, norm

from skopt import gp_minimize
from skopt.space import Real


def softplus(x, beta=1.0):
    return np.log1p(np.exp(beta * x)) / beta

def optimise_s2(R,  N_trials=10):
    """Optimize s2 params to get flat beam at water phantom."""
    N_particles = len(R)  # Number of particles in the beam

    def s2__merit_function(s2_params, N_particles, R):
            """Function to minimize - creates new lattice and tracks bunch."""
            s2_depth, s2_radius = s2_params
            dir = '/Users/sabrinawang/Desktop/Cameron_Project/'
            RFT_name = "RFT_optimised"
            dose_depth = 1 #mm
            
            setup = partrec_gaussian_optimiser_utils()
            setup.mute_terminal_output()

            setup.export_phsp(R, dir + RFT_name + '.phsp')

            setup.write_header(R, dir + RFT_name + '.header')

            setup.import_beam_topas(dir+ RFT_name, position=0)

            setup.add_gaussian_scatterer(
                s2_depth, s2_radius, dose_depth, 100, 'Aluminum', 100, show_shape=False)

            setup.add_collimator(1000,75,100, 200)
            #add water phantom 2500mm away, depth in1  mm, 50 bins in x, y, 1 bin in z
            setup.add_tank_bins(2500,dose_depth,50,50,1)     

            setup.run_topas(view_setup=False)

            fitted_map, P, sig, r_90 = fitDoseMap(N_particles, dose_depth, "S2_optimisation",plot=False)
            
            return -P
    
     # Optimization loop
    rng = np.random.default_rng()
    best_result = None
    best_merit = np.inf
    
    for _ in range(N_trials):
        # Random initial guess within bounds
        x0 = rng.uniform(low=0, high=10, size=2)
        # x0 = 3.44117706,  -6.08325848,  11.15552757, -56.11720984
        
        res = minimize(s2__merit_function,
                        args=(N_particles, R),
                      x0=x0,
                      bounds=((0,10), (0, 20)), #for s2_depth and s2_radius
                      method='Nelder-Mead',
                      options={
                        'direc': np.array([[0.1, 0],  # Step ±10 on k1
                                        [ 0, 0.1]]), # Step ±10 on k4
                        'xtol': 1.0,  # Stop if changes < 1.0
                        'ftol': 0.05    # Stop if merit changes < 0.1
                    }
                      )
        
        print(f'Trial {_}: M={res.fun:.2f}, S2 depth, radius={res.x}')
        
        if res.fun < best_merit:
            best_merit = res.fun
            best_result = res

    
            
    print('\nOptimization Complete:')
    print(f'Best M = {best_merit:.2f}')
    print(f'Optimal s2 depth and radius = {best_result.x}')
    return best_result.x



def optimise_k1s(N_particles, energy, stepsize =0.2, N_trials=10, sGoal=40, n_quads=4):
    """Optimize quadrupole strengths to minimize beam mismatch.
    
    Args:
        N_particles: Number of particles for tracking
        energy: Beam energy in MeV
        N_trials: Number of random initializations
        sGoal: Target beam size (mm)
        n_quads: Number of quadrupoles to optimize (default=4)
    """
    def quad_merit_function(k1s):
        """Function to minimize - creates new lattice and tracks bunch."""
        quadlattice = RF_track_utils(energy, k1s=k1s)
        quadlattice.add_drift(2.5)  # Add drift after quadrupoles to reach water phantom
        quadlattice.track_bunch_QR(N_particles, saveparams=False, 
                              E_deviation=0.5, #NEED TO CHANGE BACK TO 0.5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              sigma_x=1, sigma_xp=1, 
                              sigma_y=1, sigma_yp=1)
        T = quadlattice.transport_table[-1]  # Get end values
        sx, sy, ax, ay = T[5], T[6], T[3], T[4]
        N_end = len(quadlattice.phsp) #no of particles at end of tracking (add term to merit function to penalise large losses)
        
        M = (1000*(sx/sGoal-1)**2 + 1000*(sy/sGoal-1)**2 +
            np.exp(ax) + 10000*(ax/ay-1)**2) + 1000*(N_end/N_particles-1)**2
        #alpha diverging alpha^2 is fine so alphax**2 + alphay**2 instead of exp and also dont want ax/ay because ay can be 0
        #transmission- can do min(N_end/N_particles - transmission (e.g. 0.98))^2 to only penalise losses
        #diverging alpha- max(alpha+1,0)**2 to get alpha less than -1
        return M
    
    def quad_merit_function_1(k1s):
        """Function to minimize - creates new lattice and tracks bunch."""
        quadlattice = RF_track_utils(energy, k1s=k1s)
        quadlattice.add_drift(2.5)  # Add drift after quadrupoles to reach water phantom
        quadlattice.track_bunch_QR_twiss(N_particles, saveparams=False)
        I= quadlattice.trackedBunch.get_info()
         # Get end values
        sx, sy, ax, ay= I.sigma_x, I.sigma_y, I.alpha_x, I.alpha_y
        N_end = len(quadlattice.phsp) 
        print("sx, sy, ax, ay, N_end:", sx, sy, ax, ay, N_end)
        transmission = 0.98
        
        M = (100000*(sx/sGoal-1)**2 + 100000*(sy/sGoal-1)**2 +
            0* softplus(ax)**2 + (ax-ay)**2 + (N_end/N_particles-1)**2)#(softplus(transmission - N_end/N_particles))**2 )
        print("Merit M:", M)
        print("k1s:", k1s)
        # M = (1000*(sx/sGoal-1)**2 + 1000*(sy/sGoal-1)**2 +
        #     np.exp(ax) + 10000*(ax/ay-1)**2) + 1000*(N_end/N_particles-1)**2
        return M
    
# #trial quasirandom initialisations using Sobol sequence
#     # Optimization loop
#     sobol = qmc.Sobol(d=4, scramble=True)
#     u = sobol.random(N_trials)  

#     # Evaluate Sobol seeds first
#     M_vals = []
#     for x0 in u * 150 - 75:
#         M = quad_merit_function(x0)
#         M_vals.append(M)
    
#     best_idx = np.argsort(M_vals)[:3]  # take best 3
#     best_seeds = (u[best_idx] * 150 - 75)

    best_result = None
    best_merit = np.inf

#     # Run Powell only from best seeds
#     for _ in range (len(best_seeds)):
#         res = minimize(quad_merit_function, x0=best_seeds[_], bounds=[(-75, 75)]*n_quads, method='Powell')    

#trial just random initialisations
    rng = np.random.default_rng()
    
    for _ in range(N_trials):
        # Random initial guess within bounds
        x0 = rng.uniform(low=0, high=25, size=n_quads)
        x0[1] = -x0[1]
        x0[3] = -x0[3]
   
        #separate into x and y components
        
        # x0 = 3.44117706,  -6.08325848,  11.15552757, -56.11720984
        
        res = minimize(quad_merit_function_1,
                      x0=x0,
                      bounds=[(0,25), (-25,0), (0,25), (-75,0)], #for k1_1, k1_2, k1_3, k1_4
                      method='Nelder-Mead',)
                    #   options={
                    #     'direc': np.array([[stepsize, 0, 0, 0],  # Initial set of direction vectors for the Powell method.
                                        # [0, stepsize, 0, 0],   # Step  on k2
                                        # [0, 0, stepsize, 0],   # Step  on k3
                                        # [0, 0, 0, stepsize]]) # Step on k4
                        # 'xtol': 1.0,  # Stop if %changes < xtol Relative error in k1,2,3,4  acceptable for convergence
                        # 'ftol': 0.05    # Stop if merit changes < 0.1
                    # }
            
        
        print(f'Trial {_}: M={res.fun:.2f}, k1s={res.x}')
        
        if res.fun < best_merit:
            best_merit = res.fun
            best_result = res

    
            
    print('\nOptimization Complete:')
    print(f'Best M = {best_merit:.2f}')
    print(f'Optimal k1s = {best_result.x}')
    return best_result.x


if __name__ == "__main__":
    # optimise_k1s(10000, 200, N_trials=100, sGoal=40) #10000 particles, 200 energy
    # optimise_bay(1000, 200, sGoal=40)
    optimise_k1s(10000, 200, N_trials=1, sGoal=40)



#     quadlattice = RF_track_utils(200, k1s=[-5. ,        -5.  ,       -2.72216351,  1.02837097])
#     quadlattice.add_drift(2.5)  # Add drift after quadrupoles to reach water phantom
#     quadlattice.track_bunch(10000, saveparams=False, E_deviation=0, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1)
# # quadlattice.plot_phsp()

#     quadlattice.plot_phsp()