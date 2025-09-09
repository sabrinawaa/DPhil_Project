import pandas as pd
import numpy as np
from RF_track_4quad import *
# from sabrina_rf_topas_conversion import *
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from topasToDose import getDosemap
from uniformity_fit import *
from partrec_foil_plotting import partrec_foil_plotting
from RF_track_utils import *
from Optimisation_funcs import *

# parameters
quad_length = 0.3
drift_l = 0.2
k1_1, k1_2, k1_3, k1_4 =optimise_k1s(10000, 200, stepsize=20,N_trials=10, sGoal=40) #1000 particles, 200 energy
# k1_1, k1_2, k1_3, k1_4 = 8.23470106,  -5.77519295,  12.30291714, -70.89257372 #<75
# k1_1, k1_2, k1_3, k1_4 =   3.44117706,  -6.08325848,  11.15552757, -56.11720984 #

dir = '/Users/sabrinawang/Desktop/Cameron_Project/'
RFT_name = '4quad' #names of phsp and header files from RF Track output
output_filename = "4quad_col"
n_particles = 1e3 #1e5

profile = "dose" # "dose" or "intensity"
system = "quad-scatterer" # "dual-scatterer" or "quad-scatterer"

# end of params

if system == "dual-scatterer":
    k1_1, k1_2, k1_3, k1_4 = 0, 0, 0, 0

# R = four_quads(quad_length, k1_1, k1_2, k1_3, k1_4, drift_l, n_particles, 200,saveparams=True, plot=True)
# quadlattice = RF_track_utils(200, [k1_1, k1_2, k1_3, k1_4])
# # quadlattice.add_drift(2.5)  # Add drift after quadrupoles to reach water phantom
# R = quadlattice.track_bunch(n_particles, saveparams=True, E_deviation=0.5, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1)
# quadlattice.plot_phsp()

# R = np.loadtxt(f"RFT_k1s={k1_1}_{k1_2}_{k1_3}_{k1_4}_N={int(n_particles)}.txt") 
# s2_depth, s2_radius = optimise_s2( R, N_trials=10) #optimise s2 params to get flat beam at water phantom

# s1_depths = [0.95] if system == "dual-scatterer" else [0]
# for s1_depth in s1_depths:
#     for s2_depth in [3]:
#         for s2_radius in [13]:


#             for dose_depth in [1]: #mm

#                 setup = partrec_gaussian_optimiser_utils()

#                 setup.export_phsp(R, dir + RFT_name + '.phsp')

#                 setup.write_header(R, dir + RFT_name + '.header')

#                 setup.import_beam_topas(dir+ RFT_name, position=0)

#                 if system == "dual-scatterer":
#                     # add pre-scatterer to magnify beam, thickness in mm to make beam 7.5mm radius
#                     setup.add_flat_scatterer(s1_depth, 'Tantalum')
#                 # define gaussian scatterer (here with 22mm depth, 10mm radius, composed of 100 slices, situated 100mm downstream (standard convention) of first scatterer, )
                
#                 setup.add_gaussian_scatterer(
#                     s2_depth, s2_radius, 1, 100, 'Aluminum', 100, show_shape=False)

#                 setup.add_collimator(275,75,100, 2400) # # add collimator 275mm outer radius, 75mm inner radius, length 100mm, position 2400mm from origin (right infront of patient)
                
#                 if profile == "intensity":
#                         #add vacuum patient 2500mm away
#                     setup.add_patient(2500)

#                 elif profile == "dose":
#                     #add water phantom 2500mm away, depth in mm, 50 bins in x, y, 1 bin in z
#                     setup.add_tank_bins(2500,dose_depth,50,50,1)

                
                
                
#                 setup.run_topas(view_setup=False)

        #         if profile == "intensity":
        #                 # initialise plotting class
        #             plotter = partrec_foil_plotting('patient_beam.phsp' ) #filename defined inside partrec_gaussian_optimiser_utils
        #             # plot transverse distributions and energy spectrum at patient
        #             plotter.show_transverse_beam(output_filename, s1_depth, s2_depth, s2_radius,particle= 'e',fov= 150, col=75)
        #             plotter.show_transverse_beam(output_filename, s1_depth, s2_depth, s2_radius,particle= 'y',fov= 150, col=75)

        #         elif profile == "dose":

        # #             # initialise plotting class
        #             doseMap = getDosemap("DoseAtTank"+str(dose_depth)+".csv",n_particles, dose_depth, output_filename, plot = True) 
        #             #note here plots first graph
        #             fitDoseMap(n_particles, dose_depth, output_filename)
        #             #plots second graph



