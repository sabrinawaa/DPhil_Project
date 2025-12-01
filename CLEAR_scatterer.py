import numpy as np
import matplotlib.pyplot as plt
import RF_Track

from CLEAR_line import *
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from topasToDose import getDosemap
from uniformity_fit import *
from partrec_foil_plotting import partrec_foil_plotting
from RF_track_utils import *
import sys

dir = '/Users/sabrinawang/Desktop/DPhil_Project/'
mass = RF_Track.electronmass    # particle mass in MeV/c^2
population = 10 * RF_Track.nC               # number of particles per bunch
Q = -1                          # particle charge in e units
P_ref = 198  
n_particles = int(1e5)
RFT_name = "CLEAR_line"

start = 'CA.QFD0350' #'CA.ACS0270S_MECH'
end = 'CA.DHJ0840' #'CA.STLINE$END'
CLEAR_lattice = get_beamline("CLEAR_Beamline_Survey.txt",start, end, P_ref, np.array(11*[0]))
print(CLEAR_lattice.get_length())

ks = CLEAR_lattice.get_quadrupoles()

ks[3].set_K1(-70)
ks[4].set_K1(20)
ks[5].set_K1(15)
# Twiss parameters
# They are the ones at the starting point of your constructed lattice

Twiss = RF_Track.Bunch6d_twiss()

Twiss.beta_x = 43.1        # m
Twiss.beta_y = 117      # m
Twiss.alpha_x = -0.557 
Twiss.alpha_y = 0.553
Twiss.emitt_x = 3.6     # mm.mrad normalised emittance
Twiss.emitt_y = 4.48     # mm.mrad
# Twiss.sigma_t = 10 * RF_Track.ps       # mm/c   or 37 * RF_Track.ps
# Twiss.sigma_pt = 10     # permille
Twiss.mean_xp = 0.0
Twiss.mean_yp = 0.0



P0 = RF_Track.Bunch6d(mass, population, Q, np.array([0,0,0,0,0,P_ref]).T)   # reference particle
B0 = RF_Track.Bunch6d(mass, population, Q, P_ref, Twiss, n_particles)             # reference bunch

#track the bunch through the lattice
P1 = CLEAR_lattice.track(P0)        # Track the reference particle
B1 = CLEAR_lattice.track(B0)  

R = B1.get_phase_space('%x %xp %y %yp %E %z')


setup = partrec_gaussian_optimiser_utils()

setup.export_phsp(R, dir + RFT_name + '.phsp')

setup.write_header(R, dir + RFT_name + '.header')

setup.import_beam_topas(dir+ RFT_name, position=0)

s1_pos = 56+28 #mm  #position of first scatterer from RF exit (kicker 840 exit)
s1_depth = 0.1 #mm
s2_depth = 2.6 #mm
s2_radius = 1 #mm
dose_depth = 5 #mm
output_filename = "CLEAR_dual_scatterer"


setup.add_flat_scatterer(s1_depth, 'Aluminum', s1_pos)
                # define gaussian scatterer (here with 22mm depth, 10mm radius, composed of 100 slices, situated 100mm downstream (standard convention) of first scatterer, )
# s2_thickness = [0.688, 0.778, 0.608, 0.356]
s2_thickness = [0.688, 0.778, 0.581, 0.386]
s2_radii = [0.4, 0.8, 1.2, 1.6]
for i in range(len(s2_thickness)):
    sname = 'S2_slice_'+str(i)
    slice_position = s1_pos + s1_depth + 532 + sum(s2_thickness[:i-1]) #532 is distance from s1 end to s2 start
    setup.add_cylinder(sname, s2_thickness[i-1], s2_radii[i-1], 'Peek', slice_position)

setup.add_cylinder('kapton_holder', 0.05, 50, 'Kapton' , slice_position + s2_thickness[i-1]/2+ 0.025)        
# setup.add_gaussian_scatterer(
#     s2_depth, s2_radius, 0.9, 5, 'Peek', s1_pos+s1_depth+532)
setup.add_cylinder('kapton_window', 0.05, 50, 'Kapton' , slice_position+1250)

setup.add_collimator(50,15,5,50, s1_pos+s1_depth+532+s2_depth+1585) #collimator 15 outer radius, 5mm inner radius, 50mm length, position 200mm from RF
#                 add water phantom 2500mm away, depth in1  mm, 50 bins in x, y, 1 bin in z

setup.add_tank_bins(s1_pos+s1_depth+532+s2_depth+1585+50+389,dose_depth,150,150,1, output_filename, 30)

setup.run_topas(view_setup=False)
doseMap = getDosemap("DoseAtTank"+str(dose_depth)+ "_"+ output_filename+".csv",n_particles, dose_depth, output_filename, plot = True) 
fitDoseMap(n_particles, dose_depth,output_filename)