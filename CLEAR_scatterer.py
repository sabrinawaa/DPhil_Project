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
n_particles = int(1e6)
RFT_name = "CLEAR_line"
output_filename = "CLEAR_dual_scatterer"
profile = "dose" # "dose" or "intensity"

start = 'CA.QFD0350' #'CA.ACS0270S_MECH'
end = 'CA.DHJ0840' #'CA.STLINE$END'
CLEAR_lattice = get_beamline("CLEAR_Beamline_Survey.txt",start, end, P_ref, np.array([11, 32, 22, 19, 32, 18, 0, 67.5, 100, 0, 0]))
CLEAR_quads = CLEAR_lattice.get_quadrupoles()

# triplet700 = [ 14.39788773, -10.85504097,  18.88107575]
# for i in range(len(CLEAR_quads)-3, len(CLEAR_quads)):
#     CLEAR_quads[i].set_strength(triplet700[i-6])

print(CLEAR_lattice.get_length())


# Twiss parameters
# They are the ones at the starting point of your constructed lattice

Twiss = RF_Track.Bunch6d_twiss()

Twiss.beta_x = 40.5        # m
Twiss.beta_y = 36.8     # m
Twiss.alpha_x = -3.97 
Twiss.alpha_y = -4.27
Twiss.emitt_x = 22.8     # mm.mrad normalised emittance
Twiss.emitt_y = 23.8     # mm.mrad
# Twiss.sigma_t = 10 * RF_Track.ps       # mm/c   or 37 * RF_Track.ps
# Twiss.sigma_pt = 10     # permille
Twiss.mean_xp = 0.0
Twiss.mean_yp = 0.0



# P0 = RF_Track.Bunch6d(mass, population, Q, np.array([0,0,0,0,0,P_ref]).T)   # reference particle
# B0 = RF_Track.Bunch6d_QR(mass, population, Q, P_ref, Twiss, n_particles)             # reference bunch

# #track the bunch through the lattice
# P1 = CLEAR_lattice.track(P0)        # Track the reference particle
# B1 = CLEAR_lattice.track(B0)  

# R = B1.get_phase_space('%x %xp %y %yp %E %z')


# setup = partrec_gaussian_optimiser_utils()

# setup.export_phsp(R, dir + RFT_name + '.phsp')

# setup.write_header(R, dir + RFT_name + '.header')

# setup.import_beam_topas(dir+ RFT_name, position=0)

s1_pos = 84
s1_l, s2_width, s2_depth =  0.1, 1.4,0.8
s1_l, s2_width, s2_depth = 0.1, 1.6,2.43
dose_depth = 20 #mm



# setup.add_flat_scatterer(s1_l, 'Aluminum', s1_pos)
#                 # define gaussian scatterer (here with 22mm depth, 10mm radius, composed of 100 slices, situated 100mm downstream (standard convention) of first scatterer, )
                


# s2_thickness = [0.688, 0.778, 0.581, 0.386]
# s2_radii = [0.4, 0.8, 1.2, 1.6]
# # s2_radii = [1.4 , 1.05, 0.7 , 0.35]
# # s2_thickness = [0.08719553, 0.18061171, 0.26891643, 0.26327633]

# for i in range(len(s2_thickness)):
#     sname = 'S2_slice_'+str(i)
#     slice_position = s1_pos + s1_l + 532 + sum(s2_thickness[:i-1]) #532 is distance from s1 end to s2 start
#     setup.add_cylinder(sname, s2_thickness[i-1],0, s2_radii[i-1], 'Peek', slice_position)


# setup.add_cylinder('kapton_holder',0.025, 0,33,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+0.025) #check position


# setup.add_cylinder("vacuum_window",0.075,0,500,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+1524) #check position originally slice_pos+1250

# # setup.add_collimator(50,15,5,50, s1_pos+s1_l+532+s2_depth+1585) #collimator 15 outer radius, 5mm inner radius, 50mm length, position 200mm from RF

# setup.add_cylinder('tank_window',0.1, 0,100,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+2024) #this somehow blurs out the ring outside not sure why



# if profile == "dose":
#     setup.add_box('tank_layer1',20,100,100,"G4_WATER",s1_pos+532+s1_l+sum(s2_thickness)+2025) #this is the water phantom, 20mm thick, 100mm wide, 100mm high, position 200mm from RF")
#     setup.add_tank_bins(s1_pos+532+s1_l+sum(s2_thickness)+2025+20, 236, 100,100,4,output_filename,width=30)
# elif profile == "intensity":
#     setup.add_patient(s1_pos+532+s1_l+sum(s2_thickness)+2025)

# setup.run_topas(view_setup=False)

if profile == "intensity":
        # initialise plotting class
    plotter = partrec_foil_plotting('patient_beam.phsp' ) #filename defined inside partrec_gaussian_optimiser_utils
    # plot transverse distributions and energy spectrum at patient
    plotter.show_transverse_beam(output_filename, s1_l, s2_depth, s2_width,particle= 'e',fov= 30, col=75,n_bins=120)
    plotter.show_transverse_beam(output_filename, s1_l, s2_depth, s2_width,particle= 'y',fov=30, col=75,n_bins=120)

elif profile == "dose":

    #             # initialise plotting class
    # doseMap = getDosemap("DoseAtTank"+str(dose_depth)+ "_"+ output_filename+".csv",n_particles, dose_depth, output_filename, plot = True) 
    # fitDoseMap(n_particles, dose_depth,output_filename)

    x,y, doseMap = getDosemap("DoseAtTank"+str(dose_depth)+ "_"+ output_filename+".csv",n_particles, dose_depth, output_filename, acChargenC=15, plot = False)
    plot_dose(doseMap, x, y, strip_width=5, centred=True)
    plt.savefig('Output_figs/' + output_filename + "_dose_map.png")
    #plots second graph
