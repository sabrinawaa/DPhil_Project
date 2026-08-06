import numpy as np
import matplotlib.pyplot as plt
import RF_Track
import csv
# from YAG_RCF_analysis import *
from topas2numpy import BinnedResult
import os
from CLEAR_line import *
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from topasToDose import getDosemap
from uniformity_fit import *
from partrec_foil_plotting import partrec_foil_plotting
from RF_track_utils import *
import sys

def get_bin_edges(dim):
    return np.linspace(0, dim.n_bins * dim.bin_width, dim.n_bins + 1)

dir = '/Users/sabrinawang/Desktop/DPhil_Project/'
mass = RF_Track.electronmass    # particle mass in MeV/c^2
population = 10 * RF_Track.nC               # number of particles per bunch
Q = -1                          # particle charge in e units
P_ref = 197.3
n_particles = int(1e4)
RFT_name = "CLEAR_line"
output_filename = "CLEAR_dual_scatterer_0515_small_YAG_875_full"
profile = "dose" # "dose" or "intensity"

start = 'CA.QFD0350' #'CA.ACS0270S_MECH'
end = 'CA.DHJ0840' #'CA.STLINE$END'

# Twiss parameters
# They are the ones at the starting point of your constructed lattice

Twiss = RF_Track.Bunch6d_twiss()
Twiss.beta_x = 0.65        # m
Twiss.beta_y = 30.20    # m
Twiss.alpha_x = -0.20
Twiss.alpha_y = -1.87
Twiss.emitt_x = 66.26     # mm.mrad normalised emittance
Twiss.emitt_y = 86.26

CLEAR_lattice = get_beamline("CLEAR_Beamline_Survey.txt","CA.QFD0760", end, P_ref, np.array([11, 32, 22, 19, 32, 18, 0, 67.5, 100, 0, 0]))
B0 = RF_Track.Bunch6d_QR(mass, population, Q, P_ref, Twiss, n_particles)  
B1 = CLEAR_lattice.track(B0)  
R = B1.get_phase_space('%x %xp %y %yp %E %z')

s1_pos = 84
s1_l, s2_width, s2_depth =  0.1, 1.4,0.8
# s1_l, s2_width, s2_depth = 0.1, 1.6,2.43
dose_depth = 256 #robot depth -25
 #mm
setup = partrec_gaussian_optimiser_utils()
#position here always defined form the front face
setup.export_phsp(R, dir + RFT_name + '.phsp')

setup.write_header(R, dir + RFT_name + '.header')

setup.import_beam_topas(dir+ RFT_name, position=0)


setup.add_flat_scatterer(s1_l, 'Aluminum', s1_pos)
                # define gaussian scatterer (here with 22mm depth, 10mm radius, composed of 100 slices, situated 100mm downstream (standard convention) of first scatterer, )
                
# s2_thickness = [0.688, 0.778, 0.581, 0.386]
# s2_radii = [0.4, 0.8, 1.2, 1.6] #large

s2_thickness = [0.08719553, 0.18061171, 0.26891643, 0.26327633]
s2_radii = [1.4 , 1.05, 0.7 , 0.35] #small

for i in range(len(s2_thickness)):
    sname = 'S2_slice_'+str(i)
    slice_position = s1_pos + s1_l + 532 + sum(s2_thickness[:i-1]) #532 is distance from s1 end to s2 start
    setup.add_cylinder(sname, s2_thickness[i-1],0, s2_radii[i-1], 'Peek', slice_position)


setup.add_cylinder('kapton_holder',0.025, 0,33,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+0.025) #check position

setup.add_cylinder("vacuum_window",0.075,0,500,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+1489) #check position originally slice_pos+1250， then was +1524

# setup.add_collimator(50,15,5,50, s1_pos+s1_l+532+s2_depth+1585) #collimator 15 outer radius, 5mm inner radius, 50mm length, position 200mm from RF
setup.add_box("air1",435,100,100,"Air", s1_pos+532+s1_l+sum(s2_thickness)+1490) #air almost 450mm thick between elements, minus the tilted part of yag

setup.add_box("YAG", 0.55, 35,40, "YAG", s1_pos+532+s1_l+sum(s2_thickness)+1939,rotation=45) #check, originally +1570, 1939 is 85mm from tank

setup.add_box("air2",70,100,100,"Air", s1_pos+532+s1_l+sum(s2_thickness)+1953)#air almost 85mm thick between elements, minus tilted part of yag
setup.add_cylinder('tank_window',0.075, 0,100,"kapton",s1_pos+532+s1_l+sum(s2_thickness)+2024) #this somehow blurs out the ring outside not sure why

if profile == "dose":
    # setup.add_box('tank_layer1',20,100,100,"G4_WATER",s1_pos+532+s1_l+sum(s2_thickness)+2025) #this is the water phantom, 20mm thick, 100mm wide, 100mm high, position 200mm from RF")
    setup.add_tank_bins(s1_pos+532+s1_l+sum(s2_thickness)+2025, dose_depth, 100,100,dose_depth,output_filename,width=30)
elif profile == "intensity":
    setup.add_patient(s1_pos+532+s1_l+sum(s2_thickness)+2025)

setup.run_topas(view_setup=False)
