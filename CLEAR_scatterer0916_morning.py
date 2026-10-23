from turtle import setup

import numpy as np
# import matplotlib.pyplot as plt
import RF_Track
import csv
# from YAG_RCF_analysis import *
# from topas2numpy import BinnedResult
import os
from CLEAR_line import *
from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
# from topasToDose import getDosemap
# from uniformity_fit import *
# from partrec_foil_plotting import partrec_foil_plotting
# from RF_track_utils import *
import sys

def get_bin_edges(dim):
    return np.linspace(0, dim.n_bins * dim.bin_width, dim.n_bins + 1)


def main(args):
  
    run_no = int(args[0])

    # each condor job otherwise samples the same beam by default (RF_Track's
    # RNG seed is fixed, not time-based), so independent jobs must be seeded
    RF_Track.rng_set_seed(run_no)

    dir = os.path.expanduser('~/DPhil_Project/')
    # per-job private scratch dir (falls back to dir for interactive/local runs) so
    # topas_main/.phsp/.header files from concurrent condor jobs never collide
    scratch_dir = os.environ.get('_CONDOR_SCRATCH_DIR', dir)
    if not scratch_dir.endswith('/'):
        scratch_dir += '/'

    mass = RF_Track.electronmass    # particle mass in MeV/c^2
    population = 10 * RF_Track.nC               # number of particles per bunch
    Q = -1                          # particle charge in e units
    P_ref = 195.2
    n_particles = int(5e4)
    RFT_name = "CLEAR_0916" + str(run_no)
    output_filename = "CLEAR_dual_scatterer_0916_dual_col_YAG_875_" + str(run_no)
    profile = "dose" # "dose" or "intensity"

    start = 'CA.QFD0350' #'CA.ACS0270S_MECH'
    end = 'CA.DHJ0840' #'CA.STLINE$END'

    # Twiss parameters
    # They are the ones at the starting point of your constructed lattice

    Twiss = RF_Track.Bunch6d_twiss()
    Twiss.beta_x = 9.37      # m
    Twiss.beta_y = 14.26   # m
    Twiss.alpha_x = -4.12
    Twiss.alpha_y = 3.59
    Twiss.emitt_x = 185.5    # mm.mrad normalised emittance
    Twiss.emitt_y = 114.6

    CLEAR_lattice = get_beamline("CLEAR_Beamline_Survey.txt","CA.QFD0760", end, P_ref, np.array([10.7,28,18,0,0.2,0.1,44.6,77,58.2,0,0]))
    B0 = RF_Track.Bunch6d(mass, population, Q, P_ref, Twiss, n_particles)  
    B1 = CLEAR_lattice.track(B0)  
    R = B1.get_phase_space('%x %xp %y %yp %E %z')

    s1_pos = 148
    s1_l, s2_width, s2_depth =  0.1, 1.4,0.8
    # s1_l, s2_width, s2_depth = 0.1, 1.6,2.43
    dose_depth = 100 #robot depth -25
    #mm
    try:
        setup = partrec_gaussian_optimiser_utils(file_directory=scratch_dir,input_filename=f'topas_main_{run_no}.txt',topas_dir ='/home/sabrinawang/Applications/',no_of_threads=1
                                                 )
        #position here always defined form the front face
        setup.export_phsp(R, scratch_dir + RFT_name + '.phsp')

        setup.write_header(R, scratch_dir + RFT_name + '.header')

        setup.import_beam_topas(scratch_dir+ RFT_name, position=0)


        setup.add_flat_scatterer(s1_l, 'Aluminum', s1_pos)
                        # define gaussian scatterer (here with 22mm depth, 10mm radius, composed of 100 slices, situated 100mm downstream (standard convention) of first scatterer, )
                        
        # s2_thickness = [0.688, 0.778, 0.581, 0.386]
        # s2_radii = [0.4, 0.8, 1.2, 1.6] #large

        s2_thickness = [0.08719553, 0.18061171, 0.26891643, 0.26327633]
        s2_radii = [1.4 , 1.05, 0.7 , 0.35] #small

        
        for i in range(len(s2_thickness)):
            sname = 'S2_slice_'+str(i)
            slice_position = s1_pos + s1_l + 522 + sum(s2_thickness[:i]) #522 is distance from s1 end to s2 start, used to be 532
            setup.add_cylinder(sname, s2_thickness[i],0, s2_radii[i], 'Peek', slice_position)


        setup.add_cylinder('kapton_holder',0.025, 0,33,"kapton",s1_pos+522+s1_l+sum(s2_thickness)+0.025) #check position

        setup.add_cylinder("vacuum_window",0.12,0,500,"kapton",s1_pos+522+s1_l+sum(s2_thickness)+1464.5) #new one
        setup.add_cylinder("vacuum_window_coating",0.005,6,500,"G4_Cu",s1_pos+522+s1_l+sum(s2_thickness)+1464.5+0.12) #check position originally slice_pos+1250， then was +1524, old one

        setup.add_box("air", 900,200,200,"Air",s1_pos+522+s1_l+sum(s2_thickness)+1465) #in air test stand

        setup.add_collimator(50,15,5,50, 92, parent="air") #collimator 15 outer radius, 5mm inner radius, 50mm length, position 200mm from RF

        setup.add_box("YAG", 0.22, 35,40, "YAG", 306+50+92 ,rotation=22.5, parent = "air") #check, originally +1570, 1939 is 85mm from tank, originally 0.55 thick for may measurements

        setup.add_cylinder('tank_window',0.075, 0,100,"kapton",306+50+92+99, parent = "air") 

        if profile == "dose":
            setup.add_tank_bins(306+50+92+99+6, dose_depth, 100,100,dose_depth,output_filename,width=30,parent = "air")

        elif profile == "intensity":
            setup.add_patient(306+50+92+99+6+dose_depth, width=30,parent = 'air')

        setup.run_topas(view_setup=False)
    
    finally:
        for p in (
            scratch_dir + f"topas_main_{run_no}.txt",
            scratch_dir + RFT_name + '.header',
            scratch_dir + RFT_name + '.phsp',
        ):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main(sys.argv[1:])