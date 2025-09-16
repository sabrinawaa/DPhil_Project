#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:51:43 2023

@author: robertsoncl
"""
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from topasToDose import getDosemap
from uniformity_fit import *
# class of methods used for generating topas scripts for scattering foils


class partrec_gaussian_optimiser_utils():
    # set filepaths and number of threads available
    # input file is the topas script being generated and run here
    # output file is 'patient_beam'
    def __init__(
        self,
        home_directory="/Applications/",
        no_of_threads="6",
        input_filename="topas_main.txt",
        file_directory = '/Users/sabrinawang/Desktop/DPhil_Project/'
    ):
        # write new topas script
        file = open(file_directory + input_filename, "w")
        # set number of threads depending on computing power available
        file.write("i:Ts/NumberOfThreads=" + no_of_threads + "\n")
        # define arbitrarily large world
        file.write("d:Ge/World/HLX = 5.0 m\n")
        file.write("d:Ge/World/HLY = 5.0 m\n")
        file.write("d:Ge/World/HLZ = 5.0 m\n")
        # set world as vacuum for simplicity
        file.write('s:Ge/World/Material = "Vacuum"\n \n')
        self.home_directory = home_directory
        self.file_directory = file_directory
        # set filename and file object as class attributes to retrieve later
        self.input_filename = input_filename
        self.file = file
    # define Gaussian beam in terms of phase space parameters
    # sigma, sigmap, E in mm, mrad, MeV respectively

    def mute_terminal_output(self):
        self.file.write('b:Ts/QuietMode = "True"  \n')
        
    def generate_phsp_beam(self, sigma_x, sigma_y, sigma_px, sigma_py, E, delta_E, N):
        file = self.file
        # initialise beam
        file.write('s:So/acc_source/Type = "Beam"\n')
        # set number of particles in beam
        file.write("i:So/acc_source/NumberOfHistoriesInRun =" + str(N) + "\n")
        # required to set beam direction
        file.write('s:So/acc_source/Component = "BeamPosition"\n')
        file.write("d:Ge/BeamPosition/TransZ= 0 mm\n")
        # specify electron beam
        file.write('s:So/acc_source/BeamParticle="e-"\n')
        file.write("d:So/acc_source/BeamEnergy= " + str(E) + " MeV \n")
        # define initial beam distribution from method arguments
        file.write('s:So/acc_source/BeamPositionDistribution= "Gaussian"\n')

        file.write("d:So/acc_source/BeamPositionSpreadX = " +
                   str(sigma_x) + " mm\n")
        file.write("d:So/acc_source/BeamPositionSpreadY = " +
                   str(sigma_y) + " mm\n")
        file.write("d:So/acc_source/BeamAngularSpreadX= " +
                   str(sigma_px) + " mrad\n")
        file.write("d:So/acc_source/BeamAngularSpreadY= " +
                   str(sigma_py) + " mrad\n")
        # cutoff at 5 sigma, retaining >99.9% of beam
        file.write(
            "d:So/acc_source/BeamPositionCutoffX = " +
            str(5 * sigma_x) + " mm\n"
        )
        file.write(
            "d:So/acc_source/BeamPositionCutoffY = " +
            str(5 * sigma_y) + " mm\n"
        )
        file.write(
            "d:So/acc_source/BeamAngularCutoffX= " +
            str(5 * sigma_px) + " mrad\n"
        )
        file.write(
            "d:So/acc_source/BeamAngularCutoffY = " +
            str(5 * sigma_py) + " mrad\n"
        )
        file.write('s:So/acc_source/BeamAngularDistribution="Gaussian"\n')
        # set delta E
        file.write("u:So/acc_source/BeamEnergySpread =" + str(delta_E) + " \n")
        # set beam as ellipse rather than rectangle
        file.write('s:So/acc_source/BeamPositionCutoffShape = "Ellipse"\n')

    # generate Gaussian beam from twiss parameters

    def generate_twiss_beam(self, beta_x, beta_y, emitt_x, emitt_y, alpha_x, alpha_y, E, N):
        file = self.file
        file.write('s:So/acc_source/Type = "emittance"\n')
        # set number of particles in beam
        file.write("i:So/acc_source/NumberOfHistoriesInRun =" + str(N) + "\n")
        # required to set beam direction
        file.write('s:So/acc_source/Component = "BeamPosition"\n')
        file.write("d:Ge/BeamPosition/TransZ= 0 mm\n")
        # specify proton/electron beam
        file.write('s:So/acc_source/BeamParticle="electron"\n')
        # set energy
        file.write("d:So/acc_source/BeamEnergy= " + str(E) + " MeV \n")
        # define initial beam distribution as flat with 1mm radius
        file.write('s:So/acc_source/Distribution= "twiss_gaussian"\n')

        file.write("u:So/acc_source/AlphaX = " + str(alpha_x) + "\n")
        file.write("u:So/acc_source/AlphaY = " + str(alpha_y) + "\n")
        file.write("d:So/acc_source/BetaX = " + str(beta_x) + " m \n")
        file.write("d:So/acc_source/BetaY = " + str(beta_y) + " m \n")
        file.write("d:So/acc_source/EmittanceX = " + str(emitt_x) + " um \n")
        file.write("d:So/acc_source/EmittanceY = " + str(emitt_y) + " um \n")

        file.write("u:So/acc_source/ParticleFractionX = 0.90 \n")
        file.write("u:So/acc_source/ParticleFractionY = 0.90 \n")

    # script lines for pre scatterer
    # position refers to longitudinal placement of upstream scatterer face in mm
    # by default, positioned at origin
    def add_flat_scatterer(self, thickness, material, position=0):
        file = self.file
        file.write('s:Ge/S1/Type = "TsCylinder"\n')
        # defined from world centre
        file.write('s:Ge/S1/Parent="World"\n')
        # set material based on input argument
        file.write("s:Ge/S1/Material=" + '"' + material + '"' + "\n")

        # set radius of scatterer (make sure it is larger than beam radius)
        file.write("d:Ge/S1/Rmax =  100  mm\n")
        # solid scatterer - inner radius must be set to 0
        file.write("d:Ge/S1/Rmin= 0 mm\n")
        # define thickness of scatterer using previously define half length
        # topas works with half lengths rather than full lengths
        file.write("d:Ge/S1/HL = " + str(thickness / 2) + " mm\n")
        # set position of scatterer so that the edge is on the origin
        file.write("d:Ge/S1/TransZ = -" +
                   str(position+thickness / 2) + " mm\n")

    # position here refers to downstream face of scatterer (don't ask me why)
    # convolution factor = 1 for standard Gaussian scatterer
    def add_gaussian_scatterer(self, max_thickness, radius, convolution_factor, N_slices, material, position, show_shape=False):
        file = self.file

        s2_sigma = radius/2
        # define spread of gaussian shape
        # and precision (number of slices in shape) with step argument
        # x = np.arange(-half_width, half_width, step=1)
        step = radius / (N_slices)
        x = np.arange(-(radius+step), 0, step=step)
        # construct gaussian profile from method argument sigma
        # convolution factor "warps" shape
        y = norm.pdf(x, 0, s2_sigma * convolution_factor)
        # plt.plot(x, y)
        # scale for input amplitude
        y = y - min(y)
        y_scaling_factor = max_thickness / max(y)
        y = y * y_scaling_factor
        if show_shape is True:
            plt.plot(np.append(x, -np.flip(x, 0)), np.append(y, np.flip(y)))
            plt.xlabel('r[mm]')
            plt.ylabel('h[mm]')
        # scale height and normalise base to 0
        # according to method argument max_height

        # begin loop to create stack of cylinders following Gaussian shape
        for i in range(1, len(y)):
            # Don't try to create 0 height widths
            # skip relevant rows
            L = y[i] - y[i - 1]
            HL = L / 2

            # define slice name - required for Topas
            sname = "slice" + str(i)
            file.write("d:Ge/" + sname + "/HL = " + str(HL) + " mm\n")
            prev_HL = HL
            # define slice as cylinder
            file.write("s:Ge/" + sname + '/Type = "TsCylinder"\n')
            # in previously defined world
            file.write("s:Ge/" + sname + '/Parent="World"\n')
            # define material
            if HL < 0:
                file.write("s:Ge/" + sname + "/Material=" +
                           '"' + 'Vacuum' + '"' + "\n")
            else:
                file.write("s:Ge/" + sname + "/Material=" +
                           '"' + material + '"' + "\n")
            # set radius of slice from horizontal slice steps
            file.write("d:Ge/" + sname + "/Rmax = " + str(abs(x[i])) + " mm\n")
            # set inner radius of slice to 0 - slice is solid, not a hoop
            file.write("d:Ge/" + sname + "/Rmin= 0 mm\n")
            # define height of slice from difference between y values
            # of points from defined Gaussian shape
            # set position to build Gaussian pointed toward beam
            # with distance beam_to_S2 from beam source to tip
            # and distance S2_to_scorer from shape base
            file.write(
                "d:Ge/"
                + sname
                + "/TransZ = -"
                + str(position - y[i - 1] - L / 2)
                + " mm\n"
            )
            # increment to begin next slice until shape completion
            i = i + 1

    def add_collimator(self, Rmax, Rmin, length, position):
        file = self.file
        # define collimator as cylinder
        file.write('s:Ge/Collimator/Type = "TsCylinder"\n')
        # set parent to world
        file.write('s:Ge/Collimator/Parent = "World"\n')
        # set material - vacuum for simplicity
        file.write('s:Ge/Collimator/Material="G4_W"\n')
        # set radius of collimator
        file.write("d:Ge/Collimator/Rmax = " + str(Rmax) + " mm\n")
        # set inner radius to 0 - solid collimator
        file.write("d:Ge/Collimator/Rmin = " + str(Rmin) + " mm\n")
        # set half length of collimator
        # topas works with half lengths rather than full lengths
        file.write("d:Ge/Collimator/HL = " + str(length/2)+ " mm   \n")  # set arbitrary length
        # set position of collimator at appropriate distance from beam source
        file.write("d:Ge/Collimator/TransZ = -" + str(position) + " mm\n")
       

    def add_patient(self, position):
        file = self.file
        # define scorer surface
        file.write('s:Ge/ScorerSurface/Type="TsBox"\n')
        file.write('s:Ge/ScorerSurface/Parent = "World"\n')
        # set arbitrary material - vacuum for simplicity
        file.write('s:Ge/ScorerSurface/Material="G4_WATER"\n')

        # set arbitrarily large surface area of scorer
        file.write("d:Ge/ScorerSurface/HLX = 1 m\n")
        file.write("d:Ge/ScorerSurface/HLY = 1 m\n")
        # set small thickness for precision
        file.write("d:Ge/ScorerSurface/HLZ = 0.01 mm\n")
        # set at appropriate distance for consistency between variables
        file.write("d:Ge/ScorerSurface/TransZ = -" +
                   str(position) + " mm\n")
        # set up phase space scorer
        file.write('s:Sc/patient_beam/Quantity = "PhaseSpace"\n')
        # place at previously defined patient location
        file.write('s:Sc/patient_beam/Surface = "ScorerSurface/ZPlusSurface"\n')
        # file.write(
        #     's:Sc/patient_beam/OnlyIncludeParticlesOfGeneration = "Primary"\n') # if delete will also include photons
        # output as ascii file
        file.write('s:Sc/patient_beam/OutputType = "ASCII"\n')
        file.write('s:Sc/patient_beam/IfOutputFileAlreadyExists = "Overwrite"\n')
        # reduce terminal output to improve RunTime and reduce clutter
        file.write('b:Sc/patient_beam/OutputToConsole = "False"\n')
        # set various checks to 0 to decrease RunTime
        file.write('b:Ge/CheckForOverlaps = "False" \n')
        file.write('b:Ge/QuitIfOverlapDetected = "False"\n')
        file.write('b:Ph/ListProcesses = "False"\n')
        file.write('b:Ge/CheckForUnusedComponents = "False"\n')
        self.pp = position

    # position refers to upstream face of tank
    def add_tank(self, position, depth):
        file = self.file
        file.write('s:Ge/Tank/Type="TsBox"\n')
        file.write('s:Ge/Tank/Parent = "World"\n')
        # set arbitrary material - vacuum for simplicity
        file.write('s:Ge/Tank/Material="G4_WATER"\n')
        # set arbitrarily large surface area of scorer
        file.write("d:Ge/Tank/HLX = 0.1 m\n")
        file.write("d:Ge/Tank/HLY = 0.1 m\n")
        file.write("d:Ge/Tank/HLZ = " + str(depth / 2) + " mm\n")
        file.write("d:Ge/Tank/TransZ=-" +
                   str(position+depth/2) + " mm\n")
        
    def add_tank_bins(self, position, depth, x_bins, y_bins, z_bins, output_filename):
        file = self.file
        file.write('s:Ge/Tank/Type="TsBox"\n')
        file.write('s:Ge/Tank/Parent = "World"\n')
        # set arbitrary material - vacuum for simplicity
        file.write('s:Ge/Tank/Material="G4_WATER"\n')
        # set arbitrarily large surface area of scorer
        file.write("d:Ge/Tank/HLX = 0.15 m\n")
        file.write("d:Ge/Tank/HLY = 0.15 m\n")
        file.write("d:Ge/Tank/HLZ = " + str(depth / 2) + " mm\n")
        file.write("d:Ge/Tank/TransZ=-" +
                   str(position+depth/2) + " mm\n")
        file.write("i:Ge/Tank/XBins = "+ str(x_bins) + "\n")
        file.write("i:Ge/Tank/YBins = "+ str(y_bins) + "\n")
        file.write("i:Ge/Tank/ZBins = "+ str(z_bins) + "\n")
        file.write('s:Sc/DoseAtTank/Quantity = "DoseToMedium" \n')
        file.write('s:Sc/DoseAtTank/Component = "Tank"\n')
        file.write('s:Sc/DoseAtTank/Surface = "Tank/ZPlusSurface"\n')
        # file.write('s:Sc/DoseAtTank/OnlyIncludeParticlesOfGeneration = "Primary"\n')
        # output as csv file
        file.write('s:Sc/DoseAtTank/OutputFile = "DoseAtTank' + str(depth) + '_'+ output_filename+'"\n')
        file.write('s:Sc/DoseAtTank/OutputType = "CSV"\n')
        file.write('s:Sc/DoseAtTank/IfOutputFileAlreadyExists = "Overwrite"\n')
        # reduce terminal output to improve RunTime and reduce clutter
        file.write('b:Sc/DoseAtTank/OutputToConsole = "False"\n')
        # set various checks to 0 to decrease RunTime
        file.write('b:Ge/CheckForOverlaps = "False" \n')
        file.write('b:Ge/QuitIfOverlapDetected = "False"\n')

        file.write('b:Ph/ListProcesses = "False"\n')
        file.write('b:Ge/CheckForUnusedComponents = "False"\n')

    # X bending dipole

    def add_dipole(self, strength, position, lx, ly, lz):
        file = self.file
        file.write('s:Ge/Dipole/Type="TsBox"\n')
        file.write('s:Ge/Dipole/Parent = "World"\n')
        # set arbitrary material - vacuum for simplicity
        file.write('s:Ge/Dipole/Material="Vacuum"\n')
        file.write('s:Ge/Dipole/Field="DipoleMagnet"\n')
        # set arbitrarily large surface area of scorer
        file.write("d:Ge/Dipole/HLX ="+str(lx/2)+" mm\n")
        file.write("d:Ge/Dipole/HLY ="+str(ly/2)+" mm\n")
        file.write("d:Ge/Dipole/HLZ ="+str(lz/2)+" mm\n")
        file.write("d:Ge/Dipole/TransZ=-"+position+" mm\n")
        file.write("u:Ge/Dipole/MagneticFieldDirectionX=0.0\n")
        file.write("u:Ge/Dipole/MagneticFieldDirectionY=1.0\n")
        file.write("u:Ge/Dipole/MagneticFieldDirectionZ=0.0\n")
        file.write("d:Ge/Dipole/MagneticFieldStrength="+str(strength)+" T\n")

#importing rf track output phase space file into topas phsp format
    def export_phsp(self, R, path_to_phsp_file):
        X = R[:, 0]
        PX = R[:, 1]
        Y = R[:, 2]
        PY = R[:, 3]
        E = R[:, 4] #changed from 5 to 4 here
        # assume constant position in Z at this stage - not physical bu
        # as we use TOPAS we through away our knowledge of accelerator physics
        Z = np.zeros(len(X))
        # we equally weight the beam
        weight = np.full(len(X), 1)
        ptype = np.full(len(X), 11)
        # topas uses rad rather than mrad
        PX = PX / 1000
        PY = PY / 1000
        # topas header uses cm
        X = X / 10
        Y = Y / 10
        # all particles are travelling forwards
        neg_cos = np.full(len(X), 1)
        # all primary particles, so first score is always 1, like weight
        first_score = weight
        # initialise dataframe with all relevant data required by TOPAS
        phase_space_data = pd.DataFrame(
            {
                "X": X,
                "Y": Y,
                "Z": Z,
                "PX": PX,
                "PY": PY,
                "E": E,
                "Weight": weight,
                "ptype": ptype,
                "neg_cos": neg_cos,
                "first_score": first_score,
            }
        )
        # convert DataFrame into astropy table for manipulation
        phase_space_table = Table.from_pandas(phase_space_data)
        # format for topas input
        ascii.write(
            phase_space_table,
            path_to_phsp_file,
            format="fixed_width_no_header",
            delimiter="\t",
            overwrite=True,
        )
    # TOPAS beam inputs need a header alongside the .phsp file so that it knows
    # how to process the beam data
    # path_to_headerfile is just where the new file will go and its name
    # this must be the same filename as the phsp file, but ending with .header
    # rather than .phsp


    def write_header(self, R, path_to_header_file): # header file for phsp from rf track
        file = self.file
        E = R[:, 4] #changed from 5 to 4
        N_particles_str = str(len(E))
        file = open(path_to_header_file, "w")
        file.write("TOPAS ASCII Phase Space\n")
        file.write("\n")
        file.write("Number of Original Histories: " + N_particles_str + "\n")
        file.write(
            "Number of Original Histories that Reached Phase Space: "
            + N_particles_str
            + "\n"
        )
        file.write("Number of Scored Particles: " + N_particles_str + "\n")
        file.write("\n")
        file.write("Columns of data are as follows:\n")
        file.write(" 1: Position X [cm]\n")
        file.write(" 2: Position Y [cm]\n")
        file.write(" 3: Position Z [cm]\n")
        file.write(" 4: Direction Cosine X\n")
        file.write(" 5: Direction Cosine Y\n")
        file.write(" 6: Energy [MeV]\n")
        file.write(" 7: Weight\n")
        file.write(" 8: Particle Type (in PDG Format)\n")
        file.write(
            " 9: Flag to tell if Third Direction Cosine is Negative (1 means true)\n"
        )
        file.write(
            "10: Flag to tell if this is the First Scored Particle from this History (1 means true)\n"
        )
        file.write("\n")
        file.write("Number of e-: " + N_particles_str + "\n")
        file.write("\n")
        file.write("Minimum Kinetic Energy of e-: " + str(min(E)) + " MeV\n")
        file.write("\n")
        file.write("Maximum Kinetic Energy of e-: " + str(max(E)) + " MeV\n")

    # function adds
    # relevant lines in TOPAS to import beam from phsp and header files
    # acc_source is the name of the beam I've given here, this is arbitrary
    # position allows you to modify the position in Z in the world that the imported
    # beam starts from
    # infile is the path to the phase space file you want to import
    # give this without the .phsp and it will automatically find the
    # header as well - make sure those are in the same directory


    def import_beam_topas(self,infile, position=0):
        file = self.file
        file.write('s:So/acc_source/Type = "PhaseSpace"\n')
        file.write('s:So/acc_source/Component = "World"\n')
        file.write('s:So/acc_source/PhaseSpaceFileName = "'+infile+'"\n')
        file.write('d:So/acc_source/TransZ = -'+str(position)+' mm\n')

        

    def run_topas(self, topas_filename=lambda self: self.input_filename, view_setup=False):
        if callable(topas_filename):
            topas_filename = topas_filename(self) 

        file = self.file
        if view_setup is True:
            file.write('s:Gr/ViewA/Type             = "OpenGL"\n')
            file.write('b:Ts/UseQt = "True"\n')

        # Topas script complete, close file
        file.close()
        # set up environment for topas
        os.system(
            "export TOPAS_G4_DATA_DIR=/Applications/G4Data && "
            "export QT_QPA_PLATFORM_PLUGIN_PATH=/Applications/topas/Frameworks && "
            "/Applications/topas/bin/topas "
            + self.file_directory
            + topas_filename
        )

