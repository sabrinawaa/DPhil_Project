import numpy as np
import pandas as pd
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from topasToDose import getDosemap
from uniformity_fit import *


class bdsim_utils():
    def __init__(
        self,
        home_directory="/Applications/",
        no_of_threads="6",
        filename="lattice.gmad",
        file_directory='/Users/sabrinawang/Desktop/Cameron_Project/'
    ):
        file = open(file_directory + filename, "w")


        file.write('option, physicsList="em"; \n')
        

        self.home_directory = home_directory
        self.file_directory = file_directory
        self.filename = filename
        self.file = file
        self.Nquads = 0
        self.beamline_length = 0
        self.Ndrifts = 0
        self.aperture = 0.019  # m
        self.lattice = "L: line = ("

    def define_beam(self, particle="e-", energy=200, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1, sigma_E=0.5):
        file = self.file
        file.write(f'beam, particle="{particle}", energy={energy}*MeV, distrType="gauss", sigmaX={sigma_x}*mm, sigmaXp={sigma_xp}*mrad, sigmaY={sigma_y}*mm, sigmaYp={sigma_yp}*mrad, sigmaE={sigma_E/100};\n')

    def add_quadrupole(self, Lquad, k1):
        file = self.file
        self.Nquads += 1
        file.write('Q' + str(self.Nquads) + ': quadrupole, l=' + str(Lquad) + '*m, k1=' + str(k1) + ',beampipeRadius=' + str(self.aperture) + '*m;\n')
        self.beamline_length += Lquad
        self.lattice += "Q" + str(self.Nquads) + ", "

    def add_drift(self, Ldrift):
        file = self.file
        self.Ndrifts += 1

        file.write('D' + str(self.Ndrifts) + ': drift, l=' + str(Ldrift) + '*m, beampipeRadius=' + str(self.aperture) + '*m;\n')
        self.beamline_length += Ldrift
        self.lattice += "D" + str(self.Ndrifts) + ", "

    def add_gaussian_scatterer(self, max_thickness, radius, convolution_factor, N_slices, material, position, show_shape=False):
        file = self.file
        self.s2_Nslices = N_slices

        s2_sigma = radius / 2
        step = radius / (N_slices)
        x = np.arange(-(radius + step), 0, step=step)
        y = norm.pdf(x, 0, s2_sigma * convolution_factor)
        y = y - min(y)
        y_scaling_factor = max_thickness / max(y)
        y = y * y_scaling_factor
        if show_shape is True:
            plt.plot(np.append(x, -np.flip(x, 0)), np.append(y, np.flip(y)))
            plt.xlabel('r[mm]')
            plt.ylabel('h[mm]')
            plt.show()

        self.add_drift(position)

        for i in range(1, len(y)):
            L = y[i] - y[i - 1]
            HL = L / 2
            sname = "slice" + str(i)
            self.s2_length = 0

            if HL < 0:
                file.write('S2_' + sname + ':target, material="Vacuum", l=' + str(L) + '*mm, horizontalWidth=' + str(abs(x[i]) * 2) + '*mm, apertureType="circular";  \n')
            else:
                file.write('S2_' + sname + ':target, material="' + material + '", l=' + str(L) + '*mm, horizontalWidth=' + str(abs(x[i]) * 2) + '*mm, apertureType="circular";  \n')
            self.s2_length += L/1000
            self.lattice += "S2_" + sname + ", "
        self.beamline_length += self.s2_length

    def add_collimator(self, outer_radius, inner_radius, length):
        file = self.file

        file.write('collimator:ecol, material="G4_W", l=' + str(length) + '*m, xsize=' + str(inner_radius) + '*mm, ysize=' + str(inner_radius) + '*mm, apertureType="circular", horizontalWidth=' + str(outer_radius * 2) + '*mm;\n')
        self.beamline_length += length
        self.lattice += "collimator, "

    def make_lattice(self):
        file = self.file
        self.lattice = self.lattice[:-2]
        self.lattice += ");\n"

        file.write(self.lattice)
        file.write('use, L;\n')

    def add_tank(self, width, height, length, dose_depth, nx, ny, nz):
        '''
        x,y,z also xsizeOut and ysizeOut sizes in m
        '''
        file = self.file
        file.write(f'watertank:target, material="G4_WATER", xsize=0, ysize=0, l={length}*m, xsizeOut={width}*m, ysizeOut={height}*m, colour="blue";\n')
        self.lattice += "watertank, "
        self.beamline_length += length

        file.write('dose: scorer, type="depositeddose"; \n')
        file.write('eDep: scorer, type="depositedenergy"; \n')
        file.write(f'dose_scorer:scorermesh, scoreQuantity="eDep dose", referenceElement="watertank", geometryType="box", nx={nx},ny={ny},nz={nz}, xsize={width}, ysize={height}, zsize={dose_depth}, z={dose_depth}; \n')

    def add_samplers(self):
        file = self.file
        # Reversed quotes
        file.write('sample, all;\n')
   
if __name__ == "__main__":

    k1s = [8.23470106, -5.77519295, 12.30291714, -70.89257372]
    bdsim = bdsim_utils()
    bdsim.define_beam(particle="e-", energy=200, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1, sigma_E=0.5)
    for k1 in k1s:
        bdsim.add_quadrupole(0.3, k1)
        bdsim.add_drift(0.2)
    bdsim.add_gaussian_scatterer(3, 13, 1, 100, 'Al', 0.1, show_shape=False)
    bdsim.add_drift(2.5-0.1-3/1000)  # Adjust drift to reach 240mm for collimator so that tank at 2500mm -0.1 is the drift before guassian scatterer
    bdsim.add_collimator(275, 75, 0.1)
    bdsim.add_tank(0.3, 0.3, 0.5,0.01, 50,50,1)  # Add water tank 4500mm away, 300mm width, 300mm height, 500mm length
    bdsim.make_lattice()
    print("Beamline length:", bdsim.beamline_length)
    bdsim.add_samplers()
    bdsim.file.close()
    os.system(f"bdsim --file={bdsim.file_directory + bdsim.filename} --outfile=lattice_test --batch --ngenerate=10000")
    # os.system(f"bdsim --file={bdsim.file_directory + bdsim.filename}")

    print(bdsim.s2_length)
