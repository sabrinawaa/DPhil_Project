
import pybdsim
import pyg4ometry.geant4 as g4
from pyg4ometry import gdml
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from uniformity_fit import *
import os
os.system("export BDSIM=/opt/anaconda3/envs/RFT_bdsim/")  
os.system("export ROOT_INCLUDE_PATH=$BDSIM/include/bdsim/:$BDSIM/include/bdsim/analysis/:$BDSIM/include/bdsim/parser/")

# n_particles = 1e5 #1e5
# k1_1, k1_2, k1_3, k1_4 = 8.23470106,  -5.77519295,  12.30291714, -70.89257372 #<75
# quadlattice = RF_track_utils(200, [k1_1, k1_2, k1_3, k1_4])
# # # quadlattice.add_drift(2.5)  # Add drift after quadrupoles to reach water phantom
# R = quadlattice.track_bunch(n_particles, saveparams=True, E_deviation=0.5, sigma_x=1, sigma_xp=1, sigma_y=1, sigma_yp=1)
# # quadlattice.plot_phsp()




d=pybdsim.DataPandas.BDSIMOutput("/Users/sabrinawang/Desktop/Cameron_Project/testrun.root")

dose_histo = d.get_histo3d('dose_scorer-dose')
xmin = dose_histo.xedges.min()
xmax = dose_histo.xedges.max()

ymin = dose_histo.yedges.min()
ymax = dose_histo.yedges.max()

zmin = dose_histo.zedges.min()
zmax = dose_histo.zedges.max()

plt.imshow(dose_histo.contents[:,:,0],extent=[xmin,xmax,ymin,ymax]) #can later define more z bins to get different depths dont need to rereun.
plt.colorbar()

dose_depth = 100 #in mm
x,y,doseMap = dose_histo.xcentres,dose_histo.ycentres,dose_histo.contents[:,:,0]


p0=[0.2, np.max(x)//2, np.max(y)//2, 60, 60, 6]
#curvefit 2d histogram to supergaussian
# Perform the optimization
result = minimize(MSE, p0, args=(x, y, doseMap), method='L-BFGS-B')
fit_params = result.x

# Calculate the fitted super-Gaussian
fitted_map = supergaussian(x, y, *fit_params)
sig, P = fit_params[3], fit_params[5]
r_90 = r90(sig,P)
plotDoseMap(x, y, doseMap,fitted_map,P,sig,r_90, dose_depth, "bdsim_dose")