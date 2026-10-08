from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from partrec_foil_plotting import partrec_foil_plotting
from topasToDose import getDosemap
import numpy as np

def main():
    n_particles = 10000
    for dose_depth in [100]:
        # initialise script generator
        setup = partrec_gaussian_optimiser_utils()
        # generate Gaussian unit 200 MeV beam with 1 million particles (takes a while to run)
        setup.generate_phsp_beam(1, 1, 1, 1, 200, 0, n_particles)
        # add pre-scatterer to magnify beam
        setup.add_flat_scatterer(1, 'Tantalum')
        # define gaussian scatterer (here with 30mm depth, 20mm radius, situated 200mm downstream of first scatterer, composed of 250 slices)
        setup.add_gaussian_scatterer(
            22, 10, 1, 250, 'Aluminum', 200, show_shape=False)
        #add water phantom 500mm away, depth 0.1 mm, 10 bins in x, y, 1 bin in z
        setup.add_tank_bins(500,dose_depth,50,50,1)
        # run script
        setup.run_topas(view_setup=False)
        # initialise plotting class
        doseMap = getDosemap("data/topas/DoseAtTank"+str(dose_depth)+".csv",n_particles, dose_depth,"ini_trial_", plot = True)
        print(f"Scaled Dose Map Shape: {doseMap.shape}")
    
            


main()
