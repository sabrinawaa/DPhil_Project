from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
from partrec_foil_plotting import partrec_foil_plotting


def main():
    # initialise script generator
    setup = partrec_gaussian_optimiser_utils()
    # generate Gaussian unit 200 MeV beam with 1 million particles (takes a while to run)
    setup.generate_phsp_beam(1, 1, 1, 1, 200, 0, 1000000)
    # add pre-scatterer to magnify beam
    setup.add_flat_scatterer(1, 'Tantalum')
    # define gaussian scatterer (here with 30mm depth, 20mm radius, situated 250mm downstream of first scatterer, composed of 200 slices)
    setup.add_gaussian_scatterer(
        30, 20, 1, 250, 'Aluminum', 200, show_shape=False)
    # add patient phase space scorerer at z=500m
    setup.add_patient(500)
    # run script
    setup.run_topas(view_setup=False)
    # initialise plotting class
    plotter = partrec_foil_plotting('data/topas/patient_beam.phsp')
    # plot transverse distributions and energy spectrum at patient
    plotter.show_transverse_beam(fov=50, col=50)


main()
