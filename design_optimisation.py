'''Shared dual-scatterer design / optimisation helpers.

Used by CLARA_design*.ipynb and optimise_CLEAR.ipynb. All scatterer
dimensions (s1_l, s2_width, s2_depth) are in mm, drifts are in m.
'''
import numpy as np
import RF_Track as rft
from scipy.stats import norm
from flatness import mask2d, nearest_neighbor_test


# best (loss, params) seen during an optimisation; reset with reset_best_seen()
best_seen = [np.inf, None]


def reset_best_seen():
    # mutate in place so `from design_optimisation import best_seen` stays valid
    best_seen[0] = np.inf
    best_seen[1] = None


def get_s2_params(max_radius, max_thickness, N_slices, convolution_factor):
    s2_sigma = max_radius / 2
    step = max_radius / N_slices

    # Radial positions
    x = np.arange(-(max_radius + step), 0, step=step)

    # Thickness profile
    y = norm.pdf(x, 0, s2_sigma * convolution_factor)
    y = y - np.min(y)
    y *= max_thickness / np.max(y)
    slice_widths = np.diff(y) #thicknesses of each slice in L
    slice_radii = np.abs(x[1:])   # Corresponding radii (height)

    return slice_radii, slice_widths


def add_window(lattice):
    window = rft.Absorber(250e-6,'beryllium')
    window.disable_energy_straggling()
    window.set_shape ('circular', 0.5,0.5 )
    lattice.append(window)


def add_s1(lattice, s1_l):
    '''aluminium S1 foil, s1_l in mm'''
    S1 = rft.Absorber(s1_l/1000,8.897, 13,26.982,2.7, 166)
    S1.disable_energy_straggling()
    S1.set_shape ('circular', 1,1  )
    lattice.append(S1)


def add_s2(lattice, s2_width, s2_depth, N_slices=4, convolution_factor=1):
    '''gaussian-profile S2 built from stacked circular slices, returns slice thicknesses (mm)'''
    s2_r,s2_l = get_s2_params(s2_width, s2_depth, N_slices, convolution_factor)
    for i in range(len(s2_l)):
        Slice = rft.Absorber(s2_l[i]/1000,31.9, 37, 288.31,1.32,-1)
        Slice.disable_energy_straggling()
        Slice.set_shape ('circular',  abs(s2_r[i])/1000,abs(s2_r[i])/1000 )
        lattice.append(Slice)
    return s2_l


def build_scatterer_lattice(s2_width, s2_depth, s1_l=None, lattice=None, window=True,
                            drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5):
    '''window -> drift -> [S1] -> drift -> S2 -> drift.
    Pass `lattice` to append the scatterers to an existing beamline; zero-length drifts are skipped.'''
    if lattice is None:
        lattice = rft.Lattice()
    if window:
        add_window(lattice)
    if drift_to_s1 > 0:
        lattice.append(rft.Drift(drift_to_s1))
    if s1_l is not None:
        add_s1(lattice, s1_l)
    if drift_to_s2 > 0:
        lattice.append(rft.Drift(drift_to_s2))
    add_s2(lattice, s2_width, s2_depth)
    if drift_to_phsp > 0:
        lattice.append(rft.Drift(drift_to_phsp))
    return lattice


def scatterer_merit(M, target_size, size_weight=0.5, verbose=True):
    '''nearest-neighbour CV of the central 60% of particles + size_weight * |x_max/target - 1|'''
    masked_x, masked_y = mask2d(M[:,0],M[:,2])
    nn_cv = nearest_neighbor_test(masked_x,masked_y)[2]
    size_term = abs(masked_x.max()/target_size-1)
    loss = nn_cv + size_weight * size_term
    if verbose:
        print('nn contribution:', nn_cv)
        print('size contribution:', size_term)
        print('x range:',max(masked_x),min(masked_x),'y range:', max(masked_y),min(masked_y))
    return loss


def loss(params, B0, target_size=6, size_weight=0.5, lattice=None, window=True,
         drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5):  #16 target good for 10mm, 6 good for 5mm
    '''params = (s2_width, s2_depth) or (s1_l, s2_width, s2_depth), dimensions in mm.
    Updates the module-level best_seen.'''
    if len(params) == 3:
        s1_l, s2_width, s2_depth = params
    else:
        s1_l = None
        s2_width, s2_depth = params

    lattice = build_scatterer_lattice(s2_width, s2_depth, s1_l, lattice=lattice, window=window,
                                      drift_to_s1=drift_to_s1, drift_to_s2=drift_to_s2,
                                      drift_to_phsp=drift_to_phsp)

    B1 = lattice.track(B0)
    M = B1.get_phase_space('%x %xp %y %yp %E %z')

    loss = scatterer_merit(M, target_size, size_weight)
    if loss < best_seen[0]:
        best_seen[0] = loss
        best_seen[1] = np.array(params).copy()

    print('loss:', loss, 's1_l:', s1_l, 's2_2r:', s2_width, 's2_l:', s2_depth)
    return loss
