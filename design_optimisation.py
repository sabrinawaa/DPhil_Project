'''Shared dual-scatterer design / optimisation helpers.

Used by CLARA_design*.ipynb and optimise_CLEAR.ipynb. All scatterer
dimensions (s1_l, s2_width, s2_depth) are in mm, drifts are in m.
'''
import numpy as np
import RF_Track as rft
from scipy.stats import norm
from flatness import mask2d, nearest_neighbor_test, merit_uniform_xy, loss_2gauss
from uniformity_fit import flatness


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


# PEEK as (radiation length [cm], Z, A [g/mol], density [g/cm^3], mean excitation energy [eV]).
# RF-Track's scattering model (Wentzel width + Fruehwirth tails, both on by default) depends on
# Z and A, not only on the radiation length; Z=37 was tuned to match the TOPAS beam shape.
PEEK = (31.9, 37, 288.31, 1.32, -1)


def add_drift(lattice, length, air=False):
    '''drift of `length` m, filled with air if air=True'''
    if air:
        drift = rft.Absorber(length, 'air')
        drift.disable_energy_straggling()
        drift.set_shape('circular', 1, 1)
        lattice.append(drift)
    else:
        lattice.append(rft.Drift(length))


def add_s2(lattice, s2_width, s2_depth, N_slices=4, convolution_factor=1, material=PEEK):
    '''gaussian-profile S2 built from stacked circular slices, tip (smallest slice) upstream
    as in TOPAS; returns slice thicknesses (mm), base first'''
    s2_r,s2_l = get_s2_params(s2_width, s2_depth, N_slices, convolution_factor)
    for i in reversed(range(len(s2_l))):
        Slice = rft.Absorber(s2_l[i]/1000, *material)
        Slice.disable_energy_straggling()
        Slice.set_shape ('circular',  abs(s2_r[i])/1000,abs(s2_r[i])/1000 )
        lattice.append(Slice)
    return s2_l


def build_scatterer_lattice(s2_width, s2_depth, s1_l=None, lattice=None, window=True,
                            drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5, air=False,
                            s2_material=PEEK):
    '''window -> drift -> [S1] -> drift -> S2 -> drift, drifts filled with air if air=True.
    drift_to_s2 runs to the S2 tip, drift_to_phsp from the S2 base.
    Pass `lattice` to append the scatterers to an existing beamline; zero-length drifts are skipped.'''
    if lattice is None:
        lattice = rft.Lattice()
    if window:
        add_window(lattice)
    if drift_to_s1 > 0:
        add_drift(lattice, drift_to_s1, air)
    if s1_l is not None:
        add_s1(lattice, s1_l)
    if drift_to_s2 > 0:
        add_drift(lattice, drift_to_s2, air)
    add_s2(lattice, s2_width, s2_depth, material=s2_material)
    if drift_to_phsp > 0:
        add_drift(lattice, drift_to_phsp, air)
    return lattice


MERITS = ('nn', 'andrea', '2gauss', 'med_flatness')


def central_slice_hists(M, fov, n_bins=30, slice_width=1):
    '''x / y histograms of the particles within +-slice_width (mm) of the y / x axis'''
    x = M[:,0] - np.mean(M[:,0])
    y = M[:,2] - np.mean(M[:,2])
    hist_x, edges_x = np.histogram(x[abs(y) < slice_width], bins=n_bins, range=[-fov, fov])
    hist_y, edges_y = np.histogram(y[abs(x) < slice_width], bins=n_bins, range=[-fov, fov])
    return hist_x, (edges_x[:-1] + edges_x[1:]) / 2, hist_y, (edges_y[:-1] + edges_y[1:]) / 2


def scatterer_merit(M, target_size, size_weight=0.5, verbose=True, merit='nn',
                    transmission_goal=0.998, fov=None, n_bins=30, slice_width=1,
                    nn_frac=None, gauss_ratio=1.1):
    '''merit of the beam at the end of the scatterer lattice, lower is better:
    'nn'           CV of the distance to the k-th nearest neighbour in the central 60% of particles,
                   k = nn_frac * number of particles (k=1 if nn_frac is None)
    'andrea'       percentiles of projections at many angles vs a uniform disk of radius target_size,
                   plus a transmission term (transmission_goal); size_weight is not used
    '2gauss'       fit of two gaussians to central x / y slices, x0/sigma -> gauss_ratio
    'med_flatness' (max-min)/(max+min) of the smoothed central 80% of central x / y slices
    'nn', '2gauss' and 'med_flatness' add size_weight * |x_max/target - 1| on the central 60%'''
    if merit not in MERITS:
        raise ValueError(f'merit must be one of {MERITS}, got {merit!r}')

    if merit == 'andrea':
        loss = merit_uniform_xy(M[:,0], M[:,2], target_size, transmission_goal, verbose=verbose)
        if verbose:
            print('andrea merit:', loss)
        return loss

    masked_x, masked_y = mask2d(M[:,0],M[:,2])
    size_term = abs(masked_x.max()/target_size-1)

    if merit == 'nn':
        k = 1 if nn_frac is None else max(1, int(round(nn_frac * len(M))))
        shape_term = nearest_neighbor_test(masked_x,masked_y,k)[2]
    else:
        hist_x, centres_x, hist_y, centres_y = central_slice_hists(
            M, fov if fov is not None else 1.5*target_size, n_bins, slice_width)
        try:
            if merit == '2gauss':
                shape_term = loss_2gauss(hist_x, centres_x, hist_y, centres_y, gauss_ratio)
            else:
                shape_term = flatness(hist_x) + flatness(hist_y)
        except (RuntimeError, ValueError, ZeroDivisionError):  # failed fit / empty slice
            shape_term = 1e3

    loss = shape_term + size_weight * size_term
    if verbose:
        print(merit, 'contribution:', shape_term)
        print('size contribution:', size_term)
        print('x range:',max(masked_x),min(masked_x),'y range:', max(masked_y),min(masked_y))
    return loss


def loss(params, B0, target_size=6, size_weight=0.5, lattice=None, window=True,
         drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5, merit='nn', seed=None,
         air=False, s2_material=PEEK, **merit_kwargs):  #16 target good for 10mm, 6 good for 5mm
    '''params = (s2_width, s2_depth) or (s1_l, s2_width, s2_depth), dimensions in mm.
    merit is one of MERITS, see scatterer_merit; merit_kwargs (transmission_goal, nn_frac, ...) go to it.
    seed: if given, RF-Track's RNG is reseeded before tracking so every call sees the same scattering
    draw (common random numbers); differences between nearby params are then far less noisy.
    Updates the module-level best_seen.'''
    if len(params) == 3:
        s1_l, s2_width, s2_depth = params
    else:
        s1_l = None
        s2_width, s2_depth = params

    lattice = build_scatterer_lattice(s2_width, s2_depth, s1_l, lattice=lattice, window=window,
                                      drift_to_s1=drift_to_s1, drift_to_s2=drift_to_s2,
                                      drift_to_phsp=drift_to_phsp, air=air, s2_material=s2_material)

    if seed is not None:
        rft.rng_set_seed(seed)
    B1 = lattice.track(B0)
    M = B1.get_phase_space('%x %xp %y %yp %E %z')

    loss = scatterer_merit(M, target_size, size_weight, merit=merit, **merit_kwargs)
    if loss < best_seen[0]:
        best_seen[0] = loss
        best_seen[1] = np.array(params).copy()

    print('loss:', loss, 's1_l:', s1_l, 's2_2r:', s2_width, 's2_l:', s2_depth)
    return loss
