'''Shared dual-scatterer design / optimisation helpers.

Used by CLARA_design*.ipynb and optimise_CLEAR.ipynb. All scatterer
dimensions (s1_l, s2_width, s2_depth) are in mm, drifts are in m.
'''
import numpy as np
import RF_Track as rft
from scipy.stats import norm
from flatness import mask2d, central_mask, nearest_neighbor_test, merit_uniform_xy, loss_2gauss
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


# Materials as (radiation length [cm], Z, A [g/mol], density [g/cm^3], mean excitation energy [eV]).
# RF-Track's scattering model (Wentzel width + Fruehwirth tails, both on by default) depends on
# Z and A, not only on the radiation length, and treats unknown material names ('kapton',
# 'copper', 'aluminum', ...) as vacuum. The values below are effective ones, calibrated so that
# RF-Track reproduces the Geant4 (TOPAS, g4em-standard_opt4) core scattering width of a 198 MeV
# pencil beam (scattering_calibration.py, field checks in calibration_field_tests.py).
# Rule: only Z is fitted. Elements keep their real A; compounds, which have no single A, get
# A = Z / (real Z/A) so that their electron density (and collisional energy loss) stays real.
# For Al, air and Cu the fitted Z are correction factors for RF-Track's model, not material properties.
AL = (8.897, 14, 26.982, 2.7, 166)          # element, real A; true Z=13 scatters 8% too little
PEEK = (31.9, 6.4, 6.4 / (150 / 288.302), 1.32, -1)   # C19H12O3, real Z/A 0.520; core within 0.6%
AIR = (30390.0, 3.6, 3.6 / 0.4992, 0.0012048, 85.7)   # real Z/A 0.499; built-in 'air' scatters 36% too little
COPPER = (1.436, 32, 63.546, 8.96, 322.0)   # element, real A; true Z=29 scatters 15-26% too little
KAPTON = (28.57, 6.7, 6.7 / (196 / 382.33), 1.42, 79.6)   # C22H10N2O5, real Z/A 0.513; core within 0.4%
# previous values, to regenerate results made before / during the calibration
AL_UNCALIBRATED = (8.897, 13, 26.982, 2.7, 166)
PEEK_Z37 = (31.9, 37, 288.31, 1.32, -1)     # tuned by eye on whole fields; 4-6% too strong
PEEK_Z35 = (31.9, 35.2, 288.31, 1.32, -1)   # fitted at A = 288.31 (Z/A 0.12): core ok, collisional energy loss ~4x low
AIR_BUILTIN = 'air'
COPPER_REAL = (1.436, 29, 63.546, 8.96, 322.0)
KAPTON_Z65 = (28.57, 6.5, 12.67, 1.42, 79.6)


def absorber(length, material):
    '''RF-Track Absorber of `length` m from a material tuple or a built-in material name'''
    if isinstance(material, str):
        return rft.Absorber(length, material)
    return rft.Absorber(length, *material)


def add_s1(lattice, s1_l, material=AL):
    '''aluminium S1 foil, s1_l in mm'''
    S1 = absorber(s1_l/1000, material)
    S1.disable_energy_straggling()
    S1.set_shape ('circular', 1,1  )
    lattice.append(S1)


def add_drift(lattice, length, air=False, air_material=AIR):
    '''drift of `length` m, filled with air if air=True'''
    if air:
        drift = absorber(length, air_material)
        drift.disable_energy_straggling()
        drift.set_shape('circular', 1, 1)
        lattice.append(drift)
    else:
        lattice.append(rft.Drift(length))


def add_slab(lattice, thickness, material):
    '''full-aperture absorber of `thickness` mm, material given as (X0, Z, A, rho, I) or a name'''
    slab = absorber(thickness/1000, material)
    slab.disable_energy_straggling()
    slab.set_shape('circular', 1, 1)
    lattice.append(slab)


def add_s2(lattice, s2_width, s2_depth, N_slices=4, convolution_factor=1, material=PEEK):
    '''gaussian-profile S2 built from stacked circular slices, tip (smallest slice) upstream
    as in TOPAS; returns slice thicknesses (mm), base first'''
    s2_r,s2_l = get_s2_params(s2_width, s2_depth, N_slices, convolution_factor)
    for i in reversed(range(len(s2_l))):
        Slice = absorber(s2_l[i]/1000, material)
        Slice.disable_energy_straggling()
        Slice.set_shape ('circular',  abs(s2_r[i])/1000,abs(s2_r[i])/1000 )
        lattice.append(Slice)
    return s2_l


def build_scatterer_lattice(s2_width, s2_depth, s1_l=None, lattice=None, window=True,
                            drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5, air=False,
                            s2_material=PEEK, air_before_phsp=0.0, kapton=0.0, exit_window=(),
                            s1_material=AL, air_material=AIR):
    '''window -> drift -> [S1] -> drift -> S2 -> [kapton] -> drift -> [exit window] -> [air] -> plane,
    drifts filled with air if air=True.
    air_before_phsp (m): with air=False, only the last air_before_phsp of drift_to_phsp is air
    (e.g. vacuum up to an exit window, then air to the tank).
    kapton (mm): window directly on the S2 base.
    exit_window: (thickness [mm], material) layers placed directly upstream of the air section.
    All of these sit inside drift_to_phsp.
    s1_material, s2_material, air_material: material tuples (or built-in names), calibrated by default.
    drift_to_s2 runs to the S2 tip, drift_to_phsp from the S2 base to the phase-space plane.
    Pass `lattice` to append the scatterers to an existing beamline; zero-length drifts are skipped.'''
    if lattice is None:
        lattice = rft.Lattice()
    if window:
        add_window(lattice)
    if drift_to_s1 > 0:
        add_drift(lattice, drift_to_s1, air, air_material)
    if s1_l is not None:
        add_s1(lattice, s1_l, s1_material)
    if drift_to_s2 > 0:
        add_drift(lattice, drift_to_s2, air, air_material)
    add_s2(lattice, s2_width, s2_depth, material=s2_material)
    if kapton > 0:
        add_slab(lattice, kapton, KAPTON)
    window_length = sum(t for t, _ in exit_window) / 1000
    other_part = drift_to_phsp - kapton/1000 - window_length - air_before_phsp
    if other_part < 0:
        raise ValueError('kapton, exit_window and air_before_phsp do not fit in drift_to_phsp')
    if other_part > 0:
        add_drift(lattice, other_part, air, air_material)
    for thickness, material in exit_window:
        add_slab(lattice, thickness, material)
    if air_before_phsp > 0:
        add_drift(lattice, air_before_phsp, True, air_material)
    return lattice


MERITS = ('nn', 'andrea', '2gauss', 'med_flatness')


def central_slice_hists(M, fov, n_bins=30, slice_width=1):
    '''x / y histograms of the particles within +-slice_width (mm) of the y / x axis'''
    x = M[:,0] - np.mean(M[:,0])
    y = M[:,2] - np.mean(M[:,2])
    hist_x, edges_x = np.histogram(x[abs(y) < slice_width], bins=n_bins, range=[-fov, fov])
    hist_y, edges_y = np.histogram(y[abs(x) < slice_width], bins=n_bins, range=[-fov, fov])
    return hist_x, (edges_x[:-1] + edges_x[1:]) / 2, hist_y, (edges_y[:-1] + edges_y[1:]) / 2


X90_OVER_R60 = 0.80   # x90 / (radius containing 60% of the particles) of the CLEAR design (TOPAS slices: 20.1 / 25 mm)


def scatterer_merit(M, target_size, size_weight=0.5, verbose=True, merit='nn',
                    transmission_goal=0.998, fov=None, n_bins=30, slice_width=1,
                    nn_frac=None, gauss_ratio=1.1, nn_guard=True, nn_region=45):
    '''merit of the beam at the end of the scatterer lattice, lower is better:
    'nn'           CV of the distance to the k-th nearest neighbour for the central nn_region % of
                   particles by radius (45%: the plateau of the fields designed here, so the CV measures
                   the plateau and not the fall-off at the field edge),
                   k = nn_frac * number of particles (k=1 if nn_frac is None); with nn_guard the
                   neighbours are searched among all particles (no artificial boundary at the region
                   edge, a uniform field then gives the Poisson CV), nn_guard=False searches only
                   within the region
    'andrea'       percentiles of projections at many angles vs a uniform disk of radius target_size,
                   plus a transmission term (transmission_goal); size_weight is not used
    '2gauss'       fit of two gaussians to central x / y slices, x0/sigma -> gauss_ratio
    'med_flatness' (max-min)/(max+min) of the smoothed central 80% of central x / y slices
    'nn', '2gauss' and 'med_flatness' add size_weight * |x_max/target - 1| on the central 60%, i.e. they
    place the radius containing 60% of the particles at target_size (which also keeps 60% of the beam
    inside it). For a flat region of radius x90, pass target_size = x90 / X90_OVER_R60 (flat_radius_target).'''
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
        if nn_guard:
            shape_term = nearest_neighbor_test(M[:,0], M[:,2], k, query=central_mask(M[:,0], M[:,2], nn_region))[2]
        else:
            shape_term = nearest_neighbor_test(*mask2d(M[:,0], M[:,2], nn_region), k)[2]
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


def flat_radius_target(x90):
    '''target_size for the merit that gives a flat region (uniform radius x90) of the requested size,
    using the x90 / r60 ratio of the CLEAR design; check x90 of the result, the ratio depends slightly
    on the edge shape'''
    return x90 / X90_OVER_R60


def loss(params, B0, target_size=6, size_weight=0.5, lattice=None, window=True,
         drift_to_s1=0.5, drift_to_s2=0.5, drift_to_phsp=0.5, merit='nn', seed=None,
         air=False, s2_material=PEEK, air_before_phsp=0.0, kapton=0.0, exit_window=(),
         s1_material=AL, air_material=AIR, **merit_kwargs):  #16 target good for 10mm, 6 good for 5mm
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
                                      drift_to_phsp=drift_to_phsp, air=air, s2_material=s2_material,
                                      air_before_phsp=air_before_phsp, kapton=kapton, exit_window=exit_window,
                                      s1_material=s1_material, air_material=air_material)

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
