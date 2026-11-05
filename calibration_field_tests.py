'''Field-level check of the scattering-material calibration (scattering_calibration.py) on the
geometries of simulate_PDD_CLEAR.ipynb and CLARA_design.ipynb.

Each TOPAS geometry is rebuilt in RF-Track element by element and the positions of primary
electrons are scored at the same plane in both codes:
  pdd   : CLEAR main line (16 Sep 2026 Twiss, 195.2 MeV/c), S1 0.1 mm Al, small 4-slice S2, kapton
          holder, kapton + Cu-annulus vacuum window, 92 mm of air -> plane just before the collimator
          (the steel/W collimator, YAG and tank are left out: RF-Track cannot model them)
  clara : 250 MeV Gaussian beam in air, Be window, S1 1.2 mm Al, hard-coded 5.5 mm S2, kapton,
          Pb collimator (5 mm bore, a hard aperture in RF-Track), PMMA wall -> water surface
RF-Track is run with the current materials and with the calibrated ones.

    python calibration_field_tests.py topas pdd|pdd_bare|clara|clara_bare     # TOPAS reference (phase space)
    python calibration_field_tests.py compare pdd|pdd_bare|clara|clara_bare   # RF-Track cases + comparison
_bare variants keep only S1 and S2, in vacuum, with the same beam and scoring plane.
Outputs in calibration/field_tests/.
'''
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, 'calibration', 'field_tests')
os.makedirs(OUT_DIR, exist_ok=True)

N = 300000
# RF-Track materials: (X0 [cm], Z, A, rho [g/cm^3], I [eV])
AL_CURRENT = (8.897, 13, 26.982, 2.7, 166)
AL_CAL = (8.897, 14, 26.982, 2.7, 166)
PEEK_CURRENT = (31.9, 37, 288.31, 1.32, -1)
PEEK_ZA = (31.9, 6.6, 12.7, 1.32, -1)
PEEK_Z35 = (31.9, 35.2, 288.31, 1.32, -1)
PEEK_FIT = (31.9, 6.4, 6.4 / (150 / 288.302), 1.32, -1)   # fitted at the real Z/A (final)
KAPTON = (28.57, 6.7, 6.7 / (196 / 382.33), 1.42, 79.6)
COPPER = (1.436, 29, 63.546, 8.96, 322.0)
PMMA = (34.07, 6.5, 12.06, 1.19, 74.0)
# calibrated against TOPAS with scattering_calibration.py (core width, 198 MeV)
AIR_CAL = (30390.0, 3.6, 3.6 / 0.4992, 0.0012048, 85.7)   # built-in 'air' scatters ~36% too little
COPPER_CAL = (1.436, 32, 63.546, 8.96, 322.0)             # Z=29 scatters 15-26% too little
AIR = 'air'           # set per case in compare()
CU = COPPER
CASES = {   # (S1, S2[, air, copper]); kapton is always the Z/A fit (KAPTON)
    # RF-Track defaults: real Z for the elements, built-in air. PEEK and kapton have no default
    # description in RF-Track, so they keep their Z/A fits here as well
    'uncalibrated (Al Z=13, built-in air, Cu Z=29)': (AL_CURRENT, PEEK_FIT, 'air', COPPER),
    'calibrated (Al Z=14, air Z=3.6, Cu Z=32)': (AL_CAL, PEEK_FIT, AIR_CAL, COPPER_CAL),
    'calibrated, PEEK Z=35.2 (A=288.31)': (AL_CAL, PEEK_Z35, AIR_CAL, COPPER_CAL),
}

# --- geometry definitions (mm), copied from the notebooks ---------------------------------
PDD = dict(s1_pos=148, s1_l=0.1, gap_s1_s2=522,
           s2_l=[0.08719553, 0.18061171, 0.26891643, 0.26327633], s2_r=[1.4, 1.05, 0.7, 0.35],
           holder_gap=0.025, holder=0.025, window_at=1464.5, window=0.12, cu=0.005, cu_rmin=6,
           air_at=1465, air_to_plane=92)
CLARA = dict(E_kin=250, sig_x=np.sqrt(493 * 2e-3), sig_y=np.sqrt(72.5 * 13.8e-3), sig_xp=0.01, sig_yp=0.01,
             window_at=298.5, window=0.25, s1_pos=798.5, s1_l=1.2,
             s2_r=[1, 2, 2.9, 3.9], s2_l=[1.81, 1.85, 1.24, 0.60],   # tip first, as built in the notebook
             kapton=0.25, coll_gap=80, coll_len=74, coll_r=5, pmma_at=1796.5, pmma=1.0, plane_at=1798.5)


def pdd_beam(n=N):
    '''bunch at the end of the CLEAR main line, as in simulate_PDD_CLEAR cell 2'''
    import RF_Track as rft
    from CLEAR_line import get_beamline
    twiss = rft.Bunch6d_twiss()
    twiss.beta_x, twiss.beta_y, twiss.alpha_x, twiss.alpha_y = 9.37, 14.26, -4.12, 3.59
    twiss.emitt_x, twiss.emitt_y = 185.5, 114.6
    lattice = get_beamline("CLEAR_Beamline_Survey.txt", "CA.QFD0760", 'CA.DHJ0840', 195.2,
                           np.array([10.7, 28, 18, 0, 0.2, 0.1, 44.6, 77, 58.2, 0, 0]))
    return lattice.track(rft.Bunch6d_QR(rft.electronmass, 10 * rft.nC, -1, 195.2, twiss, n))


def clara_beam(n=N, seed=0):
    import RF_Track as rft
    rng = np.random.default_rng(seed)
    g = CLARA
    p = np.sqrt((g['E_kin'] + rft.electronmass)**2 - rft.electronmass**2)
    P = np.c_[rng.normal(0, g['sig_x'], n), rng.normal(0, g['sig_xp'], n), rng.normal(0, g['sig_y'], n),
              rng.normal(0, g['sig_yp'], n), np.zeros(n), np.full(n, p)]
    return rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1, P)


# --- TOPAS ---------------------------------------------------------------------------------
def write_scorer(setup, name, position, parent, material, half_width_m=0.4):
    f = setup.file
    f.write(f's:Ge/{name}/Type="TsBox"\ns:Ge/{name}/Parent="{parent}"\ns:Ge/{name}/Material="{material}"\n')
    f.write(f'd:Ge/{name}/HLX = {half_width_m} m\nd:Ge/{name}/HLY = {half_width_m} m\nd:Ge/{name}/HLZ = 0.001 mm\n')
    setup._write_transz(name, parent, position, 0.001)
    f.write(f's:Sc/plane/Quantity = "PhaseSpace"\ns:Sc/plane/Surface = "{name}/ZPlusSurface"\n')
    f.write('s:Sc/plane/OnlyIncludeParticlesOfGeneration = "Primary"\n')
    f.write(f's:Sc/plane/OutputFile = "fieldtest_{setup.tag}"\n')
    f.write('s:Sc/plane/OutputType = "ASCII"\ns:Sc/plane/IfOutputFileAlreadyExists = "Overwrite"\n')
    f.write('b:Sc/plane/OutputToConsole = "False"\nb:Ge/CheckForOverlaps = "False"\n')
    f.write('b:Ph/ListProcesses = "False"\nb:Ge/CheckForUnusedComponents = "False"\n')


def run_topas(which):
    '''which: pdd, pdd_bare, clara, clara_bare (bare = S1 + S2 only, vacuum everywhere)'''
    from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
    geometry, bare = which.split('_')[0], which.endswith('_bare')
    world = 'Vacuum' if (bare or geometry == 'pdd') else 'Air'
    setup = partrec_gaussian_optimiser_utils(world_material=world, input_filename='fieldtest_main.txt')
    setup.tag = which
    if geometry == 'pdd':
        import RF_Track as rft
        g = PDD
        R = pdd_beam().get_phase_space('%x %xp %y %yp %E %z')
        R[:, 4] -= rft.electronmass          # TOPAS phase space expects kinetic energy
        np.save(os.path.join(OUT_DIR, 'pdd_beam.npy'), R)
        base = os.path.join(setup.file_directory, 'fieldtest_pdd_beam')
        setup.export_phsp(R, base + '.phsp'); setup.write_header(R, base + '.header')
        setup.import_beam_topas(base, position=0)
        setup.add_flat_scatterer(g['s1_l'], 'Aluminum', g['s1_pos'])
        s2_start = g['s1_pos'] + g['s1_l'] + g['gap_s1_s2']
        for i, (t, r) in enumerate(zip(g['s2_l'], g['s2_r'])):
            setup.add_cylinder(f'S2_slice_{i}', t, 0, r, 'Peek', s2_start + sum(g['s2_l'][:i]))
        s2_end = s2_start + sum(g['s2_l'])
        if bare:
            write_scorer(setup, 'Plane', s2_end + g['air_at'] + g['air_to_plane'], 'World', 'Vacuum')
        else:
            setup.add_cylinder('kapton_holder', g['holder'], 0, 33, 'kapton', s2_end + g['holder_gap'])
            setup.add_cylinder('vacuum_window', g['window'], 0, 500, 'kapton', s2_end + g['window_at'])
            setup.add_cylinder('vacuum_window_coating', g['cu'], g['cu_rmin'], 500, 'G4_Cu', s2_end + g['window_at'] + g['window'])
            setup.add_box('air', 900, 200, 200, 'Air', s2_end + g['air_at'])
            write_scorer(setup, 'Plane', g['air_to_plane'], 'air', 'Air', half_width_m=0.09)
    else:
        g = CLARA
        setup.generate_phsp_beam(g['sig_x'], g['sig_y'], g['sig_xp'], g['sig_yp'], g['E_kin'], 0, N)
        if not bare:
            setup.add_cylinder('window', g['window'], 0, 500, 'G4_Be', g['window_at'])
        setup.add_flat_scatterer(g['s1_l'], 'Aluminum', g['s1_pos'])
        s2_pos = g['s1_pos'] + 500
        for i, (t, r) in enumerate(zip(g['s2_l'], g['s2_r'])):
            setup.add_cylinder(f'S2_slice_{i}', t, 0, r, 'PEEK', s2_pos + sum(g['s2_l'][:i]))
        s2_end = s2_pos + sum(g['s2_l'])
        if not bare:
            setup.add_cylinder('kapton', g['kapton'], 0, 33, 'kapton', s2_end)
            setup.add_cylinder('collimator', g['coll_len'], g['coll_r'], 50, 'G4_Pb', s2_end + g['coll_gap'])
            setup.add_cylinder('tank_edge', g['pmma'], 0, 100, 'PMMA', g['pmma_at'])
        write_scorer(setup, 'Plane', g['plane_at'], 'World', world)
    setup.mute_terminal_output()
    setup.run_topas('fieldtest_main.txt')


# --- RF-Track ------------------------------------------------------------------------------
def slab(lattice, thickness_mm, material, radius_mm=1000):
    import RF_Track as rft
    a = rft.Absorber(thickness_mm / 1000, *material) if isinstance(material, tuple) else rft.Absorber(thickness_mm / 1000, material)
    a.disable_energy_straggling()
    a.set_shape('circular', radius_mm / 1000, radius_mm / 1000)
    lattice.append(a)


def drift(lattice, length_mm, air=False, aperture_mm=None):
    import RF_Track as rft
    if air:
        d = rft.Absorber(length_mm / 1000, AIR) if isinstance(AIR, str) else rft.Absorber(length_mm / 1000, *AIR)
        d.disable_energy_straggling(); d.set_shape('circular', 1, 1)
    else:
        d = rft.Drift(length_mm / 1000)
    if aperture_mm is not None:
        d.set_aperture(aperture_mm / 1000, aperture_mm / 1000, 'circular')
    lattice.append(d)


def track(B, build):
    import RF_Track as rft
    lat = rft.Lattice(); build(lat)
    return lat.track(B)


def annulus(B, thickness_mm, material, r_min_mm):
    '''material only at r > r_min (RF-Track has no annular absorber): split, track, merge'''
    import RF_Track as rft
    P = B.get_phase_space('%x %xp %y %yp %t %Pc')
    outer = np.hypot(P[:, 0], P[:, 2]) > r_min_mm
    Bo = track(rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1, P[outer]), lambda l: slab(l, thickness_mm, material))
    return rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1,
                       np.vstack([P[~outer], Bo.get_phase_space('%x %xp %y %yp %t %Pc')]))


def rft_positions(which, al, peek):
    import RF_Track as rft
    rft.rng_set_seed(1)
    geometry, bare = which.split('_')[0], which.endswith('_bare')
    if geometry == 'pdd':
        g = PDD
        R = np.load(os.path.join(OUT_DIR, 'pdd_beam.npy'))            # identical beam to the TOPAS run
        p = np.sqrt((R[:, 4] + rft.electronmass)**2 - rft.electronmass**2)
        B = rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1, np.c_[R[:, 0], R[:, 1], R[:, 2], R[:, 3], np.zeros(len(R)), p])
        def scatterers(l):
            drift(l, g['s1_pos']); slab(l, g['s1_l'], al); drift(l, g['gap_s1_s2'])
            for t, r in zip(g['s2_l'], g['s2_r']):
                slab(l, t, peek, r)
        if bare:
            B = track(B, lambda l: (scatterers(l), drift(l, g['air_at'] + g['air_to_plane'])))
        else:
            def to_window(l):
                scatterers(l)
                drift(l, g['holder_gap']); slab(l, g['holder'], KAPTON, 33)
                drift(l, g['window_at'] - g['holder_gap'] - g['holder']); slab(l, g['window'], KAPTON)
            B = annulus(track(B, to_window), g['cu'], CU, g['cu_rmin'])
            B = track(B, lambda l: (drift(l, g['air_at'] - g['window_at'] - g['window'] - g['cu']),
                                    drift(l, g['air_to_plane'], air=True)))
        n0 = len(R)
    else:
        g = CLARA
        B0 = clara_beam(); n0 = N
        air = not bare
        s2_end = g['s1_pos'] + 500 + sum(g['s2_l'])
        def build(l):
            if bare:
                drift(l, g['s1_pos'])
            else:
                drift(l, g['window_at'], air=True); slab(l, g['window'], 'beryllium')
                drift(l, g['s1_pos'] - g['window_at'] - g['window'], air=True)
            slab(l, g['s1_l'], al)
            drift(l, 500 - g['s1_l'], air=air)
            for t, r in zip(g['s2_l'], g['s2_r']):
                slab(l, t, peek, r)
            if bare:
                drift(l, g['plane_at'] - s2_end)
                return
            slab(l, g['kapton'], KAPTON, 33)
            drift(l, g['coll_gap'] - g['kapton'], air=True)
            drift(l, g['coll_len'] / 2, aperture_mm=g['coll_r']); drift(l, g['coll_len'] / 2, aperture_mm=g['coll_r'])
            drift(l, g['pmma_at'] - (s2_end + g['coll_gap'] + g['coll_len']), air=True); slab(l, g['pmma'], PMMA, 100)
            drift(l, g['plane_at'] - g['pmma_at'] - g['pmma'], air=True)
        B = track(B0, build)
    M = B.get_phase_space('%x %y')
    return M[:, 0], M[:, 1], n0


def topas_positions(which):
    d = np.loadtxt(os.path.join(PROJECT_DIR, 'data', 'topas', f'fieldtest_{which}.phsp'), usecols=(0, 1, 5, 7))
    d = d[d[:, 3] == 11]
    if which == 'clara':
        # RF-Track's collimator is a hard aperture: drop TOPAS primaries that went through or scattered
        # off the lead (they lose energy there); S1/S2/windows/air cost only a few MeV
        d = d[d[:, 2] > 0.9 * CLARA['E_kin']]
    n0 = len(np.load(os.path.join(OUT_DIR, 'pdd_beam.npy'))) if which.startswith('pdd') else N
    return d[:, 0] * 10, d[:, 1] * 10, n0     # cm -> mm


RHO_EDGES = np.linspace(0, 2, 41)          # radius in units of the field's own x50
RHO_C = 0.5 * (RHO_EDGES[1:] + RHO_EDGES[:-1])
from flatness import FIELD_LEVELS            # x90, x50, x20


from flatness import field_radii   # super-Gaussian x_100F of a radial profile (shared with the optimiser)


def twogauss_ratio(rc, ring_mean, ring_area, r_fit, full=False):
    """|mu/sigma| of the two-Gaussian function of the paper (Sec. Two-Gaussian Fit) fitted to the radial
    profile within r < r_fit, mirrored to negative r so that every particle contributes. Fitted within
    x90 it describes the plateau (dip and shoulders); within x50 it also includes the start of the edge.
    Rings are weighted by their area, as in field_radii. full=True also returns the fit parameters (A, mu, sigma)."""
    from scipy.optimize import curve_fit
    from flatness import sum_2gaussians
    ok = (ring_area > 0) & (rc < r_fit)
    r = np.r_[-rc[ok][::-1], rc[ok]]; y = np.r_[ring_mean[ok][::-1], ring_mean[ok]]
    w = np.r_[ring_area[ok][::-1], ring_area[ok]]
    (A, mu, sig), _ = curve_fit(sum_2gaussians, r, y, p0=[0.6 * np.median(ring_mean[ok]), 0.4 * r_fit, 0.4 * r_fit],
                                sigma=1 / np.sqrt(w), maxfev=50000)
    return (abs(mu) / abs(sig), (A, mu, sig)) if full else abs(mu) / abs(sig)


def particle_profile(r, r_max, n_rings=80):
    """radial density of particle radii r (per mm^2, per particle given), its field level N (fitted
    super-Gaussian peak) and x_100F, and the shape: density vs r/x50 normalised to N"""
    edges = np.linspace(0, r_max, n_rings + 1); rc = 0.5 * (edges[1:] + edges[:-1]); area = np.pi * np.diff(edges**2)
    dens = np.histogram(r, bins=edges)[0] / area
    N, x = field_radii(rc, dens, area)
    x['mu/sigma x90'] = twogauss_ratio(rc, dens, area, x[0.9])
    x['mu/sigma x50'] = twogauss_ratio(rc, dens, area, x[0.5])
    rho_edges = RHO_EDGES * x[0.5]
    shape = np.histogram(r, bins=rho_edges)[0] / (np.pi * np.diff(rho_edges**2)) / N
    return N, x, shape


def shape_metrics(shape, x):
    """centre/ring: density at r < 0.3 x50 over 0.5-0.9 x50 (<1 dip/horns, ~1 flat, >1 peaked);
    flatness: (max-min)/(max+min) of the profile inside x90; edge: x90/x50 and x20/x50 (sharpness)"""
    core, ring, field = RHO_C < 0.3, (RHO_C > 0.5) & (RHO_C < 0.9), RHO_C < x[0.9] / x[0.5]
    return (shape[core].mean() / shape[ring].mean(),
            (shape[field].max() - shape[field].min()) / (shape[field].max() + shape[field].min()),
            x[0.9] / x[0.5], x[0.2] / x[0.5])


def compare(which):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tx, ty, tn = topas_positions(which)
    tr = np.hypot(tx - np.mean(tx), ty - np.mean(ty))
    r_max = np.percentile(tr, 97) * 1.05
    tN, tX, t_shape = particle_profile(tr, r_max); t_frac = len(tr) / tn; t_m = shape_metrics(t_shape, tX)
    fit = RHO_C < 1.5
    # comparison floor: two independent samples of the TOPAS size (halves differ by 2x one sample's noise,
    # a comparison of two full samples by sqrt(2)x)
    half = len(tr) // 2
    floor = np.sqrt(np.mean((particle_profile(tr[:half], r_max)[2] - particle_profile(tr[half:], r_max)[2])[fit]**2)) / np.sqrt(2)
    # statistical uncertainty of the full TOPAS sample: half the difference between two independent halves
    hA, hB = particle_profile(tr[:half], r_max)[1], particle_profile(tr[half:], r_max)[1]
    t_err = {k: abs(hA[k] - hB[k]) / 2 for k in hA}
    print(f"  TOPAS stat. uncertainty (halves): x90 {t_err[0.9]:.2f}  x50 {t_err[0.5]:.2f}  x20 {t_err[0.2]:.2f}  "
          f"mu/sigma within x90 {t_err['mu/sigma x90']:.3f}, within x50 {t_err['mu/sigma x50']:.3f}")
    print(f'\n=== {which}  (size: x90/x50/x20 = super-Gaussian x_100F of the radial profile; '
          f'shape: profile vs r/x50)')
    print(f'  {"":38s} {"x90":>6s} {"x50":>6s} {"x20":>6s} {"scored":>6s} | {"ctr/ring":>8s} {"flatness":>8s} '
          f'{"x90/x50":>7s} {"x20/x50":>7s} {"shape RMS":>9s} {"mu/s<x90":>8s} {"mu/s<x50":>8s}')
    def line(name, x, frac, m, rms):
        print(f'  {name:38s} {x[0.9]:6.2f} {x[0.5]:6.2f} {x[0.2]:6.2f} {frac:6.3f} | {m[0]:8.3f} {m[1]:8.3f} '
              f'{m[2]:7.3f} {m[3]:7.3f} {rms:>9s} {x["mu/sigma x90"]:8.3f} {x["mu/sigma x50"]:8.3f}')
    line('TOPAS', tX, t_frac, t_m, f'(floor {floor:.3f})')
    # paper Table 2: super-Gaussian / two-Gaussian fits to the central x and y slices (flatness.slice_fit)
    from flatness import slice_metrics, particle_slices
    def slice_line(name, xs, ys, err=False):
        m, e = slice_metrics(particle_slices(xs, ys, 15, 0.25))
        s_ = f"  [slices] {name:30s} x90 {m[0.9]:5.2f} x50 {m[0.5]:5.2f} x20 {m[0.2]:5.2f} mu/s<x90 {m['mu/sigma x90']:.3f} mu/s<x50 {m['mu/sigma x50']:.3f}"
        print(s_ + (f"   (+- x50 {e[0.5]:.2f}, x90 {e[0.9]:.2f}, mu/s<x90 {e['mu/sigma x90']:.3f}, mu/s<x50 {e['mu/sigma x50']:.3f})" if err else ''))
    slice_line('TOPAS', tx, ty, err=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[1].plot(RHO_C, t_shape, 'k-', lw=2.5, label='TOPAS')
    edges = np.linspace(0, r_max, 81); rc = 0.5 * (edges[1:] + edges[:-1]); area = np.pi * np.diff(edges**2)
    scale = tN / tn                                     # TOPAS field level per incident electron
    axs[0].plot(rc, np.histogram(tr, bins=edges)[0] / area / tn / scale, 'k-', lw=2.5, label='TOPAS')
    global AIR, CU
    for name, case in CASES.items():
        al, peek = case[:2]
        AIR, CU = (case[2], case[3]) if len(case) > 2 else ('air', COPPER)
        x, y, n0 = rft_positions(which, al, peek)
        r = np.hypot(x - np.mean(x), y - np.mean(y))
        N, X, shape = particle_profile(r, r_max)
        rms = np.sqrt(np.mean((shape - t_shape)[fit]**2))
        line(name, X, len(r) / n0, shape_metrics(shape, X), f'{rms:.3f}')
        slice_line(name, x, y)
        axs[1].plot(RHO_C, shape, label=name)
        axs[0].plot(rc, np.histogram(r, bins=edges)[0] / area / n0 / scale, label=name)
    axs[0].set_xlabel('r (mm)'); axs[0].set_ylabel('density per incident electron (TOPAS fitted peak = 1)')
    axs[0].set_title(f'{which}: absolute (size + shape)')
    axs[1].set_xlabel('r / x50'); axs[1].set_ylabel('density (fitted peak = 1)'); axs[1].set_title(f'{which}: shape only')
    for ax in axs: ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, f'{which}_profiles.png'), dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    mode, which = sys.argv[1], sys.argv[2]
    {'topas': run_topas, 'compare': compare}[mode](which)
