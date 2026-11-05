'''Figure of RF-Track's multiple-scattering models (paper, Sec. "Scattering Materials in RF-Track").

198 MeV (kinetic) electron pencil beam through 4 mm of PEEK with the calibrated material
(design_optimisation.PEEK), compared with the TOPAS run of scattering_calibration.py:
  (a) projected-angle distributions with each RF-Track scattering model switched off in turn;
  (b) calibration curves: RF-Track / TOPAS core width versus the effective Z at the true Z/A,
      for each PEEK thickness of the pencil-beam calibration.

    python scattering_model_figures.py [output_dir [preview.png]]      # default: NIMB_optimiser/figs
'''
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

E_KIN = 198.0           # MeV
THICKNESS = 4.0         # mm
N = 200000
# TOPAS run with stoichiometric PEEK (C19H12O3, 1.32 g/cm3) from scattering_calibration.py
TOPAS_FILE = os.path.join(PROJECT_DIR, 'data', 'topas', f'calib_PeekTrue_{THICKNESS:g}mm.phsp')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # categorical slots 1-4, fixed order


def pencil_beam(n=N):
    import RF_Track as rft
    p = np.sqrt((E_KIN + rft.electronmass)**2 - rft.electronmass**2)
    return rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1, np.c_[np.zeros((n, 5)), np.full(n, p)])


def rft_angles(material, wentzel=True, fruehwirth=True, thickness=THICKNESS):
    '''projected angle x' (mrad) after one absorber, with the chosen scattering models'''
    import RF_Track as rft
    a = rft.Absorber(thickness / 1000, *material)
    a.disable_energy_straggling(); a.set_shape('circular', 1, 1)
    if not wentzel:
        a.disable_wentzel_model()
    if not fruehwirth:
        a.disable_fruehwirth_model()
    lat = rft.Lattice(); lat.append(a)
    rft.rng_set_seed(1)
    return lat.track(pencil_beam()).get_phase_space('%xp')[:, 0]


def topas_angles():
    d = np.loadtxt(TOPAS_FILE, usecols=(3, 7))
    return d[d[:, 1] == 11, 0] * 1000


def highland(x_mm, X0_cm):
    p = np.sqrt((E_KIN + 0.51099895)**2 - 0.51099895**2)
    t = x_mm / 10 / X0_cm
    return 13.6 / p * np.sqrt(t) * (1 + 0.038 * np.log(t)) * 1000


def stats(theta):
    from scipy.stats import kurtosis
    s = theta.std()
    return dict(rms=s, core=np.percentile(np.abs(theta), 68.27), tail=np.percentile(np.abs(theta), 99),
                kurt=kurtosis(theta), out3=np.mean(np.abs(theta) > 3 * s))


def main(out_dir, preview=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import design_optimisation as do
    peek = do.PEEK
    cases = [('default (both models on)', dict()), ('Wentzel off', dict(wentzel=False)),
             ('Frühwirth–Regler off', dict(fruehwirth=False)),
             ('both off', dict(wentzel=False, fruehwirth=False))]
    styles = ['-', '--', '-.', ':']
    topas = topas_angles(); ts = stats(topas)
    print(f'{E_KIN:g} MeV pencil beam, {THICKNESS:g} mm PEEK, RF-Track material {peek}; Highland {highland(THICKNESS, peek[0]):.2f} mrad')
    print(f'{"":30s} {"RMS":>7s} {"core68":>7s} {"tail99":>7s} {"kurtosis":>9s} {">3 RMS":>7s}   (angles in mrad)')
    print(f'{"TOPAS (Geant4)":30s} {ts["rms"]:7.2f} {ts["core"]:7.2f} {ts["tail"]:7.2f} {ts["kurt"]:9.2f} {ts["out3"]:7.4f}')
    thetas = {}
    for name, kw in cases:
        th = rft_angles(peek, **kw); thetas[name] = th; s = stats(th)
        print(f'{"RF-Track " + name:30s} {s["rms"]:7.2f} {s["core"]:7.2f} {s["tail"]:7.2f} {s["kurt"]:9.2f} {s["out3"]:7.4f}')

    # calibration curves: RF-Track / TOPAS core width versus Z at the material's true Z/A, per thickness
    import scattering_calibration as sc
    ZA = peek[1] / peek[2]
    Zs = np.round(np.arange(5.6, 7.41, 0.2), 2)
    thick = sc.SLABS['PeekTrue']
    ratio = {}
    for t in thick:
        ref = sc.pct(sc.topas_angles('PeekTrue', t))[0]
        ratio[t] = np.array([sc.pct(sc.rft_angles(t, (peek[0], Z, Z / ZA, peek[3], peek[4])))[0] / ref for Z in Zs])
        print(f'calibration {t:g} mm: best core Z {np.interp(1.0, ratio[t], Zs):.2f}')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    edges = np.linspace(-45, 45, 91); centres = 0.5 * (edges[1:] + edges[:-1])
    dens = lambda th: np.histogram(th, bins=edges, density=True)[0]
    ax1.plot(centres, dens(topas), color=INK, lw=2.4, label='TOPAS (Geant4)', zorder=5)
    for (name, _), col, ls in zip(cases, SERIES, styles):
        ax1.plot(centres, dens(thetas[name]), color=col, lw=1.6, ls=ls, label='RF-Track, ' + name)
    ax1.set_yscale('log'); ax1.set_ylim(1e-6, 0.12); ax1.set_xlim(-45, 45)
    ax1.set_xlabel(r'projected angle $\theta_x$ (mrad)'); ax1.set_ylabel('probability density (1/mrad)')
    ax1.grid(color=GRID, lw=0.6); ax1.legend(fontsize=6.3, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9, loc='lower center')
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, va='top', color=INK, fontweight='bold')

    cmap = plt.get_cmap('Blues'); cols = [cmap(v) for v in np.linspace(0.35, 0.95, len(thick))]
    for t, col in zip(thick, cols):
        ax2.plot(Zs, ratio[t], color=col, lw=1.6, marker='o', ms=3, label=f'{t:g} mm')
    ax2.axhline(1, color=INK, lw=1.2)
    ax2.axvspan(6.3, 6.5, color=GRID, alpha=0.8, lw=0)
    ax2.axvline(peek[1], color=INK2, lw=0.8)
    ax2.annotate(f'calibrated $Z={peek[1]:g}$', (peek[1], 0.945), xytext=(3, 0), textcoords='offset points',
                 fontsize=7, color=INK2, rotation=90, va='bottom')
    ax2.set_xlabel('effective $Z$ (true $Z/A$ of PEEK)'); ax2.set_ylabel('core width, RF-Track / TOPAS')
    ax2.set_ylim(0.93, 1.07); ax2.grid(color=GRID, lw=0.6)
    ax2.legend(title='PEEK thickness', title_fontsize=6.8, fontsize=6.5, frameon=False, loc='upper left', ncol=2)
    ax2.text(0.97, 0.03, '(b)', transform=ax2.transAxes, ha='right', color=INK, fontweight='bold')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'rft_scattering_models.pdf'))
    if preview:
        fig.savefig(preview, dpi=170)
    print('saved', os.path.join(out_dir, 'rft_scattering_models.pdf'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, 'NIMB_optimiser', 'figs'),
         sys.argv[2] if len(sys.argv) > 2 else None)
