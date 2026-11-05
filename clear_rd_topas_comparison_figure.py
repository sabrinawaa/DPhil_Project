'''RF-Track vs TOPAS for the CLEAR second-beamline design (CLEAR_RD.ipynb).

RF-Track: calibrated materials (design_optimisation defaults), the TOPAS source of CLEAR_RD cell 13
(Gaussian, 1 mm, 0.001 mrad, 198 MeV kinetic, 43 mm before S1), the same scatterers, kapton, vacuum
window and 0.3 m of air. TOPAS: the dose map written by CLEAR_RD cell 13 (dose in the first
1 mm of water, 0.8 mm bins over +-40 mm). Field size and shape come from super-Gaussian and two-Gaussian fits
to the central x and y slices (flatness.slice_fit); each map is normalised to the mean fitted peak.

    python clear_rd_topas_comparison_figure.py [s1_l s2_width s2_depth] [--out DIR]
'''
import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

DEFAULT_PARAMS = (0.59282175, 3.7723004, 5.39725691)     # CLEAR_RD chosen design (s1_l, s2_width, s2_depth) mm
# CLEAR_RD cell 13 rerun for this design with 3e6 primaries (the notebook itself uses 1e5)
TOPAS_FILE = os.path.join(PROJECT_DIR, 'data', 'topas', 'DoseAtTank1_CLEAR_RD_3e6_C.csv')
HALF_WIDTH, NBIN = 40.0, 100                               # TOPAS scorer: 80 mm, 100 x 100 bins
INK, INK2, GRID, RFT_COLOUR = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
SG_COLOUR, G2_COLOUR = '#eb6834', '#1baf7a'   # categorical slots 2, 3


def topas_map(path=TOPAS_FILE):
    from topasToDose import BinnedResult
    D = BinnedResult(path).data['Sum']
    return D[:, :, 0] if D.ndim == 3 else D


def rftrack_particles(params, n=3000000, seed=0):     # same statistics as the 3e6 TOPAS run
    import RF_Track as rft
    import design_optimisation as do
    s1, w, d = params
    rng = np.random.default_rng(seed)
    p = np.sqrt((198 + rft.electronmass)**2 - rft.electronmass**2)
    B0 = rft.Bunch6d(rft.electronmass, 10 * rft.nC, -1,
                     np.c_[rng.normal(0, 1, n), rng.normal(0, 0.001, n), rng.normal(0, 1, n),
                           rng.normal(0, 0.001, n), np.zeros(n), np.full(n, p)])
    lattice = do.build_scatterer_lattice(w, d, s1, window=False, drift_to_s1=0.043, drift_to_s2=0.563,
                                         drift_to_phsp=2.45, air_before_phsp=0.3, kapton=0.25,
                                         exit_window=[(0.12, do.KAPTON), (0.005, do.COPPER)])
    rft.rng_set_seed(1)
    M = lattice.track(B0).get_phase_space('%x %y')
    return M[:, 0], M[:, 1]


def rebin(a, f):
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('params', nargs='*', type=float, default=list(DEFAULT_PARAMS))
    ap.add_argument('--out', default=os.path.join(PROJECT_DIR, 'NIMB_optimiser', 'figs'))
    ap.add_argument('--preview', default=None, help='also write a PNG here')
    ap.add_argument('--topas', default=TOPAS_FILE, help='TOPAS dose CSV (DoseAtTank...)')
    args = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    edges = np.linspace(-HALF_WIDTH, HALF_WIDTH, NBIN + 1); c = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(c, c, indexing='ij'); R = np.hypot(X, Y)
    D = topas_map(args.topas)                                  # indexed [x, y]
    x, y = rftrack_particles(args.params)
    H = np.histogram2d(x - x.mean(), y - y.mean(), bins=[edges, edges])[0]   # indexed [x, y], about the centroid

    # field characterisation from the central x and y slices (|other| < 1 mm), as in Table 2 of the paper
    from flatness import slice_fit, map_slices, sum_2gaussians, SLICE_HALF_WIDTH
    t_sl, f_sl = map_slices(c, D), map_slices(c, H)
    t_fit = [slice_fit(cc, pp) for cc, pp in t_sl]; f_fit = [slice_fit(cc, pp) for cc, pp in f_sl]
    mean = lambda fits, k: np.mean([f[k] for f in fits])
    tN, fN = mean(t_fit, 'A'), mean(f_fit, 'A')                  # field level: mean fitted peak of the two slices
    Dn = D / tN; Hn = H / fN
    def sg_curve(fit, scale):  # the slice's super-Gaussian, centred, recovered from its x90 and x50
        P = fit['P']; s2 = fit[0.9] / (-np.log(0.9))**(1 / (2 * P))
        return lambda u: fit['A'] / scale * np.exp(-((u / s2)**2)**P)
    for name, fits in (('TOPAS', t_fit), ('RF-Track', f_fit)):
        print(f"{name}: x90 {mean(fits, 0.9):.2f}  x50 {mean(fits, 0.5):.2f}  x20 {mean(fits, 0.2):.2f} mm  "
              f"P {mean(fits, 'P'):.2f}  |mu/sigma| within x90 {mean(fits, 'mu/sigma x90'):.3f}, within x50 {mean(fits, 'mu/sigma x50'):.3f}")

    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2})
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    vmax = 1.3
    for k, (Wn, title) in enumerate([(rebin(Hn, 2), 'RF-Track (particle density)'), (rebin(Dn, 2), 'TOPAS (dose, 0-1 mm water)')]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(Wn.T, origin='lower', extent=[-HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH],
                       cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
        ax.add_patch(plt.Circle((0, 0), 25, fill=False, color=INK, lw=0.8, ls='--'))
        hw = SLICE_HALF_WIDTH
        for xy, w, h in (((-HALF_WIDTH, -hw), 2 * HALF_WIDTH, 2 * hw), ((-hw, -HALF_WIDTH), 2 * hw, 2 * HALF_WIDTH)):
            ax.add_patch(plt.Rectangle(xy, w, h, fill=False, edgecolor='white', lw=0.9, ls='--'))
        ax.set_title(title, fontsize=8.5, color=INK); ax.set_xlabel('x (mm)')
        if k == 0: ax.set_ylabel('y (mm)')
        else: ax.set_yticklabels([])
        ax.text(0.03, 0.97, '(a)' if k == 0 else '(b)', transform=ax.transAxes, va='top', fontweight='bold', color=INK)
    cb = fig.colorbar(im, ax=fig.axes[:2], shrink=0.85, location='right', pad=0.02)
    cb.set_label('normalised to fitted peak', fontsize=7.5)
    for k, axis in enumerate(['x', 'y']):
        ax = fig.add_subplot(gs[1, k])
        tp, fp = t_sl[k][1] / tN, f_sl[k][1] / fN
        ax.plot(c, tp, color=INK, lw=1.4, label='TOPAS')
        ax.plot(c, fp, color=RFT_COLOUR, lw=1.2, ls='--', label='RF-Track')
        cf = np.linspace(-HALF_WIDTH, HALF_WIDTH, 400); tf = t_fit[k]
        ax.plot(cf, sg_curve(tf, tN)(cf), color=SG_COLOUR, lw=1.2, ls='-.', label='super-Gaussian fit (TOPAS)')
        cg = cf[np.abs(cf) < tf[0.9]]
        ax.plot(cg, sum_2gaussians(cg, *tf['g2 x90']) / tN, color=G2_COLOUR, lw=1.6, ls=':', label='two-Gaussian fit within $x_{90}$ (TOPAS)')
        ax.text(0.03, 0.90, f"$x_{{90}}$ = {tf[0.9]:.1f} / {f_fit[k][0.9]:.1f} mm\n$|\\mu/\\sigma|$ = {tf['mu/sigma x90']:.2f} / {f_fit[k]['mu/sigma x90']:.2f}\n(TOPAS / RF-Track)",
                transform=ax.transAxes, ha='left', va='top', fontsize=7, color=INK2,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
        for v in (-25, 25): ax.axvline(v, color=INK2, lw=0.8, ls=':')
        ax.set_xlabel(f'{axis} (mm)')
        ax.set_ylabel(f'central {axis} profile ($|{"y" if axis == "x" else "x"}| < {SLICE_HALF_WIDTH:g}$ mm)')
        ax.set_xlim(-HALF_WIDTH, HALF_WIDTH); ax.set_ylim(0, 1.6); ax.grid(color=GRID, lw=0.6)
        ax.legend(fontsize=6.3, frameon=False, loc='lower center')
        ax.text(0.02, 0.97, '(c)' if k == 0 else '(d)', transform=ax.transAxes, va='top', fontweight='bold', color=INK)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, 'clear_rd_rft_vs_topas.pdf'))
    if args.preview:
        fig.savefig(args.preview, dpi=170)
    print('saved', os.path.join(args.out, 'clear_rd_rft_vs_topas.pdf'))


if __name__ == '__main__':
    main()
