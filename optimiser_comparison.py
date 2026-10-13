'''Compare optimisers for the CLEAR dual-scatterer design (paper: "Choice of optimiser").

Runs Powell (5 restarts, as in CLEAR_RD.ipynb), CMA-ES (3 runs) and COBYQA (5 restarts) on the
same merit function, then re-scores every result on a 1e5-particle bunch with the same 5 fresh
seeds so all methods are compared on equal terms.

Setup matches CLEAR_RD.ipynb: CLEAR 2nd beamline (CS.2NDBL$START -> CS.BTV0420, quads off,
198 MeV/c), beam re-centred at the beamline exit, Al S1 / PEEK S2 with drifts 2.243, 0.563,
2.45 m, 'nn' merit with k = 0.1% of particles, target radius 25 mm. Within a run every merit
evaluation uses the same RF-Track seed (common random numbers), each restart/run its own seed.

Usage:
    python optimiser_comparison.py                 # full comparison (~20 min with 4 workers)
    python optimiser_comparison.py --quick         # small smoke test (~1 min)
    python optimiser_comparison.py --methods cma   # subset of methods
Results go to optimiser_comparison/<tag>/results.json, one log per job next to it.
'''
import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = dict(
    survey_file='CLEAR_2nd_Beamline_Survey.txt',
    start='CS.2NDBL$START', end='CS.BTV0420',
    P_ref=198, beta=20, emitt=10,          # m, mm.mrad normalised
    N_opt=50000, N_rescore=100000,
    target_size=25, nn_frac=0.001, transmission_goal=0.6,
    drift_to_s1=2.243, drift_to_s2=0.563, drift_to_phsp=2.45,
    bounds=[(0.01, 5), (0.01, 30), (0.01, 30)],   # s1_l, s2_width, s2_depth in mm
    x0_low=[0.01, 0.01, 0.01], x0_high=[5, 15, 15],
    n_powell=5, n_cma=3, n_cobyqa=5,
    powell_xtol=0.01, cma_maxfevals=900, cobyqa_maxfev=400,
    rescore_seeds=list(range(1000, 1005)),
)
QUICK = dict(N_opt=5000, N_rescore=10000, n_powell=2, n_cma=1, n_cobyqa=2,
             cma_maxfevals=60, cobyqa_maxfev=40, powell_xtol=0.5, rescore_seeds=[1000, 1001])


def setup(cfg):
    '''beam tracked through the fixed beamline and re-centred, plus the loss function'''
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    import RF_Track as rft
    import design_optimisation as do
    from CLEAR_2nd_line import get_2nd_beamline, build_rft_lattice

    mass, population, charge = rft.electronmass, 10 * rft.nC, -1
    twiss = rft.Bunch6d_twiss()
    twiss.beta_x = twiss.beta_y = cfg['beta']
    twiss.alpha_x = twiss.alpha_y = 0.0
    twiss.emitt_x = twiss.emitt_y = cfg['emitt']
    twiss.mean_xp = twiss.mean_yp = 0.0

    beamline = build_rft_lattice(get_2nd_beamline(cfg['survey_file']), start=cfg['start'], end=cfg['end'],
                                 P_ref=cfg['P_ref'], quad_currents=[0] * 6, Q=-1, include_end=True)

    def centred_bunch(n):
        P = beamline.track(rft.Bunch6d_QR(mass, population, charge, cfg['P_ref'], twiss, n)).get_phase_space(
            '%x %xp %y %yp %t %Pc')
        P[:, 0] -= np.mean(P[:, 0])
        P[:, 2] -= np.mean(P[:, 2])
        return rft.Bunch6d(mass, population, charge, P)

    loss = partial(do.loss, target_size=cfg['target_size'], window=False,
                   drift_to_s1=cfg['drift_to_s1'], drift_to_s2=cfg['drift_to_s2'],
                   drift_to_phsp=cfg['drift_to_phsp'], merit='nn',
                   transmission_goal=cfg['transmission_goal'], nn_frac=cfg['nn_frac'])
    return centred_bunch, loss


def powell_x0s(cfg):
    # same starting points as the Powell restarts in CLEAR_RD.ipynb (COBYQA reuses them)
    rng = np.random.default_rng(12345)
    return [rng.uniform(low=cfg['x0_low'], high=cfg['x0_high']) for _ in range(max(cfg['n_powell'], cfg['n_cobyqa']))]


def run_job(job, cfg, out_dir):
    '''one optimiser run; returns the converged params, evaluation count and re-scored losses'''
    method, run = job
    log_path = os.path.join(out_dir, f'{method}_{run}.log')
    with open(log_path, 'w') as log, contextlib.redirect_stdout(log):
        from scipy.optimize import minimize
        centred_bunch, loss = setup(cfg)
        B_opt = centred_bunch(cfg['N_opt'])
        lo, hi = np.array(cfg['bounds']).T
        nfev = [0]

        def f(x, seed):
            nfev[0] += 1
            return loss(np.asarray(x), B_opt, seed=seed)

        t0 = time.time()
        if method == 'powell':
            res = minimize(f, powell_x0s(cfg)[run], args=(run,), bounds=cfg['bounds'], method='Powell',
                           options={'xtol': cfg['powell_xtol']})
            x = res.x
        elif method == 'cobyqa':
            res = minimize(f, powell_x0s(cfg)[run], args=(run,), bounds=cfg['bounds'], method='COBYQA',
                           options={'final_tr_radius': 0.01, 'maxfev': cfg['cobyqa_maxfev']})
            x = res.x
        elif method == 'cma':
            import cma
            rng = np.random.default_rng(777 + run)
            u0 = (rng.uniform(cfg['x0_low'], cfg['x0_high']) - lo) / (hi - lo)
            # normalised coordinates so a single step size suits all three parameters
            es = cma.CMAEvolutionStrategy(u0, 0.2, {'bounds': [0, 1], 'tolx': 1e-4, 'tolfun': 1e-4,
                                                    'maxfevals': cfg['cma_maxfevals'], 'seed': run + 1,
                                                    'verbose': -9})
            es.optimize(lambda u: f(lo + np.asarray(u) * (hi - lo), seed=100 + run))
            x = lo + es.result.xbest * (hi - lo)
        else:
            raise ValueError(method)
        opt_time = time.time() - t0

        B_full = centred_bunch(cfg['N_rescore'])
        scores = [loss(np.asarray(x), B_full, seed=s) for s in cfg['rescore_seeds']]
    return dict(method=method, run=run, params=[float(v) for v in x], nfev=nfev[0],
                time=opt_time, scores=[float(s) for s in scores])


def summarise(results):
    lines = ['| method | run | s1_l | s2_width | s2_depth | re-scored loss | evaluations | time (min) |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        p = r['params']
        lines.append(f"| {r['method']} | {r['run']} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | "
                     f"{np.mean(r['scores']):.4f} ± {np.std(r['scores']):.4f} | {r['nfev']} | {r['time'] / 60:.1f} |")
    for method in dict.fromkeys(r['method'] for r in results):
        rs = [r for r in results if r['method'] == method]
        best = min(rs, key=lambda r: np.mean(r['scores']))
        lines.append(f"\n{method}: best re-scored loss {np.mean(best['scores']):.4f} at "
                     f"{np.round(best['params'], 3).tolist()}, {sum(r['nfev'] for r in rs)} evaluations in total")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='+', default=['powell', 'cma', 'cobyqa'],
                        choices=['powell', 'cma', 'cobyqa'])
    parser.add_argument('--quick', action='store_true', help='small smoke test, results not meaningful')
    parser.add_argument('--workers', type=int, default=4, help='parallel jobs (RF-Track is multithreaded too)')
    parser.add_argument('--tag', default=None, help='output sub-directory name')
    args = parser.parse_args()

    cfg = dict(CONFIG, **(QUICK if args.quick else {}))
    out_dir = os.path.join(PROJECT_DIR, 'optimiser_comparison', args.tag or ('quick' if args.quick else 'full'))
    os.makedirs(out_dir, exist_ok=True)

    n_runs = {'powell': cfg['n_powell'], 'cma': cfg['n_cma'], 'cobyqa': cfg['n_cobyqa']}
    jobs = [(m, i) for m in args.methods for i in range(n_runs[m])]
    print(f'{len(jobs)} jobs on {args.workers} workers, logs in {out_dir}')

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(partial(run_job, cfg=cfg, out_dir=out_dir), jobs))

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(dict(config=cfg, results=results), f, indent=1)
    summary = summarise(results)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
