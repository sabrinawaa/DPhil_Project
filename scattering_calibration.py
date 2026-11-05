'''Pencil-beam calibration of RF-Track scattering materials against TOPAS/Geant4.

A 198 MeV (kinetic) electron pencil beam crosses one slab in vacuum; the angles of the primary
electrons are scored directly behind it in TOPAS and compared with RF-Track for a range of
effective Z (A fixed) and for Z/A-correct effective values.

    python scattering_calibration.py topas [materials]   # run the TOPAS slabs (~a few min), all by default
    python scattering_calibration.py compare    # RF-Track scan + comparison table
Outputs: data/topas/calib_<material>_<t>mm.phsp, calibration/results.json, calibration/*.png
'''
import json
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

E_KIN = 198.0          # MeV, as in the TOPAS beam of CLEAR_RD
N_TOPAS = 100000
SLABS = {              # TOPAS material -> thicknesses (mm)
    'Peek': [0.25, 0.5, 1, 2, 4, 7],       # current definition in partrec_gaussian_optimiser_utils
    'PeekTrue': [0.25, 0.5, 1, 2, 4, 7],   # C19H12O3 mass fractions
    'Aluminum': [0.25, 0.76, 1.5],
    # windows and air as they appear in CLEAR_RD, simulate_PDD_CLEAR and CLARA_design
    'Kapton': [0.025, 0.12, 0.25],
    'G4_Cu': [0.005, 0.05],
    'G4_Be': [0.25],
    'Air': [92, 300, 500],
}
PERCENTILES = [68.27, 95, 99]   # of |theta_x|; 68.27% is sigma for a Gaussian
OUT_DIR = os.path.join(PROJECT_DIR, 'calibration')
os.makedirs(OUT_DIR, exist_ok=True)


def tag(material, t):
    return f'calib_{material}_{t}mm'


def run_topas(materials=None):
    '''materials: subset of SLABS to run (default all)'''
    from partrec_gaussian_optimiser_utils import partrec_gaussian_optimiser_utils
    for material, thicknesses in SLABS.items():
        if materials and material not in materials:
            continue
        for t in thicknesses:
            setup = partrec_gaussian_optimiser_utils(world_material="Vacuum", input_filename='calib_main.txt')
            f = setup.file
            f.write('sv:Ma/PeekTrue/Components = 3 "Carbon" "Hydrogen" "Oxygen"\n')
            f.write('uv:Ma/PeekTrue/Fractions = 3 0.7915 0.0420 0.1665\n')
            f.write('d:Ma/PeekTrue/Density = 1.32 g/cm3\n')
            setup.generate_phsp_beam(1e-4, 1e-4, 1e-4, 1e-4, E_KIN, 0, N_TOPAS)   # pencil beam
            setup.add_cylinder('slab', t, 0, 500, material, 10)
            # thin vacuum scorer directly behind the slab, primaries only
            f.write('s:Ge/Scorer/Type="TsBox"\ns:Ge/Scorer/Parent="World"\ns:Ge/Scorer/Material="Vacuum"\n')
            f.write('d:Ge/Scorer/HLX = 0.5 m\nd:Ge/Scorer/HLY = 0.5 m\nd:Ge/Scorer/HLZ = 0.001 mm\n')
            setup._write_transz('Scorer', 'World', 10 + t + 0.01, 0.001)
            f.write('s:Sc/calib/Quantity = "PhaseSpace"\ns:Sc/calib/Surface = "Scorer/ZPlusSurface"\n')
            f.write('s:Sc/calib/OnlyIncludeParticlesOfGeneration = "Primary"\n')
            f.write(f's:Sc/calib/OutputFile = "{tag(material, t)}"\n')
            f.write('s:Sc/calib/OutputType = "ASCII"\ns:Sc/calib/IfOutputFileAlreadyExists = "Overwrite"\n')
            f.write('b:Sc/calib/OutputToConsole = "False"\nb:Ge/CheckForOverlaps = "False"\n')
            f.write('b:Ph/ListProcesses = "False"\nb:Ge/CheckForUnusedComponents = "False"\n')
            setup.mute_terminal_output()
            print('TOPAS', material, t, flush=True)
            setup.run_topas('calib_main.txt')


def topas_angles(material, t):
    data = np.loadtxt(os.path.join(PROJECT_DIR, 'data', 'topas', tag(material, t) + '.phsp'), usecols=(3, 7))
    return data[data[:, 1] == 11, 0] * 1000   # direction cosine x -> mrad, electrons only


def rft_angles(thickness_mm, material, n=200000):
    import RF_Track as rft
    p = np.sqrt((E_KIN + rft.electronmass)**2 - rft.electronmass**2)
    B = rft.Bunch6d(rft.electronmass, 10*rft.nC, -1, np.c_[np.zeros((n, 5)), np.full(n, p)])
    lat = rft.Lattice()
    a = rft.Absorber(thickness_mm/1000, material) if isinstance(material, str) else rft.Absorber(thickness_mm/1000, *material)
    a.disable_energy_straggling(); a.set_shape('circular', 1, 1); lat.append(a)
    rft.rng_set_seed(1)
    return lat.track(B).get_phase_space('%xp')[:, 0]


def pct(theta):
    return np.percentile(np.abs(theta), PERCENTILES)


def compare():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    peek_candidates = {f'Z={Z} A=288.31': (31.9, Z, 288.31, 1.32, -1) for Z in [25, 30, 33, 37, 40, 45, 50]}
    peek_candidates['Z/A 0.52 (6.6/12.7)'] = (31.9, 6.6, 12.7, 1.32, -1)
    # final: Z fitted at the real Z/A of C19H12O3 (150 / 288.30)
    peek_candidates.update({f'Z={Z} at Z/A 0.520': (31.9, Z, Z / (150 / 288.302), 1.32, -1) for Z in [6.2, 6.3, 6.4, 6.5]})
    candidates = {
        'Peek': peek_candidates, 'PeekTrue': peek_candidates,
        'Aluminum': {f'Al Z={Z} A=26.98': (8.897, Z, 26.982, 2.7, 166) for Z in [13, 13.5, 14, 14.5]},
        # kapton C22H10N2O5: effective Z scanned at the real Z/A = 0.513
        'Kapton': {f'Z={Z} A={Z / 0.5126:.2f}': (28.57, Z, Z / 0.5126, 1.42, 79.6) for Z in [5, 6.5, 6.6, 6.7, 6.8, 8, 10, 15, 20]},
        'G4_Cu': {f'Cu Z={Z} A=63.55': (1.436, Z, 63.546, 8.96, 322.0) for Z in [29, 30, 31, 32]},
        'G4_Be': {'built-in "beryllium"': 'beryllium',
                  **{f'Be Z={Z} A=9.012': (35.28, Z, 9.012, 1.848, 63.7) for Z in [4, 4.5, 5]}},
        # air: built-in, and explicit X0/rho/I with an effective Z scanned at the real Z/A = 0.4992
        'Air': {'built-in "air"': 'air',
                **{f'Z={Z} A={Z / 0.4992:.2f}': (30390.0, Z, Z / 0.4992, 0.0012048, 85.7) for Z in [3, 3.6, 4, 7.3]}},
    }
    results = {}
    for material, thicknesses in SLABS.items():
        cands = candidates[material]
        results[material] = {}
        print(f'\n=== {material}: |theta_x| percentiles {PERCENTILES} in mrad, RF-Track / TOPAS ratio')
        for t in thicknesses:
            ref = pct(topas_angles(material, t))
            row = {'TOPAS': ref.tolist()}
            line = f'  {t:5} mm  TOPAS {np.round(ref, 2)}'
            for name, mat in cands.items():
                r = pct(rft_angles(t, mat)); row[name] = r.tolist()
                line += f' | {name}: ' + '/'.join(f'{v:.2f}' for v in r / ref)
            results[material][t] = row
            print(line)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(dict(percentiles=PERCENTILES, results=results), f, indent=1)

    # best Z (A=288.31) per thickness for the core (68%) and the tail (99%), by interpolating the ratio
    print('\n=== effective Z (A=288.31) that matches TOPAS, per thickness')
    Zs = np.array([25, 30, 33, 37, 40, 45, 50])
    for material in ['Peek', 'PeekTrue']:
        for t, row in results[material].items():
            ref = np.array(row['TOPAS'])
            out = []
            for k, label in [(0, 'core 68%'), (2, 'tail 99%')]:
                ratio = np.array([row[f'Z={Z} A=288.31'][k] for Z in Zs]) / ref[k]
                Zbest = np.interp(1.0, ratio, Zs) if ratio.min() < 1 < ratio.max() else np.nan
                out.append(f'{label}: Z={Zbest:5.1f}')
            print(f'  {material:8s} {t:5} mm  ' + '   '.join(out))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, material in zip(axs, ['Peek', 'PeekTrue']):
        ts = SLABS[material]
        for k, ls in [(0, '-'), (2, '--')]:
            ax.plot(ts, [results[material][t]['TOPAS'][k] for t in ts], 'k' + ls, lw=2.5,
                    label=f'TOPAS {PERCENTILES[k]}%')
            for name in ['Z=30 A=288.31', 'Z=37 A=288.31', 'Z=45 A=288.31', 'Z/A 0.52 (6.6/12.7)']:
                ax.plot(ts, [results[material][t][name][k] for t in ts], ls, alpha=0.8,
                        label=f'{name} {PERCENTILES[k]}%' if k == 0 else None)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('PEEK thickness (mm)')
        ax.set_ylabel('|theta_x| percentile (mrad)'); ax.set_title(f'TOPAS material "{material}"')
        ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(os.path.join(OUT_DIR, 'peek_calibration.png'), dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    if sys.argv[1] == 'topas':
        run_topas(sys.argv[2:])
    else:
        compare()
