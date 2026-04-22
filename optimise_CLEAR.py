import numpy as np
import matplotlib.pyplot as plt
import RF_Track as rft
from scipy.optimize import minimize, curve_fit
from CLEAR_line import *
from RF_track_utils import *
from flatness import *
from sklearn.neighbors import NearestNeighbors


#80% of central beam
def mask80(x):
    cdf = np.cumsum(x, dtype=float)
    cdf /= cdf[-1] # Normalize CDF to 1
    mask = (cdf >= 0.1) & (cdf <= 0.9) # Mask for central 80%
    return x[mask]


def nearest_neighbor_test(x,y):
    points = np.array([x,y]).T
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)
    nn_distances = distances[:, 1]  # skip self-distance

    mean_dist = nn_distances.mean()
    std_dist = nn_distances.std()
    print(nn_distances)
    cv = std_dist / mean_dist  
    # coefficient of variation = 0.52 is poisson uniform, <0.52 is too uniform, >0.52 clustering

    return mean_dist, std_dist, cv

def mask2d(x,y):
    r = np.sqrt(x**2 + y**2)
    threshold = np.percentile(r, 60)
    mask = r <= threshold
    return x[mask],y[mask]


mass = RF_Track.electronmass    # particle mass in MeV/c^2
population = 10 * RF_Track.nC               # number of particles per bunch                         # particle charge in e units
P_ref = 198  
N_particles = int(500000)
charge = -1

quad_currents = np.array([19.4, 20.9, 2, 30, 33, 0, 57, 61, 8, 0, 0]) # k1 values from OPTIMISE_CLEAR.py, last two quads not in selection
start = 'CA.QFD0350' #'CA.ACS0270S_MECH'
end = 'CA.DHJ0840' #'CA.STLINE$END'
CLEAR_lattice = get_beamline("CLEAR_Beamline_Survey.txt",start, end, P_ref, quad_currents)
print(CLEAR_lattice.get_length())
CLEAR_lattice.append(rft.Drift(0.084))
S1 = rft.Absorber(0.0001,8.897, 13,26.982,2.7, 166)
# S1 = rft.Absorber(0.0001,'air')
S1.disable_energy_straggling()
S1.set_shape ('circular', 1,1  )
CLEAR_lattice.append(S1)
CLEAR_lattice.append(rft.Drift(0.532)) #end of s1 to s2

s2_thickness = [0.688, 0.778, 0.581, 0.386]
s2_radii = [0.4, 0.8, 1.2, 1.6]
for i in range(len(s2_thickness)):
    Slice = rft.Absorber(s2_thickness[i]/1000,31.9, 37, 288.31,1.32,-1)
    Slice.disable_energy_straggling()
    Slice.set_shape ('circular',  abs(s2_radii[i])/1000,abs(s2_radii[i])/1000 )
    CLEAR_lattice.append(Slice)

CLEAR_lattice.append(rft.Drift(2.024))  #drift to water tank 


Twiss = RF_Track.Bunch6d_twiss()

Twiss.beta_x = 17.7        # m
Twiss.beta_y = 13.9     # m
Twiss.alpha_x = -1.14 
Twiss.alpha_y = 0.359
Twiss.emitt_x = 4.62     # mm.mrad normalised emittance
Twiss.emitt_y = 3.86     # mm.mrad
# Twiss.sigma_t = 10 * RF_Track.ps       # mm/c   or 37 * RF_Track.ps
# Twiss.sigma_pt = 10     # permille
Twiss.mean_xp = 0.0
Twiss.mean_yp = 0.0

B0 = RF_Track.Bunch6d_QR(mass, population, charge, P_ref, Twiss, N_particles)             # reference bunch

def loss (last_triplet_k1s, lattice, B0):
    Q = lattice.get_quadrupoles()
    for i in range(len(Q)-6, len(Q)):
        Q[i].set_strength(last_triplet_k1s[i-9])

    B1 = lattice.track(B0)
    I = B1.get_info()
    sx, sy, ax, ay= I.sigma_x, I.sigma_y, I.alpha_x, I.alpha_y
    # T = lattice.get_transport_table(
    # '%S %beta_x %beta_y %alpha_x %alpha_y %sigma_x %sigma_y %sigma_px %sigma_py')
    M = B1.get_phase_space('%x %xp %y %yp %E %z')
    # x,y = M[:,0], M[:,2]
    

    # loss = nearest_neighbor_test(x,y)[2]
    # loss = fatness(hist_x) + flatness(hist_y)
    # loss = merit_beam_Uniform(B1, 5, 20, transmission=0.998)
    # loss = sx**2 + sy**2 + (sx-sy)**2 + (ax-ay)**2 
    masked_x, masked_y = mask2d(M[:,0],M[:,2])
    print('max x,y:', max(masked_x), max(masked_y))
    loss = nearest_neighbor_test(masked_x,masked_y)[2]
    
    print('loss:', loss,'k1s =', last_triplet_k1s)
    return loss

B0_opt = RF_Track.Bunch6d_QR(mass, population, charge, P_ref, Twiss, 100000)     
rng = np.random.default_rng()
opt_loss = np.inf
for i in range(20):
  x0 = rng.integers(low=0, high=700, size=6) * [1,-1,1,1,-1,1]

  # x0 = [20,-20,20,20,-20,20]
  res = minimize(loss,
                        x0=x0, args=(CLEAR_lattice,B0_opt),
                        bounds=[ (0,700),(-700,0),(0,700),(0,700),(-700,0),(0,700)], #highest can be 772 apparently
                        method='Powell',
                        options={'disp': True
                                , 'xtol':0.01}
                      #   tol=1e-2
                        )
  if res.fun < opt_loss:
          opt_loss = res.fun
          opt_params = res.x

with open("opt_results.txt", "a") as f:
    f.write(f"opt_loss={opt_loss:.5f}, params={opt_params}\n")