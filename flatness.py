import numpy as np
from scipy.optimize import fminbound, curve_fit
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Global cache
PRC_UniformDisk = None


def merit_beam_Uniform(B1, R, transmission_goal=0.998):
    M1 = B1.get_phase_space()
    return merit_uniform_xy(M1[:, 0], M1[:, 2], R, transmission_goal)


def merit_uniform_xy(x, y, R, transmission_goal=0.998, verbose=True):
    '''merit_beam_Uniform on x, y arrays (mm): compares the percentiles of the beam projected
    onto many angles with those of a uniform disk of radius R, plus a transmission term'''
    global PRC_UniformDisk

    # --- Phase space extraction ---
    N_total = len(x)
    X = x - np.mean(x)
    Y = y - np.mean(y)

    # --- Constants ---
    PRC = np.arange(0, 102, 2)  # 0:2:100

    # Precompute inverse CDF of uniform disk (cached)
    if PRC_UniformDisk is None:
        # CDF of uniform disk: A(y)
        def A(y):
            return 100 * (np.arcsin(y) / np.pi + (y * np.sqrt(1 - y**2)) / np.pi + 0.5)

        # inverse via argmin (fminbound)
        PRC_UniformDisk = np.array([
            fminbound(lambda x: (A(x) - p)**2, -1, 1)
            for p in PRC
        ])
        # plt.plot(PRC_UniformDisk)
        # plt.show()

    # ============================================================
    #             Merit function — PART 1
    # ============================================================

    # keep the particles inside the target disk of radius R (this used to be a 160-sided polygon
    # with every vertex at R, i.e. the same circle; its per-angle best-fit radius was never used)
    inside = X**2 + Y**2 < R**2

    X = X[inside]
    Y = Y[inside]

    # ============================================================
    #             Merit function — PART 2
    # ============================================================

    M_uniformity = 0.0
    M_kurtosis = 0.0
    M_size = 0.0

    N = 60

    for Phid in np.linspace(0, 180, N + 1)[:-1]:
        P = X * np.cos(np.deg2rad(Phid)) + Y * np.sin(np.deg2rad(Phid))

        def err_R(R):
            return np.sum((np.percentile(P, PRC) - R * PRC_UniformDisk)**2)

        F = err_R(R)

        # uniformity term
        M_uniformity += F / (R**2) / N

        # kurtosis term
        M_kurtosis += ((kurtosis(P) / 2 - 1)**2) / N


    # scale terms
    M_uniformity *= 1e4
    M_kurtosis *= 1e4 * 0
    # M_size *= 1e8 *0

    # Transmission term
    transmission = np.sum(X**2 + Y**2 < R**2) / N_total
    M_transmission = 100 * max(transmission_goal - transmission, 0)**2
    if verbose:
        print(np.sum(X**2 + Y**2 < R**2), N_total)
        print(M_uniformity, M_kurtosis, M_transmission,R)

    # Final merit value
    return M_uniformity + M_kurtosis + M_transmission


def central_mask(x,y,pc=60):
    '''boolean mask of the central pc% of particles, ranked by radius'''
    r = np.sqrt(x**2 + y**2)
    threshold = np.percentile(r, pc)
    return r <= threshold

def mask2d(x,y,pc=60):
    mask = central_mask(x,y,pc)
    return x[mask],y[mask]

def nearest_neighbor_test(x,y,k=1,query=None):
    '''mean, std and CV of the distance to the k-th nearest neighbour; a larger k averages out
    Poisson scatter so the CV follows density variations on a larger length scale.
    query: optional boolean mask of the particles whose distances are evaluated; neighbours are
    still searched among all particles, so the evaluated region has no artificial boundary
    (otherwise particles near its edge have neighbours on one side only and inflate the CV)'''
    points = np.array([x,y]).T
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points if query is None else points[query])
    nn_distances = distances[:, k]  # column 0 is the self-distance

    mean_dist = nn_distances.mean()
    std_dist = nn_distances.std()
    cv = std_dist / mean_dist  
    # for k=1: coefficient of variation = 0.52 is poisson uniform, <0.52 is too uniform, >0.52 clustering

    return mean_dist, std_dist, cv

def sum_2gaussians(x, A, x0, sigma_x):
    return A * (np.exp(-( (x-x0)**2 /(2*sigma_x**2) )) + np.exp(-( (x+x0)**2 /(2*sigma_x**2) )) )

def sum_2gaussians_skewed(x, A, x0, sigma_x, m, c,mu):
    return A * (np.exp(-( (x-x0-mu)**2 /(2*sigma_x**2) )) + np.exp(-( (x+x0-mu)**2 /(2*sigma_x**2) )) ) + m * x + c

def loss_2gauss(hist_x, bin_centers_x, hist_y, bin_centers_y, target_ratio=1.1):
    '''|x0/sigma - target_ratio| for x and y; x0/sigma = 1 is exactly flat at the centre,
    1.1 gives a ~2.6% central dip'''
    p0 = [np.max(hist_x),  10, np.std(bin_centers_x)]
    params_x, _ = curve_fit(sum_2gaussians, bin_centers_x, hist_x, p0=p0)
    params_y, _ = curve_fit(sum_2gaussians, bin_centers_y, hist_y, p0=p0) 
    # sum_2gaussians is symmetric in the signs of x0 and sigma, so the fit returns either sign
    ratio_x = abs(params_x[1])/abs(params_x[2])
    ratio_y = abs(params_y[1])/abs(params_y[2])
    return abs(ratio_x - target_ratio) + abs(ratio_y - target_ratio)

FIELD_LEVELS = (0.9, 0.5, 0.2)              # x90, x50, x20


def field_radii(rc, ring_mean, ring_area):
    """field level and radii x_100F of a radial profile (ring centres rc, mean value per ring, ring areas),
    from a super-Gaussian fit f(r) = A exp(-(r^2 / 2 sigma^2)^P) as in the paper (Sec. Supergaussian Fit):
    x_100F = sigma sqrt(2) (-ln F)^(1/2P) is the radius where the fit falls to F times its peak A.
    Returns A (the field level used for normalisation) and {F: x_100F} for x90, x50, x20.
    Rings are weighted by their area, since the statistical error of a ring mean falls as 1/sqrt(area)."""
    from scipy.optimize import curve_fit
    sg = lambda r, A, sig, P: A * np.exp(-(r**2 / (2 * sig**2))**P)
    ok = ring_area > 0
    A0 = np.average(ring_mean[ok][:max(3, ok.sum() // 4)], weights=ring_area[ok][:max(3, ok.sum() // 4)])
    sig0 = rc[ok][np.argmin(np.abs(ring_mean[ok] - 0.5 * A0))] / np.sqrt(2 * np.log(2))
    (A, sig, P), _ = curve_fit(sg, rc[ok], ring_mean[ok], p0=[A0, sig0, 2], sigma=1 / np.sqrt(ring_area[ok]),
                               bounds=([0, 1e-3, 0.5], [np.inf, np.inf, 20]), maxfev=20000)
    return A, {F: sig * np.sqrt(2) * (-np.log(F))**(1 / (2 * P)) for F in FIELD_LEVELS}


def field_radii_particles(x, y, n_rings=80):
    """field_radii for particle positions: radial density about the centroid in n_rings rings out to
    1.05 x the 97th percentile radius. Returns (A, {F: x_100F}, r)."""
    r = np.hypot(x - np.mean(x), y - np.mean(y))
    edges = np.linspace(0, 1.05 * np.percentile(r, 97), n_rings + 1)
    rc = 0.5 * (edges[1:] + edges[:-1]); area = np.pi * np.diff(edges**2)
    A, X = field_radii(rc, np.histogram(r, bins=edges)[0] / area, area)
    return A, X, r


# ---- field characterisation from 1D central slices (paper: Table 2, Fig. 2) -------------------------
SLICE_HALF_WIDTH = 1.0    # mm, |other coordinate| < 1 mm, as in uniformity_fit.plot_phsp


def slice_fit(c, prof):
    """super-Gaussian (Sec. Supergaussian Fit) and two-Gaussian (Sec. Two-Gaussian Fit) fits of one
    central slice (bin centres c relative to the beam centre, profile prof), all bins weighted equally.
    Returns A (fitted peak), P, x90/x50/x20 = sigma sqrt(2) (-ln F)^(1/2P), and |mu/sigma| of the
    two-Gaussian fitted within |c| < x90 and within |c| < x50."""
    def sg(x, A, x0, sig, P):
        return A * np.exp(-(((x - x0)**2) / (2 * sig**2))**P)
    scale = prof.max(); prof = prof / scale      # fit on a profile of order 1 (dose maps are ~1e-6 Gy)
    half = c[np.argmin(np.abs(prof[c > 0] - 0.5 * prof.max())) + np.sum(c <= 0)]
    (A, x0, sig, P), _ = curve_fit(sg, c, prof, p0=[prof.max(), 0, half / 1.2, 2],
                                   bounds=([0, -np.inf, 1e-3, 0.5], [np.inf, np.inf, np.inf, 20]), maxfev=20000)
    out = {'A': A * scale, 'P': P}
    out.update({F: abs(sig) * np.sqrt(2) * (-np.log(F))**(1 / (2 * P)) for F in FIELD_LEVELS})
    for key, lim in (('mu/sigma x90', out[0.9]), ('mu/sigma x50', out[0.5])):
        m = np.abs(c - x0) < lim
        g, _ = curve_fit(sum_2gaussians, c[m] - x0, prof[m], p0=[0.6 * np.median(prof[m]), 0.4 * lim, 0.4 * lim], maxfev=50000)
        out[key] = abs(g[1]) / abs(g[2])
        out['g2 ' + key.split()[-1]] = (g[0] * scale, g[1], g[2])   # (A, mu, sigma) in the units of prof, centred
    return out


def slice_metrics(profiles):
    """mean of slice_fit over the given slices [(c, prof), ...] (x and y) and half their spread"""
    fits = [slice_fit(c, p) for c, p in profiles]
    keys = [k for k in fits[0] if not str(k).startswith('g2')]
    return ({k: np.mean([f[k] for f in fits]) for k in keys},
            {k: np.ptp([f[k] for f in fits]) / 2 for k in keys})


def particle_slices(x, y, half_range, bin_width, half_width=SLICE_HALF_WIDTH):
    """x and y central slices (|other| < half_width) of particle positions, about the centroid"""
    x = x - np.mean(x); y = y - np.mean(y)
    edges = np.arange(-half_range, half_range + 1e-9, bin_width); c = 0.5 * (edges[1:] + edges[:-1])
    return [(c, np.histogram(x[np.abs(y) < half_width], edges)[0].astype(float)),
            (c, np.histogram(y[np.abs(x) < half_width], edges)[0].astype(float))]


def map_slices(c, W, half_width=SLICE_HALF_WIDTH):
    """x and y central slices of a map W[x, y] with bin centres c (map centred on the beam)"""
    sel = np.abs(c) < half_width
    return [(c, W[:, sel].mean(axis=1)), (c, W[sel, :].mean(axis=0))]
