import numpy as np
from scipy.optimize import fminbound, curve_fit
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Global cache
PRC_UniformDisk = None


def merit_beam_Uniform(B1, Rmin, Rmax, transmission=0.998):
    global PRC_UniformDisk

    # --- Phase space extraction ---
    M1 = B1.get_phase_space()
    X = M1[:, 0] - np.mean(M1[:, 0])
    Y = M1[:, 2] - np.mean(M1[:, 2])

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

    N = 80
    XV = []
    YV = []

    # 1%–99% percentiles → PRC[:-1]
    prc_short = PRC[:-1]
    disk_short = PRC_UniformDisk[:-1]

    for Phid in np.linspace(0, 180, N + 1)[:-1]:
        P = X * np.cos(np.deg2rad(Phid)) + Y * np.sin(np.deg2rad(Phid))

        # best-fit radius
        def err_R(R):
            return np.sum((np.percentile(P, prc_short) - R * disk_short)**2)

        R = fminbound(err_R, 0, 2 * Rmax)
        XV.append(R * 0.98 * np.cos(np.deg2rad(Phid)))
        YV.append(R * 0.98 * np.sin(np.deg2rad(Phid)))

    # complete polygon using opposite points
    XV = np.concatenate([XV, -np.array(XV)])
    YV = np.concatenate([YV, -np.array(YV)])

    # inpolygon (matplotlib path)
    from matplotlib.path import Path
    poly = Path(np.column_stack([XV, YV]))
    inside = poly.contains_points(np.column_stack([X, Y]))

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

        R = fminbound(err_R, 0, 2 * Rmax)
        F = err_R(R)

        # uniformity term
        M_uniformity += F / (R**2) / N

        # kurtosis term
        M_kurtosis += ((kurtosis(P) / 2 - 1)**2) / N

        # size penalties
        M_size += max(R / Rmin - 1, 0)**2      # R too small
        M_size += min(R / Rmax - 1, 0)**2      # R too large

    # scale terms
    M_uniformity *= 1e4
    M_kurtosis *= 1e4 * 0
    M_size *= 1e8 *0

    # Transmission term
    trans_ratio = B1.get_ngood() / B1.size()
    M_transmission = 1e8 * min(trans_ratio - transmission, 0)**2
    # print(M_uniformity, M_kurtosis, M_size,M_transmission,R)

    # Final merit value
    return M_uniformity + M_kurtosis + M_size + M_transmission


def mask2d(x,y,pc=60):
    r = np.sqrt(x**2 + y**2)
    threshold = np.percentile(r, pc)
    mask = r <= threshold
    return x[mask],y[mask]

def nearest_neighbor_test(x,y):
    points = np.array([x,y]).T
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)
    nn_distances = distances[:, 1]  # skip self-distance

    mean_dist = nn_distances.mean()
    std_dist = nn_distances.std()
    cv = std_dist / mean_dist  
    # coefficient of variation = 0.52 is poisson uniform, <0.52 is too uniform, >0.52 clustering

    return mean_dist, std_dist, cv

def sum_2gaussians(x, A, x0, sigma_x):
    return A * (np.exp(-( (x-x0)**2 /(2*sigma_x**2) )) + np.exp(-( (x+x0)**2 /(2*sigma_x**2) )) )

def sum_2gaussians_skewed(x, A, x0, sigma_x, m, c,mu):
    return A * (np.exp(-( (x-x0-mu)**2 /(2*sigma_x**2) )) + np.exp(-( (x+x0-mu)**2 /(2*sigma_x**2) )) ) + m * x + c

def loss_2gauss(hist_x, bin_centers_x, hist_y, bin_centers_y):
    p0 = [np.max(hist_x),  10, np.std(bin_centers_x)]
    params_x, _ = curve_fit(sum_2gaussians, bin_centers_x, hist_x, p0=p0)
    params_y, _ = curve_fit(sum_2gaussians, bin_centers_y, hist_y, p0=p0) 
    return abs(params_x[1]/params_x[2] -1.1)  + abs(params_y[1]/params_y[2] -1.1) #x0/sigma  