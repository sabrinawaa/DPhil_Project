import numpy as np
from scipy.optimize import fminbound
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

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
        plt.plot(PRC_UniformDisk)
        plt.show()

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
    print(M_uniformity, M_kurtosis, M_size,M_transmission,R)

    # Final merit value
    return M_uniformity + M_kurtosis + M_size + M_transmission
