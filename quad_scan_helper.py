
import numpy as np
from clear_quadrupole_scans.quad_scan import full_twiss, lorentz_factor
from clear_quadrupole_scans.clear_lattice import get_lattice
from clear_quadrupole_scans.linear_optics import Quad
from scipy.optimize import curve_fit


QUAD_END_POSITIONS = np.array([30.297400000, 30.747400000, 31.197400000])  # m, exit of 760,765,770
QUAD_LENGTH = 0.226  # m
SCREEN_POSITION = 34.167400000  # m, thin screen

# QUAD_END_POSITIONS = np.array([18.509400000, 18.949400000, 19.389400000])  # m, exit of 350,355,360
# QUAD_LENGTH = 0.226  # m
# SCREEN_POSITION = 20.556400000 # m, thin screen screen 390

def _thin_quad_matrix(k, L):
    """Thin-lens quad matrix: position unchanged, angle kicked by -k*L."""
    return np.array([[1.0, 0.0], [-k * L, 1.0]])

def _drift_matrix(d):
    return np.array([[1.0, d], [0.0, 1.0]])


def _downstream_chain(scan_quad_index, quad_L=QUAD_LENGTH,
                       quad_end_positions=QUAD_END_POSITIONS,
                       screen_position=SCREEN_POSITION):
    """
    List of (kind, value) describing everything from the scanned quad's
    entrance to the screen: the scanned quad itself, then only the FIXED
    quads/drifts downstream of it. Upstream quads are not included at all.
    kind is 'scan_quad', 'fixed_quad', or 'drift'.
    """
    quad_end_positions = np.asarray(quad_end_positions, dtype=float)
    quad_start_positions = quad_end_positions - quad_L
    names = [0, 1, 2]  # 760, 765, 770

    elements = [("scan_quad", scan_quad_index)]
    pos = quad_end_positions[scan_quad_index]
    for j in names[scan_quad_index + 1:]:
        gap = quad_start_positions[j] - pos
        elements.append(("drift", gap))
        elements.append(("fixed_quad", j))
        pos = quad_end_positions[j]
    elements.append(("drift", screen_position - pos))
    return elements

def _scan_quad_to_screen_matrix_thin(k_scan_value, k_fixed_by_index, scan_quad_index,
                                      quad_L=QUAD_LENGTH,
                                      quad_end_positions=QUAD_END_POSITIONS,
                                      screen_position=SCREEN_POSITION):
    """
    Same beamline chain as `_scan_quad_to_screen_matrix` (scanned quad plus
    any downstream fixed quads/drifts, upstream quads ignored), but every
    quad -- scanned and fixed -- is approximated as a thin lens of strength
    k*quad_L instead of the thick trig/hyperbolic matrix.
    """
    elements = _downstream_chain(scan_quad_index, quad_L, quad_end_positions, screen_position)

    M = np.eye(2)
    for kind, val in elements:
        if kind == "scan_quad":
            m = _thin_quad_matrix(k_scan_value, quad_L)
        elif kind == "fixed_quad":
            m = _thin_quad_matrix(k_fixed_by_index[val], quad_L)
        else:  # drift
            m = _drift_matrix(val)
        M = m @ M
    return M


def quad_scan_emittance_thinlens(k_scan, sigma, scan_quad_index, k_fixed_downstream=None,
                                  quad_L=QUAD_LENGTH,
                                  quad_end_positions=QUAD_END_POSITIONS,
                                  screen_position=SCREEN_POSITION,
                                  energy=None):
    """
    Parameters
    ----------
    k_scan : array-like, shape (N,)
        Signed k [m^-2] of the scanned quad at each scan point.
    sigma : array-like, shape (N,)
        Measured beam sizes [m] at the screen.
    scan_quad_index : int
        0 -> 760, 1 -> 765, 2 -> 770.
    k_fixed_downstream : dict, optional
        {quad_index: k_value_or_array} for quads DOWNSTREAM of the scanned
        one only. Values can be scalars or shape-(N,) arrays.
    energy : float, optional
        Beam energy [MeV] for normalized emittance.

    Returns
    -------
    dict with sigma11_quad, sigma12_quad, sigma22_quad (at the entrance of
    the scanned quad), emittance, alpha, beta, gamma, and emittance_n if
    energy given.
    """
    k_scan = np.asarray(k_scan, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    N = len(k_scan)
    sigma_sq = sigma**2

    if k_fixed_downstream is None:
        k_fixed_downstream = {}
    k_fixed_arrays = {
        idx: np.broadcast_to(np.asarray(v, dtype=float), (N,))
        for idx, v in k_fixed_downstream.items()
    }

    design = np.zeros((N, 3))
    for i in range(N):
        k_fixed_i = {idx: arr[i] for idx, arr in k_fixed_arrays.items()}
        M = _scan_quad_to_screen_matrix_thin(k_scan[i], k_fixed_i, scan_quad_index,
                                              quad_L, quad_end_positions, screen_position)
        M11, M12 = M[0, 0], M[0, 1]
        design[i, 0] = M11**2
        design[i, 1] = 2 * M11 * M12
        design[i, 2] = M12**2

    sol, residuals, rank, sv = np.linalg.lstsq(design, sigma_sq, rcond=None)
    sigma11_quad, sigma12_quad, sigma22_quad = sol

    # Geometric emittance: eps^2 = sigma11*sigma22 - sigma12^2
    eps_sq = sigma11_quad * sigma22_quad - sigma12_quad**2
    if eps_sq < 0:
        raise ValueError(
            "Negative value under sqrt for emittance "
            f"(sigma11*sigma22 - sigma12^2 = {eps_sq:.3e}). "
            "Check fit quality, sign convention of k, geometry, or the "
            "downstream fixed-k values."
        )
    emittance = np.sqrt(eps_sq)

    # Twiss parameters at the quad location
    alpha = -sigma12_quad / emittance
    beta = sigma11_quad / emittance
    gamma = sigma22_quad / emittance

    result = {
        "sigma11_quad": sigma11_quad,
        "sigma12_quad": sigma12_quad,
        "sigma22_quad": sigma22_quad,
        "emittance": emittance,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    if energy is not None:
        beta_rel = np.sqrt(1 - (0.511 / energy)**2)  # MeV/c^2 for electron rest mass
        gamma_rel = energy / 0.511  # MeV/c^2 for electron
        result["emittance_n"] = beta_rel * gamma_rel * emittance

    return result




def _quad_matrix(k, L):
    if k > 0:
        sqk = np.sqrt(k)
        c, s = np.cos(sqk * L), np.sin(sqk * L)
        return np.array([[c, s / sqk], [-sqk * s, c]])
    elif k < 0:
        sqk = np.sqrt(-k)
        c, s = np.cosh(sqk * L), np.sinh(sqk * L)
        return np.array([[c, s / sqk], [sqk * s, c]])
    else:
        return np.array([[1.0, L], [0.0, 1.0]])





def _scan_quad_to_screen_matrix(k_scan_value, k_fixed_by_index, scan_quad_index,
                                 quad_L=QUAD_LENGTH,
                                 quad_end_positions=QUAD_END_POSITIONS,
                                 screen_position=SCREEN_POSITION):
    """
    Transfer matrix from the ENTRANCE of the scanned quad to the screen,
    for a single scan point. Only the scanned quad plus any downstream
    fixed quads/drifts are used -- upstream quads are irrelevant.

    k_fixed_by_index : dict, e.g. {1: k765_value, 2: k770_value} when
        scan_quad_index == 0. Only downstream indices need to be present.
    """
    elements = _downstream_chain(scan_quad_index, quad_L, quad_end_positions, screen_position)
    M = np.eye(2)
    for kind, val in elements:
        if kind == "scan_quad":
            m = _quad_matrix(k_scan_value, quad_L)
        elif kind == "fixed_quad":
            m = _quad_matrix(k_fixed_by_index[val], quad_L)
        else:  # drift
            m = _drift_matrix(val)
        M = m @ M

    return M


def quad_scan_emittance_thick(k_scan, sigma, scan_quad_index, k_fixed_downstream=None,
                               quad_L=QUAD_LENGTH,
                               quad_end_positions=QUAD_END_POSITIONS,
                               screen_position=SCREEN_POSITION,
                               energy=None):
    """
    Reconstruct the beam sigma matrix at the ENTRANCE of the scanned quad
    (760, 765, or 770), correctly ignoring any quads upstream of it.

    Parameters
    ----------
    k_scan : array-like, shape (N,)
        Signed k [m^-2] of the scanned quad at each scan point.
    sigma : array-like, shape (N,)
        Measured beam sizes [m] at the screen.
    scan_quad_index : int
        0 -> 760, 1 -> 765, 2 -> 770.
    k_fixed_downstream : dict, optional
        {quad_index: k_value_or_array} for quads DOWNSTREAM of the scanned
        one only. E.g. scanning 760 -> {1: k765, 2: k770}; scanning 765 ->
        {2: k770}; scanning 770 -> {} or None (nothing needed).
        Values can be scalars or shape-(N,) arrays.
    energy : float, optional
        Beam energy [MeV] for normalized emittance.

    Returns
    -------
    dict with sigma11_quad, sigma12_quad, sigma22_quad (at the entrance of
    the SCANNED quad -- not a fixed common reference), emittance, alpha,
    beta, gamma, and emittance_n if energy given.
    """
    k_scan = np.asarray(k_scan, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    N = len(k_scan)
    sigma_sq = sigma**2

    if k_fixed_downstream is None:
        k_fixed_downstream = {}
    # broadcast any scalar fixed-k entries to length N for uniform indexing
    k_fixed_arrays = {
        idx: np.broadcast_to(np.asarray(v, dtype=float), (N,))
        for idx, v in k_fixed_downstream.items()
    }
 

    design = np.zeros((N, 3))
    for i in range(N):
        k_fixed_i = {idx: arr[i] for idx, arr in k_fixed_arrays.items()} #create dict of index and strength of fixed quad
        M = _scan_quad_to_screen_matrix(k_scan[i], k_fixed_i, scan_quad_index, 
                                         quad_L, quad_end_positions, screen_position)#compute transfer matrix for each scanned 
        M11, M12 = M[0, 0], M[0, 1]
        design[i, 0] = M11**2
        design[i, 1] = 2 * M11 * M12
        design[i, 2] = M12**2

    sol, residuals, rank, sv = np.linalg.lstsq(design, sigma_sq, rcond=None) #matrix inversion
    sigma11_quad, sigma12_quad, sigma22_quad = sol

    eps_sq = sigma11_quad * sigma22_quad - sigma12_quad**2
    if eps_sq < 0:
        raise ValueError(
            "Negative value under sqrt for emittance "
            f"(sigma11*sigma22 - sigma12^2 = {eps_sq:.3e}). "
            "Check fit quality, sign convention of k, geometry, or the "
            "downstream fixed-k values."
        )
    emittance = np.sqrt(eps_sq)

    alpha = -sigma12_quad / emittance
    beta = sigma11_quad / emittance
    gamma = sigma22_quad / emittance

    result = {
        "sigma11_quad": sigma11_quad,
        "sigma12_quad": sigma12_quad,
        "sigma22_quad": sigma22_quad,
        "emittance": emittance,
        "beta": beta,
        "alpha": alpha,
        "gamma": gamma,
    }

    if energy is not None:
        beta_rel = np.sqrt(1 - (0.511 / energy)**2)
        gamma_rel = energy / 0.511
        result["emittance_n"] = beta_rel * gamma_rel * emittance

    return result

def propagate_sigma_matrix(sigma_matrix, M):
    """sigma_matrix at point A -> sigma_matrix at point B, given transfer M (A->B)."""
    return M @ sigma_matrix @ M.T

def backpropagate_to_reference(sigma_at_scan_quad, scan_quad_index, k_fixed_upstream,
                                 quad_L=QUAD_LENGTH,
                                 quad_end_positions=QUAD_END_POSITIONS):
    """
    Re-express a fitted sigma matrix (defined at the entrance of the scanned
    quad) at the entrance of quad 760, using the KNOWN fixed k of whatever
    quads sit between 760 and the scanned quad. No-op if scan_quad_index==0.
    """
    if scan_quad_index == 0:
        return sigma_at_scan_quad

    quad_end_positions = np.asarray(quad_end_positions, dtype=float)
    quad_start_positions = quad_end_positions - quad_L

    M = np.eye(2)
    pos = quad_start_positions[0]
    for j in range(scan_quad_index):
        gap = quad_start_positions[j] - pos
        M = _drift_matrix(gap) @ M
        M = _quad_matrix(k_fixed_upstream[j], quad_L) @ M
        pos = quad_end_positions[j]
    M = _drift_matrix(quad_start_positions[scan_quad_index] - pos) @ M

    Minv = np.linalg.inv(M)
    return Minv @ sigma_at_scan_quad @ Minv.T


def quad_scan_fit1( P_ref, s_x, s_y, current_setpoints, screen ="CA.BTV0875", reconstruction_point = 'CA.QFD0765'):

    n = len(s_x)
    results = {}
    

    lattices = []
    beta_x_matrix = np.empty((n, 3))
    beta_y_matrix = np.empty((n, 3))
    for i in range(n):
        quad_currents = current_setpoints[i]

        lattice = get_lattice(reconstruction_point, screen, P_ref, quad_currents)
        # lattice = get_lattice(reconstruction_point, "CA.BTV0390", P_ref, quad_currents)

        lattices.append(lattice)

        # Transfer matricies from twiss to beta for each current
        beta_x_matrix[i], beta_y_matrix[i] = np.array(lattice.get_twiss_matrix())[:, 0, :]

    g = lorentz_factor(P_ref)
    
    # Get the beam size squared
    sigma_sq = lambda beta_matrix, emitt, beta, alpha: emitt/g*beta_matrix@np.array([beta, alpha, (1+alpha**2)/beta])

    try:
        twiss_x_fit, twiss_x_cov, info_x = curve_fit(
            f=sigma_sq,
            xdata=beta_x_matrix, 
            ydata=s_x**2, 
            p0=(5, 10, 0), 
            absolute_sigma=False, # Increases the estimated variance by a factor chi2/dof, i.e. increases the variance for a bad fit
            full_output=True
        )[0:3]

        twiss_x, twiss_x_std = full_twiss(twiss_x_fit, twiss_x_cov)
        chi2_x = np.sum(info_x['fvec']**2)
    except (RuntimeError, ValueError):
        # Fit did not converge
        twiss_x = twiss_x_std = (np.nan, np.nan, np.nan, np.nan)
        chi2_x = np.nan

    
    try:
        twiss_y_fit, twiss_y_cov, info_y = curve_fit(
            f=sigma_sq,
            xdata=beta_y_matrix, 
            ydata=s_y**2, 
            p0=(5, 10, 0), 
            absolute_sigma=False, # Increases the estimated variance by a factor chi2/dof, i.e. increases the variance for a bad fit
            full_output=True
        )[0:3]

        twiss_y, twiss_y_std = full_twiss(twiss_y_fit, twiss_y_cov)
        #twiss x and y return np.array([emitt_n, beta, alpha, gamma]), np.array([emitt_std, beta_std, alpha_std, gamma_std])
        chi2_y = np.sum(info_y['fvec']**2)
    except (RuntimeError, ValueError):
        # Fit did not converge
        twiss_y = twiss_y_std = (np.nan, np.nan, np.nan, np.nan)
        chi2_y = np.nan

    dof = n - 3
    chi2_reduced = (chi2_x/dof, chi2_y/dof)

    twiss = (*twiss_x, *twiss_y)
    twiss_std = (*twiss_x_std, *twiss_y_std)

    results[reconstruction_point] = {'twiss': twiss, 'twiss_std': twiss_std, 'chi2_reduced': chi2_reduced, 'lattices': lattices,}
    return twiss_x,twiss_y, twiss_std, chi2_reduced, lattices