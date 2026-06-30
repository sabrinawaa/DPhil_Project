
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uniformity_fit import *
from flatness import *


def beam_centroid(img,):
    # coordinate grids
    y, x = np.indices(img.shape)

    I = img.astype(float)

    total_intensity = np.nansum(I)

    if total_intensity == 0:
        return None, None  # or fallback to max pixel

    x_centroid = np.nansum(I * x) / total_intensity
    y_centroid = np.nansum(I * y) / total_intensity

    return int(x_centroid), int(y_centroid)

def beam_area_mask(img, fraction, cx,cy):
    peak_val = np.nanmax(img)
    threshold = fraction * peak_val

    # mask of pixels above threshold
    mask = img >= threshold

    return mask

def rFerr(sig, p, sigerr, perr, f=0.95):
            # print('Errs: serr='+str(serr)+',perr='+str(perr))
            lTerm = np.log(1/f)
            d1 = np.sqrt(2)*lTerm**(1/(2*p))
            d2 = (lTerm**(1/(2*p))*sig*np.log(lTerm))/(np.sqrt(2)*p**2)
            return np.sqrt(d1**2*sigerr**2+d2**2*perr**2)



def beam_area_mask(img, fraction, cx,cy):
    peak_val = np.nanmax(img)
    threshold = fraction * peak_val

    # mask of pixels above threshold
    mask = img >= threshold

    return mask

def plot_dose1(dosemap, im_type, cx=None, cy=None):
    h, w = dosemap.shape
    strip_width = 10
    r_mm = 2.0

    if im_type == 'RCF':
        pixel_calibration = 0.08467  # mm/pixel
    elif im_type == "YAG":
        pixel_calibration = 0.0378 # mm/pixel
    else:
        raise ValueError("Unknown im_type")

    # Beam centroid in pixel coordinates， use for initial guess (maybe just centre of screen is enough)
    if cx is None or cy is None:
        cx, cy = beam_centroid(dosemap)

    # Centered physical coordinates in mm
    x = (np.arange(w) - cx) * pixel_calibration
    y = (np.arange(h) - cy) * pixel_calibration
    xx, yy = np.meshgrid(x, y)
    X = np.vstack((xx.ravel(), yy.ravel()))
    Z = dosemap.ravel()
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    # Fit the slices
    A0 = float(np.nanmax(dosemap) - np.nanmin(dosemap))
    c0 = float(np.nanmin(dosemap))
    p0 = [A0, 0.0, 0.0, 5.0, 5.0, 2.0, 2.0, 0.0, 0.0, c0]

    lower = [0.0, -np.inf, -np.inf, 1e-6, 1e-6, 1.0, 1.0, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf, 20.0, 20.0, np.inf, np.inf, np.inf]
    

    # lower = [0, -np.inf, 1e-6, 1.0, -np.inf, -np.inf]
    # upper = [np.inf, np.inf, 20, np.inf, np.inf, np.inf]

    # params_x, cov_x = curve_fit(supergaussian1D_skewed, x, slice_row, p0=p0, bounds=(lower, upper))
    # params_y, cov_y = curve_fit(supergaussian1D_skewed, y, slice_col, p0=p0, bounds=(lower, upper))
    params, cov = curve_fit(
        supergaussian2D_skewed,
        X,Z,p0=p0,
        bounds=(lower, upper))
    
    A, x0, y0, sig_x, sig_y, P_x, P_y, mx, my, c = params
    err = np.sqrt(np.diag(cov))

    # sig_x, sig_y = params_x[2], params_y[2]
    # errSigx, errSigy = np.sqrt(np.diag(cov_x)[2]), np.sqrt(np.diag(cov_y)[2])

    # P_x, P_y = params_x[3], params_y[3]
    # errPx_fit, errPy_fit = np.sqrt(np.diag(cov_x)[3]), np.sqrt(np.diag(cov_y)[3])

    errSigx, errSigy, errP_x, errP_y = err[3], err[4], err[5], err[6]
    x90_x = abs(x90(sig_x, P_x))
    x90_y = abs(x90(sig_y, P_y))

    err_x90_x = rFerr(sig_x, P_x, errSigx, errP_x, 0.9)
    err_x90_y = rFerr(sig_y, P_y, errSigy, errP_y, 0.9)

    # Update center from fitted superGaussian center
    cx = int(cx + x0 / pixel_calibration)
    cy = int(cy + y0 / pixel_calibration)

    new_x = (np.arange(w) - cx) * pixel_calibration #recentre x and y coordinates to fitted centre
    new_y = (np.arange(h) - cy) * pixel_calibration
    xx, yy = np.meshgrid(new_x, new_y)
    circle_mask = (xx**2 + yy**2) <= r_mm**2

    if np.any(circle_mask):
        dose_centre = float(np.mean(dosemap[circle_mask]))
        dose_std = float(np.std(dosemap[circle_mask]))
    else:
        dose_centre = np.nan
        dose_std = np.nan

    row0 = max(0, cy - strip_width // 2)
    row1 = min(h, cy + strip_width // 2)
    col0 = max(0, cx - strip_width // 2)
    col1 = min(w, cx + strip_width // 2)

    slice_row = np.mean(dosemap[row0:row1, :], axis=0)
    slice_col = np.mean(dosemap[:, col0:col1], axis=1)

    lower = [0.0,    0.0,   0.0, -np.inf,   -np.inf, -np.inf]
    upper = [np.inf, 10.0, 10.0, np.inf, np.inf, np.inf]
   
    p00 = [np.max(slice_row), (sig_x+sig_y)/2 *1.1, (sig_x+sig_y)/2, 0, 0, 0]
    try:
        params_xx, cov_xx = curve_fit(sum_2gaussians_skewed, new_x, slice_row, p0=p00, bounds=(lower, upper),maxfev=20000)
    except Exception as e:
        print(f"Failed to fit sum of 2 Gaussians: {e}")
        traceback.print_exc()
    try:
        params_yy, cov_yy = curve_fit(sum_2gaussians_skewed, new_y, slice_col, p0=p00, bounds=(lower, upper),maxfev=20000)
    except Exception as e:
        print(f"Failed to fit sum of 2 Gaussians: {e}")
        traceback.print_exc()

    err_dose = np.sqrt(dose_std**2 + 0) #add contribution from dosimetry

    fig, ax_main = plt.subplots(figsize=(10, 6))
    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")

    im = ax_main.imshow( dosemap, origin="lower", aspect="equal", cmap="viridis", extent=[new_x[0], new_x[-1], new_y[0], new_y[-1]] )
    circle = Circle((0, 0), r_mm, fill=False, linestyle=":", linewidth=1.8, edgecolor="white")
    ax_main.add_patch(circle)
    if im_type == "RCF":
        ax_main.text(
            0.35, -3,
            f"Mean dose =\n{dose_centre:.2f} ± {dose_std:.2f} Gy",
            color="white",fontsize=10,va="center",ha="left",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2)
        )
    elif im_type == "YAG":
        ax_main.text(
            0.35, -3,
            f"Mean CD =\n{dose_centre:.2f} ± {dose_std:.2f} pC/mm²",
            color="white",fontsize=10,va="center",ha="left",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
        
        
    # Axes for slices and colorbar
    divider = make_axes_locatable(ax_main) 
    ax_x = divider.append_axes("top", size="25%", pad=0.1, sharex=ax_main) 
    ax_y = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main) 
    cax = divider.append_axes("right", size="2%", pad=0.1)
    #


    # Top plot: X slice
    ax_x.bar(new_x, slice_row, width=dx, alpha=0.7)
    
    ax_x.plot( new_x,supergaussian1D_skewed(new_x, A,0,sig_x, P_x, mx, c),  'r-',label=f"SuperGaussian Fit (P={P_x:.2f}, x90={x90_x:.2f})")
    ax_x.plot(new_x, sum_2gaussians_skewed(new_x, *params_xx), 'b-', label=f"2-Gaussian Fit (x0/sigma={abs(params_xx[1])/params_xx[2]:.2f})")
    if im_type == 'YAG':
        ax_x.set_ylabel("CD [pC/mm²]")
    elif im_type == 'RCF':
        ax_x.set_ylabel("Dose (Gy)")
    ax_x.legend(loc='lower left')
    plt.setp(ax_x.get_xticklabels(), visible=False)

    # Right plot: Y slice
    ax_y.barh(new_y, slice_col, height=dy, alpha=0.7)
    ax_y.plot(supergaussian1D_skewed(new_y, A,0,sig_y, P_y, my, c), new_y,  'r-',label=f"SuperGaussian Fit (P={P_y:.2f}, x90={x90_y:.2f})")
    ax_y.plot(sum_2gaussians_skewed(new_y, *params_yy),new_y, 'b-', label=f"2-Gaussian Fit (x0/sigma={abs(params_yy[1])/params_yy[2]:.2f})")
    if im_type == 'YAG':
        ax_y.set_xlabel("CD [pC/mm²]")
    elif im_type == 'RCF':
        ax_y.set_xlabel("Dose (Gy)")
    ax_y.legend(loc='lower right')
    plt.setp(ax_y.get_yticklabels(), visible=False)

    # Colorbar
    if im_type == 'YAG':
        fig.colorbar(im, cax=cax, orientation='vertical', label="Charge Density [pC/mm²]")
    elif im_type == 'RCF':
        fig.colorbar(im, cax=cax, orientation='vertical', label="Dose (Gy)")
    
    # Centered slice bands on main image
    rect_h = Rectangle(
        (new_x[0], new_y[row0]),
        new_x[-1] - new_x[0],
        (row1 - row0) * pixel_calibration,
        edgecolor="white", facecolor="none",linewidth=1.5,linestyle="--")

    rect_v = Rectangle(
        (new_x[col0], new_y[0]),
        (col1 - col0) * pixel_calibration,
        new_y[-1] - new_y[0],
        edgecolor="white",facecolor="none",linewidth=1.5,linestyle="--")

    ax_main.add_patch(rect_h)
    ax_main.add_patch(rect_v)

    # Make sure shared limits align
    ax_x.set_xlim(ax_main.get_xlim())
    ax_y.set_ylim(ax_main.get_ylim())
    return fig, cx, cy, dose_centre, P_x, P_y, x90_x, x90_y, abs(params_xx[1])/params_xx[2], abs(params_yy[1])/params_yy[2], err_dose, errP_x, errP_y, err_x90_x, err_x90_y
