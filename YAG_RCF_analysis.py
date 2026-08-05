
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uniformity_fit import *
from flatness import *
import matplotlib.transforms as transforms

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



def plot_dose1(dosemap, im_type, x, y,strip_width=10, cx = None, cy = None,p0=None, p00 = None):
    """
    dosemap: 2D array, dosemap[row, col] with row <-> y, col <-> x
    x, y: 1D physical-coordinate arrays (mm), already pixel-calibrated by the
          caller (len(x) == dosemap.shape[1], len(y) == dosemap.shape[0]).
    """
    h, w = dosemap.shape
    r_mm = 2.0

    # Beam centroid in pixel/array-index coordinates, used for initial guess (maybe just centre of screen is enough)
    if cx is not None and cy is not None:
        cx_idx, cy_idx = cx, cy
    else:
        cx_idx, cy_idx = beam_centroid(dosemap)
    x, y = x - x[cx_idx], y - y[cy_idx]

    xx, yy = np.meshgrid(x, y)
    X = np.vstack((xx.ravel(), yy.ravel()))
    Z = dosemap.ravel()
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    # Fit the slices
    A0 = float(np.nanmax(dosemap) - np.nanmin(dosemap))
    c0 = float(np.nanmin(dosemap))
    p0_2d = [A0, 0.0, 0.0, 5.0, 5.0, 2.0, 2.0, 0.0, 0.0, c0]

    lower = [0.0, -np.inf, -np.inf, 1e-6, 1e-6, 1.0, 1.0, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf, 20.0, 20.0, np.inf, np.inf, np.inf]
    
    params_2d, _ = curve_fit(
        supergaussian2D_skewed,
        X,Z,p0=p0_2d,
        bounds=(lower, upper))
    
    A, x0, y0, sig_x2d, sig_y2d, P_x2d, P_y2d, mx, my, c = params_2d

    if cx is not None and cy is not None:
        # Centre forced by the caller: ignore the fitted offset entirely
        new_x, new_y = x, y
        new_cx_idx, new_cy_idx = cx_idx, cy_idx
    else:
        # Recentre x and y coordinates (mm) on the fitted superGaussian centre
        new_x, new_y = x - x0, y - y0

        # Convert the fitted centre offset (mm) back to an array index using the
        # pixel spacing (mm/pixel) so it stays in the same units as cx_idx/cy_idx
        new_cx_idx = int(round(cx_idx + x0 / dx))
        new_cy_idx = int(round(cy_idx + y0 / dy))
    xx, yy = np.meshgrid(new_x, new_y)
    circle_mask = (xx**2 + yy**2) <= r_mm**2

    if np.any(circle_mask):
        dose_centre = float(np.mean(dosemap[circle_mask]))
        dose_std = float(np.std(dosemap[circle_mask]))
    else:
        dose_centre = np.nan
        dose_std = np.nan

    row0 = max(0, new_cy_idx - strip_width // 2) #indices
    row1 = min(h, new_cy_idx + strip_width // 2)
    col0 = max(0, new_cx_idx - strip_width // 2)
    col1 = min(w, new_cx_idx + strip_width // 2)

    slice_row = np.mean(dosemap[row0:row1, :], axis=0)
    slice_col = np.mean(dosemap[:, col0:col1], axis=1)
    # supergaussian 1d fits
    lower = [0, -np.inf, 1e-6, 0.0, -np.inf, -np.inf]
    upper = [np.inf, np.inf, 40, np.inf, np.inf, np.inf]

    if p0 is None:
        p0 = [A, 0, abs(sig_x2d), P_x2d, 0, c]
    params_x, cov_x = curve_fit(supergaussian1D_skewed, new_x, slice_row, p0=p0, bounds=(lower, upper))
    params_y, cov_y = curve_fit(supergaussian1D_skewed, new_y, slice_col, p0=p0, bounds=(lower, upper))

    sig_x, sig_y = params_x[2], params_y[2]
    errSigx, errSigy = np.sqrt(np.diag(cov_x)[2]), np.sqrt(np.diag(cov_y)[2])

    P_x, P_y = params_x[3], params_y[3]
    err_Px, err_Py = np.sqrt(np.diag(cov_x)[3]), np.sqrt(np.diag(cov_y)[3])

    x90_x = abs(x90(sig_x, P_x))
    x90_y = abs(x90(sig_y, P_y))

    err_x90_x = rFerr(sig_x, P_x, errSigx, err_Px, 0.9)
    err_x90_y = rFerr(sig_y, P_y, errSigy, err_Py, 0.9)

    # 2-Gaussian fits
    lower = [0.0,    0.0,   1e-6, -np.inf,   -np.inf, -np.inf]
    upper = [np.inf, 30.0, 30.0, np.inf, np.inf, np.inf]
    if p00 is None:
        p00 = [np.max(slice_row), np.std(slice_row)*1.1, np.std(slice_row), 0, 0, 0]

    plot_2g_x = P_x < 2.1
    plot_2g_y = P_y < 2.1

    if plot_2g_x:
        try:
            params_xx, cov_xx = curve_fit(sum_2gaussians_skewed, new_x, slice_row, p0=p00, bounds=(lower, upper), maxfev=20000)
            err_ratio_x = np.sqrt((np.diag(cov_xx)[1]/params_xx[2])**2 + (np.diag(cov_xx)[2]*params_xx[1]/params_xx[2]**2)**2)
        except Exception as e:
            print(f"Failed to fit sum of 2 Gaussians (x): {e}")
            traceback.print_exc()
            params_xx = np.full(6, np.nan)
            err_ratio_x = np.nan
            plot_2g_x = False
    else:
        params_xx = np.full(6, np.nan)
        err_ratio_x = np.nan

    if plot_2g_y:
        try:
            params_yy, cov_yy = curve_fit(sum_2gaussians_skewed, new_y, slice_col, p0=p00, bounds=(lower, upper), maxfev=20000)
            err_ratio_y = np.sqrt((np.diag(cov_yy)[1]/params_yy[2])**2 + (np.diag(cov_yy)[2]*params_yy[1]/params_yy[2]**2)**2)
        except Exception as e:
            print(f"Failed to fit sum of 2 Gaussians (y): {e}")
            traceback.print_exc()
            params_yy = np.full(6, np.nan)
            err_ratio_y = np.nan
            plot_2g_y = False
    else:
        params_yy = np.full(6, np.nan)
        err_ratio_y = np.nan

    err_dose = np.sqrt(dose_std**2 + 0) #add contribution from dosimetry

    fig, ax_main = plt.subplots(figsize=(10, 6))
    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")

    im = ax_main.imshow(dosemap, origin="lower", aspect="equal", cmap="viridis", extent=[new_x[0], new_x[-1], new_y[0], new_y[-1]] )
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
    ax_x.plot( new_x,supergaussian1D_skewed(new_x, *params_x),  'r-',label=f"SuperGaussian Fit (P={P_x:.2f}, x90={x90_x:.2f})")
    if plot_2g_x:
        ax_x.plot(new_x, sum_2gaussians_skewed(new_x, *params_xx), 'b-', label=f"2-Gaussian Fit (x0/sigma={abs(params_xx[1])/params_xx[2]:.2f})")
    if im_type == 'YAG':
        ax_x.set_ylabel("CD [pC/mm²]")
    elif im_type == 'RCF':
        ax_x.set_ylabel("Dose (Gy)")
    ax_x.legend(loc='lower left')
    plt.setp(ax_x.get_xticklabels(), visible=False)

    # Right plot: Y slice
    ax_y.barh(new_y, slice_col, height=dy, alpha=0.7)
    ax_y.plot(supergaussian1D_skewed(new_y, *params_y), new_y,  'r-',label=f"SuperGaussian Fit (P={P_y:.2f}, x90={x90_y:.2f})")
    if plot_2g_y:
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
        (row1 - row0) * dy,
        edgecolor="white", facecolor="none",linewidth=1.5,linestyle="--")

    rect_v = Rectangle(
        (new_x[col0], new_y[0]),
        (col1 - col0) * dx,
        new_y[-1] - new_y[0],
        edgecolor="white",facecolor="none",linewidth=1.5,linestyle="--")

    ax_main.add_patch(rect_h)
    ax_main.add_patch(rect_v)

    # Make sure shared limits align
    ax_x.set_xlim(ax_main.get_xlim())
    ax_y.set_ylim(ax_main.get_ylim())
    return fig, new_cx_idx, new_cy_idx, dose_centre, P_x, P_y, x90_x, x90_y, abs(params_xx[1])/params_xx[2], abs(params_yy[1])/params_yy[2], err_dose, err_Px, err_Py, err_x90_x, err_x90_y, err_ratio_x, err_ratio_y


def plot_vs_x(
    df,
    x_col,
    y_cols,
    err_cols=None,
    y_labels=None,
    group_col="beam_type",      # col / uncol
    data_col="data_kind",       # exp / sim
    groups=("col", "uncol"),
    data_kinds=("exp", "sim"),
    markers=None,
    min_vals=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(8, 5),
    hline=None,
    region_labels=None,
    sim_alpha=0.5,
    sim_size=3,
):
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if err_cols is None:
        err_cols = [None] * len(y_cols)
    elif isinstance(err_cols, str):
        err_cols = [err_cols]

    if y_labels is None:
        y_labels = y_cols

    if min_vals is None:
        min_vals = [None] * len(y_cols)
    elif not isinstance(min_vals, (list, tuple)):
        min_vals = [min_vals] * len(y_cols)

    # Fix 1: markers must cover every y_col, not just the first len(markers) of them.
    # Default marker cycle, extended/truncated to match y_cols exactly.
    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    if markers is None:
        markers = default_markers
    if len(markers) < len(y_cols):
        raise ValueError(
            f"Got {len(y_cols)} y_cols but only {len(markers)} markers; "
            "pass an explicit `markers` list at least as long as y_cols."
        )
    markers = markers[: len(y_cols)]

    # Fix (crash): groups / data_kinds must be iterable even when the
    # corresponding *_col is None (i.e. "don't split on this dimension").
    if groups is None:
        groups = [None]
    if data_kinds is None:
        data_kinds = [None]

    fig, ax = plt.subplots(figsize=figsize)

    for group in groups:
        # Fix 3: one color per (group, y_col) pair, shared between that
        # series' exp and sim, but distinct across different y_cols so
        # e.g. r90_x and r90_y don't collide.
        series_colors = {}

        for data_kind in data_kinds:
            # Fix 4: no need to copy the whole df before filtering; boolean
            # indexing already returns a new object.
            sub = df
            if group_col is not None:
                sub = sub[sub[group_col] == group]
            if data_col is not None:
                sub = sub[sub[data_col] == data_kind]

            if sub.empty:
                continue

            for y_col, err_col, y_label, marker, min_val in zip(
                y_cols, err_cols, y_labels, markers, min_vals
            ):
                plot_data = sub if min_val is None else sub[sub[y_col] > min_val]
                if plot_data.empty:
                    continue

                plot_data = plot_data.sort_values(x_col)
                x = plot_data[x_col]
                y = plot_data[y_col]

                err_note = f"±{err_col}" if err_col is not None else "no err"
                label = f"{group} {y_label} ({data_kind}, {err_note})"

                color = series_colors.get(y_col)  # None on first use for this series

                # Fix 6: when data_col/data_kinds is None there's no exp/sim
                # distinction at all — treat that case as a plain errorbar
                # series instead of silently matching neither branch below.
                if data_kind is None or data_kind == "exp":
                    yerr = plot_data[err_col] if err_col is not None else None
                    line = ax.errorbar(
                        x, y,
                        yerr=yerr,
                        marker=marker,
                        linestyle="--",
                        capsize=4,
                        label=label,
                        color=color,  # None -> matplotlib picks the next cycle color
                    )
                    if color is None:
                        series_colors[y_col] = line.lines[0].get_color()

                elif data_kind == "sim":
                    sc = ax.scatter(
                        x, y,
                        s=sim_size,
                        marker=marker,
                        label=label,
                        zorder=3,
                        color=color,  # None -> matplotlib picks the next cycle color
                    )
                    if color is None:
                        color = sc.get_facecolor()[0]
                        series_colors[y_col] = color

                    if err_col is not None:
                        yerr = plot_data[err_col]
                        ax.fill_between(
                            x,
                            y - yerr,
                            y + yerr,
                            color=color,
                            alpha=sim_alpha,
                            linewidth=0,
                            zorder=2,
                        )

    if hline is not None:
        ax.axhline(hline, color="k", linestyle=":", linewidth=1.5)
        if region_labels is not None:
            above_label, below_label = region_labels
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            y_min, y_max = ax.get_ylim()
            pad = 0.03 * (y_max - y_min)

            if above_label:
                ax.text(
                    0.98, hline + pad, above_label,
                    transform=trans, ha="right", va="bottom",
                    fontsize=9, color="dimgray"
                )
            if below_label:
                ax.text(
                    0.98, hline - pad, below_label,
                    transform=trans, ha="right", va="top",
                    fontsize=9, color="dimgray"
                )

    ax.set_xlabel(xlabel or x_col.capitalize())
    ax.set_ylabel(ylabel or "")
    ax.set_title(title or f"Evolution with {x_col}")
    ax.grid(True, alpha=0.3)

    # Fix 2: only draw a legend if something was actually plotted.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fig.tight_layout()
    plt.show()




