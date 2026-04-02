import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from topasToDose import getDosemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

#characterise uniformity
def supergaussian(x, y, A, x0, y0, sigma_x, sigma_y, P):
    return A * np.exp(-( (x-x0)**2 /(2*sigma_x**2) + (y-y0)**2 /(2*sigma_y**2))**P)


def supergaussian1D(x, A, x0, sigma_x, P):
    return A * np.exp(-( (x-x0)**2 /(2*sigma_x**2) )**P)


def MSE(params, x, y, dose_map):
    model = supergaussian(x, y, *params)
    return np.sum((model - dose_map)**2)  # Mean squared error

def r90(sig,P):
    return sig * np.sqrt(2) * (-np.log(0.9))**(1/(2*P))

def plotDoseMap(x, y, doseMap,fitted_map,P,sig,r_90, depth,output_filename):
    dose_y = fitted_map[fitted_map.shape[0] // 2, :]  # Middle row (horizontal slice)
    dose_x = fitted_map[:, fitted_map.shape[1] // 2]  # Middle column (vertical slice)

    orig_dose_y = doseMap[doseMap.shape[0] // 2, :]  # Middle row (horizontal slice)
    orig_dose_x = doseMap[:, doseMap.shape[1] // 2]  # Middle column (vertical slice)

    # Create the figure with subplots
    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(4, 5, hspace=0.4, wspace=0.6, width_ratios=[1, 1, 1, 0.1, 1])  
    # Main 2D scatter plot
    ax_main = fig.add_subplot(grid[1:, :-2])  # Use only the first 3 columns (exclude colorbar space)
    im = ax_main.scatter(x, y, c=doseMap, cmap="viridis", s=120)
    ax_main.set_title(f"Super-Gaussian  P = {P:.2f}, sigma = {sig:.2f}, r90 = {r_90:.2f}")
    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")

    # Colorbar (aligned with the main plot)
    ax_cbar = fig.add_subplot(grid[1:, -2])
    cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label("Supergaussian Dose (Gy)")
    print(x[0,:])
    # Histogram along X-axis (aligned only with the main plot)
    ax_x = fig.add_subplot(grid[0, :-2], sharex=ax_main)  # Avoid the last two columns
    # ax_x.bar(x[0,:], dose_x, width=(x[0,1] - x[0,0]), alpha=0.4, label="fitted", color="blue", edgecolor="black")
    ax_x.plot(x[0,:], dose_x, label="fitted", color="blue" )
    ax_x.bar(x[0,:], orig_dose_x, width=(x[0,1] - x[0,0]), alpha=0.4, label="orig", color="green", edgecolor="black")
    ax_x.set_ylabel("Slice Dose (Gy)")
    ax_x.set_title("X Histogram")
    ax_x.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_x.legend()

    # Histogram along Y-axis
    ax_y = fig.add_subplot(grid[1:, -1], sharey=ax_main)

    # ax_y.barh(y[:,0], dose_y, height=(y[1,0] - y[0,0]), alpha=0.4, label="fitted", color="blue", edgecolor="black")
    ax_y.plot(dose_y, y[:,0], label="fitted", color="blue")
    ax_y.barh(y[:,0], orig_dose_y, height=(y[1,0] - y[0,0]), alpha=0.4, label="orig", color="green", edgecolor="black")
    ax_y.set_xlabel("Slice Dose (Gy)")
    ax_y.set_title("Y Histogram")
    ax_y.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_y.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_figs/" +output_filename+ f"Depth{depth}_SupergaussianFit.png")

def fitDoseMap (n_particles, dose_depth,output_filename,acChargenC=10, zoom_factor=1, plot=True):
    x,y, doseMap = getDosemap("DoseAtTank"+str(dose_depth)+ "_"+ output_filename+".csv",n_particles, dose_depth, output_filename, acChargenC, plot = False)
    print(doseMap.shape)
    
    print(x.shape)
    print(y.shape)
    x_center = (x.max() + x.min()) / 2  # Find the midpoint of x
    y_center = (y.max() + y.min()) / 2  # Find the midpoint of y

    # Shift x and y to center them at (0,0)
    x = x - x_center
    y = y - y_center
    x, y = np.meshgrid(x, y)

    lower_index =  int( (len(x) -len(x) * zoom_factor) /2 )# Lower bound for the dose values
    upper_index = int( len(x)- lower_index) # Upper bound for the dose values
    x,y,doseMap = x[lower_index:upper_index, lower_index:upper_index],y[lower_index:upper_index, lower_index:upper_index], doseMap[lower_index:upper_index, lower_index:upper_index]  #only the central 60% of the beam
    #print("average dose",np.mean(doseMap))
    
    p0=[np.max(doseMap), np.mean(x), np.mean(y), np.std(x), np.std(y), 6]
    #curvefit 2d histogram to supergaussian
    # Perform the optimization
    result = minimize(MSE, p0, args=(x, y, doseMap), method='L-BFGS-B')
    fit_params = result.x

    # Calculate the fitted super-Gaussian
    fitted_map = supergaussian(x, y, *fit_params)
    print(fitted_map.shape)
    sig, P = fit_params[3], fit_params[5]
    r_90 = r90(sig,P)
    if plot:
        plotDoseMap(x, y, doseMap,fitted_map,P,sig,r_90, dose_depth, output_filename)
    else:
        return fitted_map, P, sig, r_90
    
def plot_dose(dosemap, x, y, strip_width=10, centred=False,twogaussian=False): #cleaner version with supergaussian and 2gaussian fits
    h, w = dosemap.shape


    # Infer spacing from coordinates
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    # Use image center instead of centroid
    if centred:
        cx_idx = w // 2
        cy_idx = h // 2
        
    else:# Centroid
        total_intensity = np.nansum(dosemap)

        if total_intensity == 0:
            cx_idx,cy_idx, cx, cy =  None, None, None,  None  # or fallback to max pixel
        
        else:
            X, Y = np.meshgrid(x, y, indexing="xy")
            cx = np.nansum(dosemap * X) / total_intensity   # mm
            cy = np.nansum(dosemap * Y) / total_intensity   # mm

            cx_idx = int(np.argmin(np.abs(x - cx)))
            cy_idx = int(np.argmin(np.abs(y - cy)))

    x, y = x-x[cx_idx], y-y[cy_idx]



    # Slice bounds
    row0 = max(0, cy_idx - strip_width // 2)
    row1 = min(h, cy_idx + strip_width // 2)
    col0 = max(0, cx_idx - strip_width // 2)
    col1 = min(w, cx_idx + strip_width // 2)

    # Slices
    slice_row = np.mean(dosemap[row0:row1, :], axis=0)
    slice_col = np.mean(dosemap[:, col0:col1], axis=1)

    fig, ax_main = plt.subplots(figsize=(10, 6))

    # Main image
    im = ax_main.imshow(
        dosemap,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        extent=[x[0], x[-1], y[0], y[-1]]
    )

    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")

    # Layout
    divider = make_axes_locatable(ax_main)
    ax_x = divider.append_axes("top", size="25%", pad=0.1, sharex=ax_main)
    ax_y = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    cax  = divider.append_axes("right", size="2%", pad=0.1)

    # Fit
    p0x = [np.max(slice_row)/2, 0, 5, 3]
    p0y = [np.max(slice_col)/2, 0, 5, 3]
    p00x = [np.max(slice_row)/2, 6, 12]
    p00y = [np.max(slice_col)/2, 6, 12]

    params_x, _ = curve_fit(supergaussian1D, x, slice_row, p0=p0x)
    params_y, _ = curve_fit(supergaussian1D, y, slice_col, p0=p0y)

    params_xx, _ = curve_fit(sum_2gaussians, x, slice_row, p0=p00x)
    params_yy, _ = curve_fit(sum_2gaussians, y, slice_col, p0=p00y)

    sig_x, sig_y = params_x[2], params_y[2]
    P_x, P_y = params_x[3], params_y[3]
    r90_x, r90_y = r90(sig_x, P_x), r90(sig_y, P_y)

    # X histogram
    ax_x.bar(x, slice_row, width=dx, alpha=0.7)
    ax_x.plot(x, supergaussian1D(x, *params_x), 'r-',
              label=f"SuperGaussian (P={params_x[3]:.2f}, r90={r90_x:.2f})")
    if twogaussian:
        ax_x.plot(x, sum_2gaussians(x, *params_xx), 'g-',
                  label=f"2-Gaussian (x0/σ={abs(params_xx[1])/params_xx[2]:.2f})")
    ax_x.set_ylabel("Dose")
    ax_x.legend(loc='lower left')
    plt.setp(ax_x.get_xticklabels(), visible=False)

    # Y histogram
    ax_y.barh(y, slice_col, height=dy, alpha=0.7)
    ax_y.plot(supergaussian1D(y, *params_y), y, 'r-',
              label=f"SuperGaussian (P={params_y[3]:.2f}, r90={r90_y:.2f})")
    if twogaussian:
        ax_y.plot(sum_2gaussians(y, *params_yy), y, 'g-',
                label=f"2-Gaussian (x0/σ={abs(params_yy[1])/params_yy[2]:.2f})")
    ax_y.set_xlabel("Dose")
    ax_y.legend(loc='lower right')
    plt.setp(ax_y.get_yticklabels(), visible=False)

    # Colorbar
    fig.colorbar(im, cax=cax, orientation='vertical', label="Dose (Gy)")

    # Slice rectangles
    rect_h = Rectangle(
        (x[0], y[row0]),
        x[-1] - x[0],
        (row1 - row0) * dy,
        edgecolor="white", facecolor="none", linewidth=1.5, linestyle="--"
    )

    rect_v = Rectangle(
        (x[col0], y[0]),
        (col1 - col0) * dx,
        y[-1] - y[0],
        edgecolor="white", facecolor="none", linewidth=1.5, linestyle="--"
    )

    ax_main.add_patch(rect_h)
    ax_main.add_patch(rect_v)

    # Align axes
    ax_x.set_xlim(ax_main.get_xlim())
    ax_y.set_ylim(ax_main.get_ylim())

    plt.tight_layout()

    return fig, ax_main, ax_x, ax_y

#80% of central beam
def mask80(x):
    cdf = np.cumsum(x, dtype=float)
    cdf /= cdf[-1] # Normalize CDF to 1
    mask = (cdf >= 0.1) & (cdf <= 0.9) # Mask for central 80%
    return x[mask]



def moving_average(x):
    n = int(len(x)/10)
    """Simple moving average with window size n."""
    return np.convolve(x, np.ones(n)/n, mode='same')

def sum_2gaussians(x, A, x0, sigma_x):
    return A * (np.exp(-( (x-x0)**2 /(2*sigma_x**2) )) + np.exp(-( (x+x0)**2 /(2*sigma_x**2) )) )
            

def flatness(x):
    
    x = moving_average(x) #smoothing
    x = mask80(x)
    return (max(x)-min(x))/ (max(x)+min(x))

def plot_phsp(T, M, n_bins=50, fov=200, title=None, slice_width=1): #cleaner version same format as dose map plotting with supergaussian and 2gaussian fits

    fig, ax_main = plt.subplots(figsize=(10, 6))
    x,y = M[:, 0], M[:, 2]

    # Main image
    im = ax_main.scatter(x, y, s=1, alpha=0.05)
    ax_main.set_xlim(-fov, fov)
    ax_main.set_ylim(-fov, fov)
    ax_main.set_aspect('equal', adjustable='box')

    ax_main.set_xlabel("X (mm)")
    ax_main.set_ylabel("Y (mm)")


    # Layout
    divider = make_axes_locatable(ax_main)
    ax_histx = divider.append_axes("top", size="25%", pad=0.2, sharex=ax_main)
    ax_histy = divider.append_axes("right", size="25%", pad=0.2, sharey=ax_main)

    # horizontal slice: |y| < slice_width
    rect_h = Rectangle(
        (-fov, -slice_width),
        2 * fov,
        2 * slice_width,
        edgecolor="white",
        facecolor="none",
        linewidth=1.5,
        linestyle="--"
    )

    # vertical slice: |x| < slice_width
    rect_v = Rectangle(
        (-slice_width, -fov),
        2 * slice_width,
        2 * fov,
        edgecolor="white",
        facecolor="none",
        linewidth=1.5,
        linestyle="--"
    )

    ax_main.add_patch(rect_h)
    ax_main.add_patch(rect_v)

    phsp_xslice = M[(M[:, 2] < slice_width)]
    phsp_xslice = phsp_xslice[(phsp_xslice[:, 2] > -slice_width)]

    phsp_yslice = M[(M[:, 0] < slice_width)]
    phsp_yslice = phsp_yslice[(phsp_yslice[:, 0] > -slice_width)]

    hist_x, bin_edges_x = np.histogram(phsp_xslice[:, 0], bins=n_bins, range=[-fov, fov])
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2

    hist_y, bin_edges_y = np.histogram(phsp_yslice[:, 2], bins=n_bins, range=[-fov, fov])
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

    # Fits
    p0 = [np.max(hist_x), np.mean(phsp_xslice[:, 0]), np.std(phsp_xslice[:, 0]), 4]
    params_x, _ = curve_fit(supergaussian1D, bin_centers_x, hist_x, p0=p0)
    params_y, _ = curve_fit(supergaussian1D, bin_centers_y, hist_y, p0=p0)

    p00 = [np.max(hist_x), 10, np.std(phsp_xslice[:, 0])]
    params_xx, _ = curve_fit(sum_2gaussians, bin_centers_x, hist_x, p0=p00)
    params_yy, _ = curve_fit(sum_2gaussians, bin_centers_y, hist_y, p0=p00)

    xy_fit_curve = np.linspace(-fov, fov, 500)

    r90_x = r90(params_x[2], params_x[3])
    r90_y = r90(params_y[2], params_y[3])

    # X histogram
    ax_histx.hist(phsp_xslice[:, 0], bins=n_bins, range=[-fov, fov],
                    color="b", alpha=0.6)
    ax_histx.plot(xy_fit_curve, supergaussian1D(xy_fit_curve, *params_x),
                    'r-', label=f"SuperGaussian Fit (P={params_x[3]:.2f}, r90={r90_x:.2f})")
    ax_histx.plot(xy_fit_curve, sum_2gaussians(xy_fit_curve, *params_xx),
                    'g-', label=f"2-Gaussian Fit (x0/sigma={params_xx[1]/params_xx[2]:.2f})")
    ax_histx.legend(loc="lower left")
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.set_ylabel("Intensity")


    # Y histogram
    ax_histy.hist(phsp_yslice[:, 2], bins=n_bins, range=[-fov, fov],
                    color="b", alpha=0.6, orientation="horizontal")
    ax_histy.plot(supergaussian1D(xy_fit_curve, *params_y), xy_fit_curve,
                    'r-', label=f"SuperGaussian Fit (P={params_y[3]:.2f}, r90={r90_y:.2f})")
    ax_histy.plot(sum_2gaussians(xy_fit_curve, *params_yy), xy_fit_curve,
                    'g-', label=f"2-Gaussian Fit (x0/sigma={params_yy[1]/params_yy[2]:.2f})")
    ax_histy.legend(loc="lower right")
    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histy.set_xlabel("Intensity")

    print(slice_width)
    if title:
        plt.suptitle(title)

    plt.show()


    
def main():
    n_particles = 10000
    dose_depth = 100
    output_filename = "ini_trial"
    
    fitDoseMap(n_particles, dose_depth, output_filename)
    
    




        