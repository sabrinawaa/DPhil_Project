import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from topasToDose import getDosemap

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

def fitDoseMap (n_particles, dose_depth,output_filename, zoom_factor=1, plot=True):
    x,y, doseMap = getDosemap("DoseAtTank"+str(dose_depth)+ "_"+ output_filename+".csv",n_particles, dose_depth, output_filename, plot = False)
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

def plot_phsp(T,M, n_bins=50,fov=200):
     
    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        
        ax_histx.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=True)

        # the scatter plot:
        ax.scatter(x, y,s=1,alpha=0.5)
        ax.set_xlim(-fov,fov)
        ax.set_ylim(-fov,fov)

        ax.set_xlabel('X (mm)')  # Set x-axis label for scatter plot
        ax.set_ylabel('Y (mm)')

        slice_width = 1
        phsp_xslice = M[(M[:,2] < slice_width)]
        phsp_xslice = phsp_xslice[(phsp_xslice[:,2] > -slice_width)]
        
        phsp_yslice = M[(M[:,0] < slice_width)]
        phsp_yslice = phsp_yslice[(phsp_yslice[:,0] > -slice_width)]

        
        hist_x, bin_edges_x = np.histogram(phsp_xslice[:,0], bins=n_bins, range=[-fov, fov])
        bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2


        hist_y, bin_edges_y = np.histogram(phsp_yslice[:,2], bins=n_bins, range=[-fov, fov])
        bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

        #fits
        p0=[np.max(hist_x),  np.mean(phsp_xslice[:,0]), np.std(phsp_xslice[:,0]), 4]
        params_x, _ = curve_fit(supergaussian1D, bin_centers_x, hist_x, p0=p0)

        p00 = [np.max(hist_x),  10, np.std(phsp_xslice[:,0])]
        params_xx, _ = curve_fit(sum_2gaussians, bin_centers_x, hist_x, p0=p00)

        params_y, _ = curve_fit(supergaussian1D, bin_centers_y, hist_y, p0=p0)
        params_yy, _ = curve_fit(sum_2gaussians, bin_centers_y, hist_y, p0=p00)
        xy_fit_curve = np.linspace(-fov, fov, 500)
       
        sig_x,sig_y, P_x, P_y = params_x[2], params_y[2], params_x[3], params_y[3]
        r90_x, r90_y = r90(sig_x, P_x), r90(sig_y, P_y)
    
        # Plot SuperGaussian fits

        ax_histx.hist(phsp_xslice[:,0], bins=n_bins, range=[-fov, fov], color="b",alpha=0.6,label= ' X-Intensity')
        ax_histx.plot(xy_fit_curve, supergaussian1D(xy_fit_curve, *params_x), 'r-', label=f"SuperGaussian Fit (P={params_x[3]:.2f},r90={r90_x:.2f})")
        ax_histx.plot(xy_fit_curve, sum_2gaussians(xy_fit_curve, *params_xx), 'g-', label=f"2-Gaussian Fit (x0/sigma={params_xx[1]/params_xx[2]:.2f})")
        ax_histx.plot(bin_centers_x, moving_average(hist_x), 'k-', label=f'Smoothed (F={flatness(hist_x):.3f})')

        ax_histx.legend()


        

        ax_histy.hist(phsp_yslice[:,2], bins=n_bins, range=[-fov, fov], color="b",alpha=0.6,label= ' Y-Intensity',orientation="horizontal")
        ax_histy.plot(supergaussian1D(xy_fit_curve, *params_y), xy_fit_curve,  'r-', label=f"SuperGaussian Fit (P={params_y[3]:.2f},r90={r90_y:.2f})")
        ax_histy.plot( sum_2gaussians(xy_fit_curve, *params_yy), xy_fit_curve, 'g-', label=f"2-Gaussian Fit (x0/sigma={params_yy[1]/params_yy[2]:.2f})")
        ax_histy.plot(moving_average(hist_y),bin_centers_y, 'k-', label=f'Smoothed (F={flatness(hist_y):.3f})')
        ax_histy.legend()

    fig, axs = plt.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                figsize=(10, 8),
                                width_ratios=(4, 1), height_ratios=(1, 4),
                                layout='constrained')
    scatter_hist(M[:,0], M[:,2], axs['scatter'], axs['histx'], axs['histy'])
    plt.show()


    
def main():
    n_particles = 10000
    dose_depth = 100
    output_filename = "ini_trial"
    
    fitDoseMap(n_particles, dose_depth, output_filename)
    
    




        