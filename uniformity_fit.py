import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    
    
    p0=[0.2, np.max(x)//2, np.max(y)//2, 60, 60, 6]
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
    




    
def main():
    n_particles = 10000
    dose_depth = 100
    output_filename = "ini_trial"
    
    fitDoseMap(n_particles, dose_depth, output_filename)
    
    




        