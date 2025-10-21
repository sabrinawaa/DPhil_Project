import numpy as np
from topas2numpy import BinnedResult
import matplotlib.pyplot as plt
# simParticles is the number of particles that the simulation was run with
# acChargenC is the simulated total charge to create an appropriate scaling factor (effectively arbitrary, 10nC is reasonable for now)

def getDosemap(filePath,simParticles,dose_depth,outputFileName,acChargenC=10, plot=False):
    
    doseMap = BinnedResult(filePath)#  2D array where each element corresponds to the dose deposited in a specific bin. The dose information is stored in the numerical values of the array elements.
    rawDosemap=np.squeeze(doseMap.data['Sum'])
    x,y = doseMap.dimensions[0].get_bin_centers(), doseMap.dimensions[0].get_bin_centers() #converts bins to cm
    x,y = x * 10, y * 10  # Convert from cm to mm
    eCharge=1.60217663e-19
    trueChargenC=(simParticles * eCharge)*1e9 #charge in simulation in nC
    scalingFactor=acChargenC/trueChargenC #number of particles simulated vs number of particles in actual beam
    scaledDosemap=np.rot90(rawDosemap * scalingFactor)

    if plot:
        # plt.figure(figsize=(8, 6))
        # plt.imshow(scaledDosemap, extent=[0, scaledDosemap.shape[1], 0, scaledDosemap.shape[0]],
        #            origin='lower', cmap='viridis')
        # plt.colorbar(label='Dose (Gy)')
        # plt.title('Dose Distribution at Depth' + str(dose_depth) + 'mm')
        # plt.xlabel('X Bins')
        # plt.ylabel('Y Bins')
        
        # slice of dose map across axes at origin

        dose_y = scaledDosemap[scaledDosemap.shape[0] // 2, :]  # Middle row (horizontal slice)
        dose_x = scaledDosemap[:, scaledDosemap.shape[1] // 2]  # Middle column (vertical slice)

        # Create the figure with subplots
        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)  # Create a grid for subplots
        # Plot the 2D dose map
        ax_main = fig.add_subplot(grid[1:, :-1])
        im = ax_main.imshow(scaledDosemap, cmap='viridis', origin='lower')
        ax_main.set_title("Dose Map at Depth =" + str(dose_depth) + "mm")
        ax_main.set_xlabel("X bins")
        ax_main.set_ylabel("Y bins")
        cbar = plt.colorbar(im, ax=ax_main, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Dose (Gy)")

        # Plot the histogram along the X-axis
        ax_x = fig.add_subplot(grid[0, :-1], sharex=ax_main)
        ax_x.bar(range(dose_x.shape[0]), dose_x, color='blue', alpha=0.7)
        ax_x.set_ylabel("Sum Dose (Gy)")
        ax_x.set_title("X Histogram")
        ax_x.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-ticks
        ax_x.set_xlim(ax_main.get_xlim()) 

        # Plot the histogram along the Y-axis
        ax_y = fig.add_subplot(grid[1:, -1], sharey=ax_main)
        ax_y.barh(range(dose_y.shape[0]), dose_y, color='blue', alpha=0.7)
        ax_y.set_xlabel("Sum Dose (Gy)")
        ax_y.set_title("Y Histogram")
        ax_y.tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide y-ticks
        ax_y.set_ylim(ax_main.get_ylim())  # Align the Y-axis with the 2D map
        plt.tight_layout()
        plt.savefig("Output_figs/"+ outputFileName + "dosedepth=" + str(dose_depth) +"dose.png")
        # Plot X and Y 1D histograms

        
        # fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        # # X Projection
        # ax1[0].bar(range(len(dose_x)), dose_x, width=1.0)
        # ax1[0].set_title('X Projection of Dose Distribution')   
        # ax1[0].set_xlabel('X Distribution')   
        # ax1[0].set_ylabel('Summed Dose (Gy)')   

        # # Y Projection
        # ax1[1].bar(range(len(dose_y)), dose_y, width=1.0)
        # ax1[1].set_title('Y Projection of Dose Distribution')   
        # ax1[1].set_xlabel('Y Distribution')   
        # ax1[1].set_ylabel('Summed Dose (Gy)')   

        # plt.tight_layout()  # Adjust layout for better spacing
        # fig1.savefig("Output_figs/" + outputFileName + "dosedepth=" + str(dose_depth) + "_xy_dose.png")
    return x,y, scaledDosemap

#still need to convert bins to mm 