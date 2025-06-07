import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil 
import constants 
from utils import conserved_to_primitive

def save_results_to_csv(output_dir_name, params, cell_centers, T_final, rho, p, u, e_internal):
    """
    Saves simulation parameters and final results to CSV files in a specified directory.
    This version will overwrite the directory if it exists.

    Args:
        output_dir_name (str): The name of the output directory (e.g., "output").
        params (dict): Dictionary of simulation parameters for saving in params.txt.
        cell_centers (np.ndarray): Array of cell center coordinates.
        T_final (float): The final time of the simulation for these results.
        rho (np.ndarray): Density array.
        p (np.ndarray): Pressure array.
        u (np.ndarray): Velocity array.
        e_internal (np.ndarray): Specific internal energy array.
    """
    # If the output directory exists, remove it to ensure a clean overwrite
    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
        print(f"Removed existing output directory: {output_dir_name}")
    
    try:
        os.makedirs(output_dir_name, exist_ok=True) # exist_ok=True is fine, but rmtree handles overwrite
        print(f"Saving results to: {output_dir_name}")

        # Save parameters to a text file
        with open(os.path.join(output_dir_name, "params.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            # T_final is already in params as T_final_achieved_output or T_final_target

        # Helper to save data
        def save_array_csv(filename, arr1, arr2, header_str):
            filepath = os.path.join(output_dir_name, filename)
            data_to_save = np.column_stack((arr1, arr2))
            np.savetxt(filepath, data_to_save, delimiter=',', header=header_str, comments='')

        # save_array_csv("cell_centers.csv", np.arange(len(cell_centers)), cell_centers, "index,x_center")
        save_array_csv("density.csv", cell_centers, rho, "x_center,density")
        save_array_csv("pressure.csv", cell_centers, p, "x_center,pressure")
        save_array_csv("velocity.csv", cell_centers, u, "x_center,velocity")
        save_array_csv("internal_energy.csv", cell_centers, e_internal, "x_center,internal_energy")

        print("Results saved successfully.")
        
    except Exception as e:
        print(f"Error saving results to {output_dir_name}: {e}")

def load_analytical_data_1d(analytical_data_path, filename, x_scale_factor=1.0, skiprows=1, delimiter=','):
    """
    Loads analytical data from a CSV file using NumPy.
    Assumes x is the first column and value is the second.
    """
    full_path = os.path.join(analytical_data_path, filename)
    try:
        data_columns = np.loadtxt(
            full_path, 
            delimiter=delimiter, 
            skiprows=skiprows, 
            usecols=(0, 1), # Assume x in col 0, value in col 1
            unpack=True
        )
        x_data = data_columns[0]
        val_data = data_columns[1]
        x_data_scaled = x_data * x_scale_factor
        print(f"Successfully loaded analytical data from: {full_path}")
        return x_data_scaled, val_data
    except Exception as e:
        print(f"Error reading analytical file {full_path}: {e}")
        return None, None

def plot_simulation_results_1d(numerical_data, run_params,
                               plot_analytical_flag=False, 
                               analytical_data_path="../analytical_data", # Relative to src/ where main.py is
                               analytical_x_scale=0.01): 
    """
    Plots the final state of the 1D simulation and optionally overlays analytical data.

    Args:
        numerical_data (tuple): (cell_centers, rho, p, u, e_internal)
        run_params (dict): Dictionary of parameters for the numerical run (for titles).
        plot_analytical_flag (bool): If True, attempts to load and plot analytical data.
        analytical_data_path (str): Path to the directory containing analytical CSVs.
        analytical_x_scale (float): Scale factor for analytical x-data.
    """
    cell_centers, rho_num, p_num, u_num, e_num = numerical_data
    T_final_achieved = run_params.get("T_final_achieved_output", run_params.get("t_final_target", "N/A"))
    if isinstance(T_final_achieved, str): # If it's 'N/A' or not converted
        try: T_final_achieved = float(T_final_achieved)
        except: T_final_achieved = 0.0 # Fallback
    
    # --- Load Analytical Data if requested ---
    analytical_plots = {}
    if plot_analytical_flag:
        print(f"\nAttempting to load analytical data for t=0.2s from: {analytical_data_path}")
        # Assuming analytical files are named consistently and are for t=0.2s
        x_rho_an, rho_an = load_analytical_data_1d(analytical_data_path, "analytical_density.csv", x_scale_factor=analytical_x_scale)
        if x_rho_an is not None: analytical_plots['rho'] = (x_rho_an, rho_an)
        
        x_p_an, p_an = load_analytical_data_1d(analytical_data_path, "analytical_pressure.csv", x_scale_factor=analytical_x_scale)
        if x_p_an is not None: analytical_plots['p'] = (x_p_an, p_an)

        x_u_an, u_an = load_analytical_data_1d(analytical_data_path, "analytical_U.csv", x_scale_factor=analytical_x_scale)
        if x_u_an is not None: analytical_plots['u'] = (x_u_an, u_an)
        
        # You might need an analytical internal energy file or calculate it if p_an, rho_an are available
        # For now, let's assume you might have it or skip it
        # x_e_an, e_an = load_analytical_data_1d(analytical_data_path, "analytical_internal_energy.csv", x_scale_factor=analytical_x_scale)
        # if x_e_an is not None: analytical_plots['e'] = (x_e_an, e_an)


    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    ((ax_rho, ax_p), (ax_u, ax_e)) = axs

    plot_title_main = (f"1D Sod Tube: N={run_params.get('N_cells','?')}, CFL={run_params.get('CFL','?')}\n"
                       f"Spatial: {run_params.get('scheme','?').upper()}, Time: {run_params.get('integrator','?').upper()}, "
                       f"Riemann: {run_params.get('solver','?').upper()}")
    if run_params.get('solver') == constants.SOLVER_HLLC:
        plot_title_main += f" (HLLC WS: {run_params.get('hllc_wave_speeds','?').upper()})"
    fig.suptitle(plot_title_main, fontsize=14)

    # Density
    ax_rho.set_title('Density') # Set titles first
    ax_rho.set_ylabel('Density ($\\rho$)')
    if 'rho' in analytical_plots:
        ax_rho.plot(analytical_plots['rho'][0], analytical_plots['rho'][1], 
                    'k.', markersize=3, label='Analytical (t=0.2s)') # Changed to 'k.' for black dots
    ax_rho.plot(cell_centers, rho_num, 'b-o', markersize=2, alpha=0.7, label=f'Numerical (t={T_final_achieved:.3f})')
    ax_rho.grid(True); ax_rho.legend()

    # Pressure
    ax_p.set_title('Pressure')
    ax_p.set_ylabel('Pressure ($p$)')
    if 'p' in analytical_plots:
        ax_p.plot(analytical_plots['p'][0], analytical_plots['p'][1], 
                  'k.', markersize=3, label='Analytical (t=0.2s)') # Changed to 'k.'
    ax_p.plot(cell_centers, p_num, 'r-o', markersize=2, alpha=0.7, label=f'Numerical (t={T_final_achieved:.3f})')
    ax_p.grid(True); ax_p.legend()

    # Velocity
    ax_u.set_title('Velocity')
    ax_u.set_xlabel('x'); ax_u.set_ylabel('Velocity ($u$)')
    if 'u' in analytical_plots:
        ax_u.plot(analytical_plots['u'][0], analytical_plots['u'][1], 
                  'k.', markersize=3, label='Analytical (t=0.2s)') # Changed to 'k.'
    ax_u.plot(cell_centers, u_num, 'g-o', markersize=2, alpha=0.7, label=f'Numerical (t={T_final_achieved:.3f})')
    ax_u.grid(True); ax_u.legend()

    # Specific Internal Energy
    ax_e.set_title('Specific Internal Energy')
    ax_e.set_xlabel('x'); ax_e.set_ylabel('Internal Energy ($e$)')
    # Add analytical plot for 'e' if loaded and you want dots
    # if 'e' in analytical_plots:
    #     ax_e.plot(analytical_plots['e'][0], analytical_plots['e'][1], 
    #               'k.', markersize=3, label='Analytical (t=0.2s)')
    ax_e.plot(cell_centers, e_num, 'm-o', markersize=2, alpha=0.7, label=f'Numerical (t={T_final_achieved:.3f})')
    ax_e.grid(True); ax_e.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect for suptitle
    plt.show()



def plot_and_save_single_frame(output_dir, frame_number, 
                               cell_centers, T_current, 
                               rho_num, p_num, u_num, e_num, 
                               run_params):
    """
    Creates and saves a single plot frame for an animation.
    """
    # We need gamma for the y-axis limits calculation
    gamma_anim = run_params.get('gamma_eos', 1.4)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    ((ax_rho, ax_p), (ax_u, ax_e)) = axs
    
    # --- Create a dynamic title for each frame ---
    plot_title_main = (f"1D Sod Tube: N={run_params.get('N_cells','?')}, CFL={run_params.get('CFL','?')}\n"
                       f"Spatial: {run_params.get('scheme','?').upper()}, Time: {run_params.get('integrator','?').upper()}, "
                       f"Riemann: {run_params.get('solver','?').upper()}")
    fig.suptitle(plot_title_main, fontsize=14)

    # --- Plot the data for the current time step ---
    ax_rho.plot(cell_centers, rho_num, 'b-o', markersize=3)
    ax_p.plot(cell_centers, p_num, 'r-o', markersize=3)
    ax_u.plot(cell_centers, u_num, 'g-o', markersize=3)
    ax_e.plot(cell_centers, e_num, 'm-o', markersize=3)

    # --- Set plot titles, labels, and fixed limits for consistency ---
    # Setting fixed limits is important so the axes don't jump between frames
    # These limits should be based on the overall simulation, not just one frame.
    # For simplicity here, we'll set some reasonable defaults for Sod's problem.
    ax_rho.set_title('Density'); ax_rho.set_ylabel('Density ($\\rho$)'); ax_rho.grid(True); ax_rho.set_ylim(0, 1.1)
    ax_p.set_title('Pressure'); ax_p.set_ylabel('Pressure ($p$)'); ax_p.grid(True); ax_p.set_ylim(0, 1.1)
    ax_u.set_title('Velocity'); ax_u.set_xlabel('x'); ax_u.set_ylabel('Velocity ($u$)'); ax_u.grid(True); ax_u.set_ylim(-0.1, 1.1)
    ax_e.set_title('Specific Internal Energy'); ax_e.set_xlabel('x'); ax_e.set_ylabel('Internal Energy ($e$)'); ax_e.grid(True); ax_e.set_ylim(1.8, 3.0)
    
    # Add a time text annotation
    time_text = ax_rho.text(0.05, 0.95, f'Time = {T_current:.4f} s', 
                            transform=ax_rho.transAxes, ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    # --- Save the figure ---
    # Use zero-padded frame number for correct ordering (e.g., frame_0001.png)
    frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
    plt.savefig(frame_filename, dpi=100) # dpi controls resolution
    plt.close(fig) # IMPORTANT: Close the figure to free up memory
