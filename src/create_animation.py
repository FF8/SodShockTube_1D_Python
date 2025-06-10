
# create_animation.py
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import subprocess

def load_history_data(directory_path):
    """
    Loads time history data from the CSV files saved by the simulation.
    """
    print(f"Loading data from: {directory_path}")
    data = {}

    try:
        # Load params.txt for plot titles and metadata
        params = {}
        with open(os.path.join(directory_path, "params.txt"), "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    params[key.strip()] = value.strip()
        data['params'] = params

        # Helper function to load a single CSV history file
        def load_csv(filename):
            full_path = os.path.join(directory_path, filename)
            # Read header to get time steps
            with open(full_path, 'r') as f:
                header = f.readline().strip().lstrip('#').strip()
                time_values = np.array(header.split(',')[1:], dtype=float)

            # Load numerical data, skipping header
            numerical_data = np.loadtxt(full_path, delimiter=',', skiprows=1)
            x_coords = numerical_data[:, 0]
            value_history = numerical_data[:, 1:]
            return x_coords, time_values, value_history

        # Load all history files
        data['cell_centers'], data['times'], data['rho_hist'] = load_csv("density_history.csv")
        _, _, data['p_hist'] = load_csv("pressure_history.csv")
        _, _, data['u_hist'] = load_csv("velocity_history.csv")
        _, _, data['e_hist'] = load_csv("internal_energy_history.csv")

        print("Data loaded successfully.")
        return data

    except Exception as e:
        print(f"Error loading history data from {directory_path}: {e}")
        return None

def create_movie_from_frames(frames_dir, output_filename, fps=20):
    """
    Uses ffmpeg to compile PNG frames into a movie.
    """
    print(f"\nCompiling frames from '{frames_dir}' into '{output_filename}'...")

    # Check if ffmpeg is available
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not found. Please install ffmpeg to create movies.")
        return

    # ffmpeg command
    command = [
        'ffmpeg',
        '-framerate', str(fps),          # Frames per second
        '-i', f'{frames_dir}/frame_%04d.png', # Input files pattern
        '-c:v', 'libx264',               # Video codec
        '-pix_fmt', 'yuv420p',           # Pixel format for compatibility
        '-y',                            # Overwrite output file if it exists
        output_filename
    ]

    try:
        # Use subprocess.run for better error handling
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Movie created successfully!")
    except subprocess.CalledProcessError as e:
        print("--- FFMPEG ERROR ---")
        print(e.stderr)
        print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation from saved 1D solver data.")
    parser.add_argument("result_dir", type=str, help="Path to the directory containing the simulation history CSV files.")
    parser.add_argument("--output", type=str, default="simulation_movie.mp4", help="Name of the output movie file (e.g., movie.mp4 or movie.gif).")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the output animation.")
    args = parser.parse_args()

    # --- 1. Load the Data ---
    history_data = load_history_data(args.result_dir)

    if not history_data:
        print("Exiting due to data loading failure.")
        exit()

    # --- 2. Create Individual PNG Frames ---
    temp_frames_dir = "temp_animation_frames"
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir) # Clear old frames
    os.makedirs(temp_frames_dir)

    # Get data from the dictionary
    params = history_data['params']
    gamma_val = float(params.get('gamma_eos', 1.4))
    cell_centers = history_data['cell_centers']
    times = history_data['times']
    rho_hist = history_data['rho_hist']
    p_hist = history_data['p_hist']
    u_hist = history_data['u_hist']
    e_hist = history_data['e_hist']

    num_frames = len(times)
    print(f"\nGenerating {num_frames} frames in '{temp_frames_dir}'...")

    # Set fixed axis limits for a smooth animation
    y_limits = {
        'rho': (np.min(rho_hist) * 0.95, np.max(rho_hist) * 1.05),
        'p': (np.min(p_hist) * 0.95, np.max(p_hist) * 1.05),
        'u': (np.min(u_hist) - 0.1, np.max(u_hist) + 0.1),
        'e': (np.min(e_hist) * 0.95, np.max(e_hist) * 1.05)
    }

    for i in range(num_frames):
        # Create a plot for the current frame
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        ((ax_rho, ax_p), (ax_u, ax_e)) = axs

        plot_title = (f"N={params.get('N_cells','?')}, CFL={params.get('CFL','?')}, "
                      f"{params.get('scheme','?').upper()}, {params.get('integrator','?').upper()}, "
                      f"{params.get('solver','?').upper()}")
        fig.suptitle(plot_title, fontsize=14)

        # Plot data for this frame
        ax_rho.plot(cell_centers, rho_hist[:, i], 'b-o', ms=3)
        ax_p.plot(cell_centers, p_hist[:, i], 'r-o', ms=3)
        ax_u.plot(cell_centers, u_hist[:, i], 'g-o', ms=3)
        ax_e.plot(cell_centers, e_hist[:, i], 'm-o', ms=3)

        # Apply fixed limits and labels
        ax_rho.set_title('Density'); ax_rho.set_ylabel('Density ($\\rho$)'); ax_rho.grid(True); ax_rho.set_ylim(y_limits['rho'])
        ax_p.set_title('Pressure'); ax_p.set_ylabel('Pressure ($p$)'); ax_p.grid(True); ax_p.set_ylim(y_limits['p'])
        ax_u.set_title('Velocity'); ax_u.set_xlabel('x'); ax_u.set_ylabel('Velocity ($u$)'); ax_u.grid(True); ax_u.set_ylim(y_limits['u'])
        ax_e.set_title('Specific Internal Energy'); ax_e.set_xlabel('x'); ax_e.set_ylabel('Internal Energy ($e$)'); ax_e.grid(True); ax_e.set_ylim(y_limits['e'])

        # Add time annotation
        ax_rho.text(0.05, 0.95, f'Time = {times[i]:.4f} s', transform=ax_rho.transAxes,
                    ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Save the frame
        frame_filename = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_filename, dpi=100)
        plt.close(fig) # IMPORTANT: Free up memory

        if (i+1) % 10 == 0 or i == num_frames - 1:
            print(f"  ... saved frame {i+1}/{num_frames}")

    # --- 3. Compile Frames into a Movie ---
    create_movie_from_frames(temp_frames_dir, args.output, args.fps)

    # --- 4. Clean up ---
    print(f"Cleaning up temporary frames directory: '{temp_frames_dir}'")
    shutil.rmtree(temp_frames_dir)

    print("\nDone.")
