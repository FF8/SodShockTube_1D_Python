import json
import argparse
import numpy as np
import os

import constants
from solver_engine import run_simulation
from utils import conserved_to_primitive
from postprocessing_1d import save_results_to_csv, plot_simulation_results_1d

gamma = 1.4

def load_config(config_filepath):
    """Loads simulation configuration from a JSON file."""
    try:
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded configuration from: {config_filepath}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_filepath}")
        # exit() # Or raise error
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {config_filepath}. Malformed JSON? Details: {e}")
        # exit() # Or raise error
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        return None

def print_configuration(config):
    """Prints the loaded configuration in a readable format."""
    if not config:
        print("No configuration to print.")
        return

    print("\n--- Simulation Configuration ---")
    
    sim_setup = config.get("simulation_setup", {})
    print("\n[Simulation Setup]")
    print(f"  Number of Cells (N_cells): {sim_setup.get('N_cells', 'N/A')}")
    print(f"  Domain Length:           {sim_setup.get('domain_length', 'N/A')}")
    print(f"  Final Time (t_final):      {sim_setup.get('t_final', 'N/A')}")
    print(f"  CFL Number:              {sim_setup.get('CFL', 'N/A')}")
    print(f"  Problem Type:            {sim_setup.get('problem_type', 'N/A')}")

    num_schemes = config.get("numerical_schemes", {})
    print("\n[Numerical Schemes]")
    print(f"  Spatial Scheme:          {num_schemes.get('spatial_scheme', 'N/A')}")
    print(f"  Time Integrator:         {num_schemes.get('time_integrator', 'N/A')}")
    print(f"  Riemann Solver:          {num_schemes.get('riemann_solver', 'N/A')}")
    if num_schemes.get('riemann_solver') == 'hllc':
        print(f"  HLLC Wave Speeds:      {num_schemes.get('hllc_wave_speeds', 'N/A')}")

    bc_params = config.get("boundary_conditions", {})
    print("\n[Boundary Conditions]")
    print(f"  Left BC Type:            {bc_params.get('left_bc_type', 'N/A')}")
    print(f"  Right BC Type:           {bc_params.get('right_bc_type', 'N/A')}")

    eos_params = config.get("eos_parameters", {})
    print("\n[Equation of State]")
    print(f"  EOS Type:                {eos_params.get('type', 'N/A')}")
    print(f"  Gamma:                   {eos_params.get('gamma', 'N/A')}")
    
    output_params = config.get("output_control", {})
    print("\n[Output Control]")
    print(f"  Base Output Directory:   {output_params.get('base_output_directory', 'N/A')}")
    print(f"  Results Save Frequency:  {output_params.get('results_save_frequency', 'N/A')}")
    print("------------------------------")


def main():
    
    global gamma

    parser = argparse.ArgumentParser(description="Run 1D Euler Solver with specified configuration.")
    parser.add_argument("--config", type=str, default="simulation_config.json",
                        help="Path to the JSON configuration file (default: simulation_config.json)")
    args = parser.parse_args()

    print(f"Attempting to load configuration from: {args.config}")
    config_data = load_config(args.config)

    if not config_data:
        print("Exiting due to configuration load error.")
        return
    print_configuration(config_data)

    # Extract parameters from config
    eos_params = config_data.get("eos_parameters", {})
    gamma = float(eos_params.get("gamma", 1.4)) # Updates the module-level gamma

    # Extract other parameters for run_simulation
    sim_setup = config_data.get("simulation_setup", {})
    N_cells_sim = int(sim_setup.get("N_cells", 200))
    domain_length_sim = float(sim_setup.get("domain_length", 1.0))
    t_final_sim = float(sim_setup.get("t_final", 0.2))
    C_cfl_sim = float(sim_setup.get("CFL", 0.5))
    problem_type_sim = str(sim_setup.get("problem_type", "sod_shock_tube"))

    num_schemes = config_data.get("numerical_schemes", {})
    current_scheme = str(num_schemes.get("spatial_scheme", constants.SCHEME_MUSCL))
    current_time_integrator = str(num_schemes.get("time_integrator", constants.INTEGRATOR_SSPRK2))
    current_riemann_solver = str(num_schemes.get("riemann_solver", constants.SOLVER_HLLC))
    current_hllc_wave_speeds = str(num_schemes.get("hllc_wave_speeds", constants.HLLC_WAVE_SPEED_ROE))

    bc_params = config_data.get("boundary_conditions", {})
    current_bc_left = str(bc_params.get("left_bc_type", constants.BC_TRANSMISSIVE))
    current_bc_right = str(bc_params.get("right_bc_type", constants.BC_REFLECTIVE))
    
    output_ctrl = config_data.get("output_control", {})
    do_save_results = output_ctrl.get("save_results", False)
    do_plot_results = output_ctrl.get("plot_results", False)
    plot_analytical = output_ctrl.get("plot_against_analytical_0.2s", False)
    # create_animation_flag = output_ctrl.get("create_animation", False)

    # Fixed output directory name for simplicity if overwriting
    fixed_output_dir_name = "output_1d_results" 


    run_params_for_saving = {
        "N_cells": N_cells_sim, "domain_length": domain_length_sim,
        "t_final_target": t_final_sim, "CFL": C_cfl_sim, "problem": problem_type_sim,
        "scheme": current_scheme, "integrator": current_time_integrator,
        "solver": current_riemann_solver,
        "hllc_wave_speeds": current_hllc_wave_speeds if current_riemann_solver == constants.SOLVER_HLLC else "N/A",
        "bc_left": current_bc_left, "bc_right": current_bc_right,
        "gamma_eos": gamma # Save the gamma used
    }
    
    print(f"\n--- STARTING SIMULATION (using gamma = {gamma}) ---")
    
    output_ctrl = config_data.get("output_control", {})
    create_animation_flag = output_ctrl.get("create_animation", False)
    
    # Define animation parameters
    animation_parameters = {
        "dir": "animation_frames", # Folder to save frames
        "freq": 10 # Save a frame every 10 iterations
    }
    
    cell_centers_res, T_res, U_res_list = run_simulation(
        N_cells=N_cells_sim, domain_length=domain_length_sim, t_final=t_final_sim, C_cfl=C_cfl_sim,
        problem_type=problem_type_sim, scheme=current_scheme, time_integrator=current_time_integrator,
        riemann_solver=current_riemann_solver, bc_left=current_bc_left, bc_right=current_bc_right,
        hllc_wave_speed_config=current_hllc_wave_speeds,
        gamma_eos=gamma,
        create_animation_flag=create_animation_flag,
        animation_params=animation_parameters
    )

    if U_res_list:
        U_final_numerical = U_res_list[-1]
        T_final_achieved = T_res[-1]
        
        rho_numerical, u_numerical, p_numerical = conserved_to_primitive(U_final_numerical, gamma)
        e_internal_numerical = p_numerical / ((gamma - 1.0) * np.maximum(rho_numerical, constants.EPSILON) + constants.EPSILON)
        
        run_params_for_saving["T_final_achieved_output"] = T_final_achieved

        if do_save_results:
            save_results_to_csv(
                fixed_output_dir_name, # Use fixed name, will overwrite
                run_params_for_saving, 
                cell_centers_res,
                T_final_achieved, 
                rho_numerical,
                p_numerical,
                u_numerical,
                e_internal_numerical
            )
        
        if do_plot_results:
            numerical_data_tuple = (cell_centers_res, rho_numerical, p_numerical, u_numerical, e_internal_numerical)
            # Define path to analytical data relative to where main_1D_solver.py is run
            # If main_1D_solver.py is in src/, and analytical_data is ../analytical_data
            script_dir = os.path.dirname(__file__)
            analytical_data_dir = os.path.abspath(os.path.join(script_dir, "../analytical_data"))

            plot_simulation_results_1d(
                numerical_data_tuple,
                run_params_for_saving,
                plot_analytical_flag=plot_analytical,
                analytical_data_path=analytical_data_dir
            )

    else:
        print("Simulation did not produce results.")

if __name__ == "__main__":
    main()
