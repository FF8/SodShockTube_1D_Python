import numpy as np
import constants 
import os
import shutil
from mesh_setup import setup_mesh
from initial_conditions import initialize_state
from utils import conserved_to_primitive, primitive_to_conserved
from riemann_solvers import hll_solver, hllc_solver
from schemes_spatial import calculate_muscl_slopes, minmod_limiter 
from boundary_conditions import apply_boundary_conditions

# gamma would be passed as an argument to functions here, not global to this file
# unless run_simulation sets a module-level one for its callees if they are also in this file

import numpy as np
from utils import primitive_to_conserved, conserved_to_primitive
import constants
# Assume the new apply_boundary_conditions and the rigorous calculate_muscl_slopes are available

def calculate_rhs_for_stage(U_stage, N_cells, N_ghost, physical_slice, dx, gamma_eos, scheme,
                            riemann_solver_choice, bc_left, bc_right,
                            hllc_wave_method):
    """
    Calculates the Right-Hand Side (RHS = dU/dt) for a stage U on a grid with ghost cells.
    
    Args:
        U_stage (np.ndarray): Full array (3, N_total) for the current stage.
        N_cells (int): Number of PHYSICAL cells.
        N_ghost (int): Number of ghost cells on each side.
        physical_slice (slice): Slice object for the physical domain.
        ... other parameters
    """
    
    # --- Step 1: Apply Boundary Conditions ---
    # This populates the ghost cell regions of U_stage for this specific stage.
    U_stage = apply_boundary_conditions(U_stage, N_ghost, bc_left, bc_right, gamma_eos)

    # --- Step 2: Perform Data Reconstruction (if not first-order) ---
    slopes_s = np.zeros_like(U_stage)
    if scheme == constants.SCHEME_MUSCL:
        # Use the rigorous slope calculator that works with ghost cells
        slopes_s = calculate_muscl_slopes(U_stage, N_ghost, limiter_func=minmod_limiter)
    
    # --- Step 3: Calculate Fluxes at All Physical Interfaces ---
    N_total = U_stage.shape[1]
    # We need N_total+1 interfaces for a grid with N_total cells
    F_numerical_fluxes = np.zeros((3, N_total + 1)) 

    # Loop through the interfaces of the physical cells.
    # The first physical interface is at index `N_ghost` (left of cell `N_ghost`).
    # The last physical interface is at index `N_ghost + N_cells` (right of the last physical cell).
    for j in range(N_ghost, N_ghost + N_cells + 1):
        # Cell indices to the left and right of interface j
        cell_idx_L = j - 1 
        cell_idx_R = j   

        # Reconstruct states at the interface j
        if scheme == constants.SCHEME_MUSCL:
            U_L_eff = U_stage[:, cell_idx_L] + 0.5 * slopes_s[:, cell_idx_L]
            U_R_eff = U_stage[:, cell_idx_R] - 0.5 * slopes_s[:, cell_idx_R]
        else: # First-order Godunov
            U_L_eff = U_stage[:, cell_idx_L]
            U_R_eff = U_stage[:, cell_idx_R]
        
        # Call chosen Riemann solver to get the flux at interface j
        if riemann_solver_choice == constants.SOLVER_HLL:
            F_numerical_fluxes[:, j] = hll_solver(U_L_eff, U_R_eff, gamma_eos)
        elif riemann_solver_choice == constants.SOLVER_HLLC:
            F_numerical_fluxes[:, j] = hllc_solver(U_L_eff, U_R_eff, gamma_eos, 
                                                   wave_speed_method=hllc_wave_method)
        else:
            raise ValueError(f"Unknown Riemann solver: {riemann_solver_choice}")

    # --- Step 4: Compute the Final RHS for each physical cell ---
    # The RHS is only computed for the physical cells, so the size is (3, N_total)
    RHS_1D = np.zeros_like(U_stage)
    
    # Loop ONLY over the physical cells
    for i in range(physical_slice.start, physical_slice.stop):
        # RHS_i = - (Flux_right_face - Flux_left_face) / dx
        # Right face of cell i is interface i+1
        # Left face of cell i is interface i
        RHS_1D[:, i] = -(F_numerical_fluxes[:, i+1] - F_numerical_fluxes[:, i]) / dx
        
    return RHS_1D




def run_simulation(N_cells, domain_length, t_final, C_cfl,
                   problem_type,
                   scheme, time_integrator, riemann_solver,
                   bc_left, bc_right,
                   hllc_wave_speed_config,
                   gamma_eos,  N_ghost=2,
                   save_interval=-1.0): 
    """
    Main loop for the 1D Euler solver.
    If save_interval > 0, stores snapshots at specified time intervals.
    """

    dx, cell_centers, cell_interfaces, p_slice = setup_mesh(N_cells, domain_length, N_ghost)
    
    U_current = initialize_state(N_cells, N_ghost, p_slice, cell_centers, cell_interfaces, gamma_eos,
                                    problem_type=problem_type) 
    
    
    t = 0.0
    iteration = 0
    results_t = [t]
    results_U = [U_current.copy()]

    # --- logic for time-based saving ---
    save_next_time = save_interval
    if save_interval <= 0: # If interval is invalid or not set, only save final result
        save_next_time = t_final + 1.0 # Ensure it won't trigger


    print(f"Starting 1D simulation: N_cells={N_cells}, Scheme={scheme}, Integrator={time_integrator}, Riemann={riemann_solver}", end="")
    if riemann_solver == constants.SOLVER_HLLC:
        print(f" (HLLC Wave Speeds: {hllc_wave_speed_config})", end="")
    print(f"\nBC Left: {bc_left}, BC Right: {bc_right}, t_final={t_final}, C_cfl={C_cfl}")

    while t < t_final:
        # --- Calculate Time Step (dt) based on CFL condition ---
        S_max_global = 0.0
        for i in range(p_slice.start, p_slice.stop):
            # Ensure primitive conversion uses the passed gamma_eos
            rho_i, u_i, p_i = conserved_to_primitive(U_current[:, i], gamma_eos)
            # Ensure sound speed calculation uses the passed gamma_eos and positive rho, p
            a_i = np.sqrt(gamma_eos * max(p_i, constants.EPSILON) / max(rho_i, constants.EPSILON)) 
            S_max_global = max(S_max_global, abs(u_i) + a_i)
        
        dt = C_cfl * dx / (S_max_global + constants.EPSILON) 
        
        # --- logic to clip dt to not miss a save point ---
        if save_interval > 0 and t < save_next_time and (t + dt) >= save_next_time:
            dt = save_next_time - t + constants.EPSILON * 1e-2 # Step exactly to the save time + tiny amount
        
        if t + dt > t_final:
            dt = t_final - t
        
        if dt <= constants.EPSILON * 10:
            print(f"Time step too small ({dt:.2e}), stopping.")
            break


        # Prepare arguments for calculate_rhs_1d_for_stage
        rhs_args = {
            "N_cells": N_cells, "N_ghost": N_ghost, "physical_slice": p_slice,
            "dx": dx, "gamma_eos": gamma_eos, "scheme": scheme,
            "riemann_solver_choice": riemann_solver,
            "bc_left": bc_left, "bc_right": bc_right,
            "hllc_wave_method": hllc_wave_speed_config
        }

        # --- Time Integration ---
        if time_integrator == constants.INTEGRATOR_EULER:
            RHS_n = calculate_rhs_for_stage(U_current, **rhs_args)
            U_next = U_current + dt * RHS_n
        elif time_integrator == constants.INTEGRATOR_SSPRK2:

            RHS_n = calculate_rhs_for_stage(U_current, **rhs_args)
            U_intermediate = U_current + dt * RHS_n
            
            RHS_intermediate = calculate_rhs_for_stage(U_intermediate, **rhs_args)
            U_next = 0.5 * U_current + 0.5 * (U_intermediate + dt * RHS_intermediate)
        else:
            raise ValueError(f"Unknown time integrator: {time_integrator}")      

        U_current = U_next
        t += dt
        iteration += 1

        # ---logic for saving snapshots ---
        if save_interval > 0 and t >= save_next_time:
            print(f"Iter: {iteration}, Time: {t:.4f}, dt: {dt:.3e} --- Saving snapshot")
            results_t.append(t)
            results_U.append(U_current.copy())
            save_next_time += save_interval # Set the next save time
        elif iteration % 100 == 0: # Keep periodic printing to console
            print(f"Iter: {iteration}, Time: {t:.4f}, dt: {dt:.3e}")
        
        if np.any(np.isnan(U_current)) or np.any(np.isinf(U_current)):
            print("Error: NaN or Inf detected in solution. Stopping.")
            break


    print(f"Simulation finished at t={t:.4f} after {iteration} iterations.")
    if results_t[-1] < t: # Avoid duplicating if last step was a save step
        results_t.append(t) 
        results_U.append(U_current.copy())


    return cell_centers, results_t, results_U