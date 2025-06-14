import numpy as np
import constants 
import os
import shutil
from mesh_setup import setup_mesh
from initial_conditions import initialize_state
from utils import conserved_to_primitive, primitive_to_conserved
from riemann_solvers import hll_solver, hllc_solver
from schemes_spatial import calculate_muscl_slopes, minmod_limiter 

# gamma would be passed as an argument to functions here, not global to this file
# unless run_simulation sets a module-level one for its callees if they are also in this file

def calculate_rhs_for_stage(U_stage, N_cells, dx, gamma_eos, scheme,
                            riemann_solver_choice,
                            bc_left, bc_right,
                            hllc_wave_method):


    """
    Calculates the Right-Hand Side (RHS = dU/dt) for a given 1D stage U.
    Includes MUSCL reconstruction and chosen Riemann solver.
    U_stage is expected to be (3, N_cells)
    """
    slopes_s = np.zeros_like(U_stage) # Array for limited slopes, shape (3, N_cells)

    if scheme == constants.SCHEME_MUSCL:
        slopes_s = calculate_muscl_slopes(U_stage, N_cells, limiter_func=minmod_limiter)
    
    # F_numerical_fluxes stores fluxes at interfaces: F_{i-1/2} and F_{i+1/2} for cell i
    # Size: (3, N_cells + 1). F_numerical_fluxes[:, j] is flux at interface j.
    # Interface 0: left of cell 0. Interface N_cells: right of cell N_cells-1.
    F_numerical_fluxes = np.zeros((3, N_cells + 1))

    for j_interface_idx in range(N_cells + 1): # Loop through all N_cells+1 interfaces
        U_L_eff = np.zeros(3) # Effective state to the LEFT of interface j
        U_R_eff = np.zeros(3) # Effective state to the RIGHT of interface j

        if j_interface_idx == 0:  # Leftmost boundary interface (left of cell 0)
            # State from physical cell 0
            U_physical_cell_R = U_stage[:, 0] 
            rho_phys_R, u_phys_R, p_phys_R = conserved_to_primitive(U_physical_cell_R, gamma_eos)
            
            if bc_left == constants.BC_REFLECTIVE:
                # Ghost cell state (L_eff)
                U_L_eff = primitive_to_conserved(rho_phys_R, -u_phys_R, p_phys_R, gamma_eos)
            else: # Transmissive (default)
                U_L_eff = U_physical_cell_R # Ghost state = first physical cell state (first-order for ghost)

            # Right state for Riemann problem is from cell 0, reconstructed if MUSCL
            if scheme == constants.SCHEME_MUSCL:
                U_R_eff = U_stage[:, 0] - 0.5 * slopes_s[:, 0] # Value at LEFT face of cell 0
            else: # First-order
                U_R_eff = U_stage[:, 0]

        elif j_interface_idx == N_cells:  # Rightmost boundary interface (right of cell N_cells-1)
            # State from physical cell N_cells-1
            U_physical_cell_L = U_stage[:, N_cells-1]
            rho_phys_L, u_phys_L, p_phys_L = conserved_to_primitive(U_physical_cell_L, gamma_eos)
            
            # Left state for Riemann problem is from cell N_cells-1, reconstructed if MUSCL
            if scheme == constants.SCHEME_MUSCL:
                U_L_eff = U_stage[:, N_cells-1] + 0.5 * slopes_s[:, N_cells-1] # Value at RIGHT face of cell N-1
            else: # First-order
                U_L_eff = U_stage[:, N_cells-1]

            if bc_right == constants.BC_REFLECTIVE:
                # Ghost cell state (R_eff)
                U_R_eff = primitive_to_conserved(rho_phys_L, -u_phys_L, p_phys_L, gamma_eos)
            else: # Transmissive (default)
                U_R_eff = U_physical_cell_L # Ghost state = last physical cell state (first-order for ghost)
            
        else:  # Internal interface j (between cell j-1 (index L) and cell j (index R))
            cell_idx_L = j_interface_idx - 1 
            cell_idx_R = j_interface_idx   

            if scheme == constants.SCHEME_MUSCL:
                U_L_eff = U_stage[:, cell_idx_L] + 0.5 * slopes_s[:, cell_idx_L]
                U_R_eff = U_stage[:, cell_idx_R] - 0.5 * slopes_s[:, cell_idx_R]
            else: # First-order
                U_L_eff = U_stage[:, cell_idx_L]
                U_R_eff = U_stage[:, cell_idx_R]
        
        # Call chosen Riemann solver
        if riemann_solver_choice == constants.SOLVER_HLL:
            F_numerical_fluxes[:, j_interface_idx] = hll_solver(U_L_eff, U_R_eff, gamma_eos)
        elif riemann_solver_choice == constants.SOLVER_HLLC:
            F_numerical_fluxes[:, j_interface_idx] = hllc_solver(U_L_eff, U_R_eff, gamma_eos, 
                                                                wave_speed_method=hllc_wave_method)
        else:
            raise ValueError(f"Unknown Riemann solver: {riemann_solver_choice}")

    # Compute Right-Hand Side (RHS) for each cell
    RHS_1D = np.zeros_like(U_stage) # Shape (3, N_cells)
    for i_cell in range(N_cells):
        # RHS_i = - (Flux_right_face_of_cell_i - Flux_left_face_of_cell_i) / dx
        # Flux at right face of cell i_cell is F_numerical_fluxes[:, i_cell+1]
        # Flux at left face of cell i_cell is F_numerical_fluxes[:, i_cell]
        RHS_1D[:, i_cell] = -(F_numerical_fluxes[:, i_cell+1] - F_numerical_fluxes[:, i_cell]) / dx
        
    return RHS_1D



def run_simulation(N_cells, domain_length, t_final, C_cfl,
                   problem_type,
                   scheme, time_integrator, riemann_solver,
                   bc_left, bc_right,
                   hllc_wave_speed_config,
                   gamma_eos, save_interval=-1.0): 
    """
    Main loop for the 1D Euler solver.
    If save_interval > 0, stores snapshots at specified time intervals.
    """

    dx, cell_centers, cell_interfaces = setup_mesh(N_cells, domain_length)
    
    U_current = initialize_state(N_cells, cell_centers, cell_interfaces, gamma_eos,
                                    problem_type=problem_type, ) 
    
    
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
        for i in range(N_cells):
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
            "N_cells": N_cells, "dx": dx, "gamma_eos": gamma_eos, "scheme": scheme,
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