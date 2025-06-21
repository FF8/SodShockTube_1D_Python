import numpy as np
from utils import primitive_to_conserved, conserved_to_primitive
import constants

def apply_boundary_conditions(U_arr, N_ghost, bc_left, bc_right, gamma_eos):
    """
    Applies boundary conditions by filling the ghost cells of the state vector U.

    Args:
        U_arr (np.ndarray): The full state vector array, including ghost cell regions.
        N_ghost (int): The number of ghost cells on each side.
        bc_left (str): The boundary condition type for the left side.
        bc_right (str): The boundary condition type for the right side.
        gamma_eos (float): Adiabatic index.

    Returns:
        np.ndarray: The state vector with ghost cells correctly populated.
    """
    N_total = U_arr.shape[1]
    
    # --- Left Boundary ---
    # Index of the first physical cell
    first_physical_cell_idx = N_ghost
    
    # Get the state of the first physical cell
    U_first_phys = U_arr[:, first_physical_cell_idx]
    rho_fp, u_fp, p_fp = conserved_to_primitive(U_first_phys, gamma_eos)

    for i in range(N_ghost):
        ghost_cell_idx = N_ghost - 1 - i
        
        if bc_left == constants.BC_REFLECTIVE:
            # Reflective (solid wall): same rho, p, but opposite velocity
            U_ghost = primitive_to_conserved(rho_fp, -u_fp, p_fp, gamma_eos)
            U_arr[:, ghost_cell_idx] = U_ghost
        else: # Default to Transmissive/Outflow
            # Transmissive: state in ghost cell is same as first physical cell
            U_arr[:, ghost_cell_idx] = U_first_phys

    # --- Right Boundary ---
    # Index of the last physical cell
    last_physical_cell_idx = N_total - N_ghost - 1

    # Get the state of the last physical cell
    U_last_phys = U_arr[:, last_physical_cell_idx]
    rho_lp, u_lp, p_lp = conserved_to_primitive(U_last_phys, gamma_eos)

    for i in range(N_ghost):
        ghost_cell_idx = N_total - N_ghost + i
        
        if bc_right == constants.BC_REFLECTIVE:
            # Reflective (solid wall): same rho, p, but opposite velocity
            U_ghost = primitive_to_conserved(rho_lp, -u_lp, p_lp, gamma_eos)
            U_arr[:, ghost_cell_idx] = U_ghost
        else: # Default to Transmissive/Outflow
            # Transmissive: state in ghost cell is same as last physical cell
            U_arr[:, ghost_cell_idx] = U_last_phys
            
    return U_arr