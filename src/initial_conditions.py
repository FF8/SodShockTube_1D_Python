import numpy as np
from utils import primitive_to_conserved

def initialize_state(N_cells, N_ghost, physical_slice, cell_centers, cell_interfaces, gamma_val,
                        problem_type="sod_shock_tube", **problem_params):
    """
    Initializes the 1D conserved state vector U.

    Args:
        N_cells (int): Number of cells.
        N_ghost (int): Number of ghost cells on each side.
        physical_slice (slice): A slice object to access the physical part of the array.
        cell_centers (np.ndarray): Coordinates of cell centers.
        cell_interfaces (np.ndarray): Coordinates of cell interfaces (used for diaphragm).
        gamma_val (float): Adiabatic index.
        problem_type (str): Type of problem (e.g., "sod_shock_tube", "uniform_flow_1d").
        **problem_params: Keyword arguments for specific problem types.
                          For "sod_shock_tube": rho_L, u_L, p_L, rho_R, u_R, p_R.
                          For "uniform_flow_1d": rho0, u0, p0.

    Returns:
        np.ndarray: Initialized 1D state vector U (shape: 3, N_cells + 2*N_ghost).
    """
    # Total number of cells in the computational array
    N_total = N_cells + 2 * N_ghost
    U_initial = np.zeros((3, N_total)) # rho, rho*u, E

    if problem_type == "sod_shock_tube":
        print(f"\nInitializing 1D Sod Shock Tube for {N_cells} cells:")
        # Default Sod parameters if not provided in problem_params
        rho_L = problem_params.get("rho_L", 1.0)
        u_L = problem_params.get("u_L", 0.0)
        p_L = problem_params.get("p_L", 1.0)

        rho_R = problem_params.get("rho_R", 0.125)
        u_R = problem_params.get("u_R", 0.0)
        p_R = problem_params.get("p_R", 0.1)
        
        # Use middle of domain defined by interfaces as diaphragm position
        # This assumes cell_interfaces are for the full domain [0, Lx]
        #diaphragm_x_position = (cell_interfaces[0] + cell_interfaces[-1]) / 2.0

        # Diaphragm is in the middle of the PHYSICAL domain.
        # We must use the indices corresponding to the physical interfaces.
        first_physical_interface_idx = N_ghost
        last_physical_interface_idx = N_ghost + N_cells
        diaphragm_x_position = (cell_interfaces[first_physical_interface_idx] + 
                                cell_interfaces[last_physical_interface_idx]) / 2.0       
        
        print(f"  Left State: rho={rho_L}, u={u_L}, p={p_L}")
        print(f"  Right State: rho={rho_R}, u={u_R}, p={p_R}")
        print(f"  Diaphragm at x = {diaphragm_x_position:.3f}")

        U_L_conserved = primitive_to_conserved(rho_L, u_L, p_L, gamma_val)
        U_R_conserved = primitive_to_conserved(rho_R, u_R, p_R, gamma_val)

        # Loop over the PHYSICAL cells to set the initial conditions
        # The loop range is defined by the `physical_slice` object
        for i in range(physical_slice.start, physical_slice.stop):
            # Check the position of the cell center relative to the diaphragm
            if cell_centers[i] < diaphragm_x_position:
                U_initial[:, i] = U_L_conserved
            else:
                U_initial[:, i] = U_R_conserved
    
    
    elif problem_type == "uniform_flow_1d":
        rho0 = problem_params.get("rho0", 1.0)
        u0 = problem_params.get("u0", 0.1)
        p0 = problem_params.get("p0", 1.0)
        print(f"\nInitializing 1D Uniform Flow: rho0={rho0}, u0={u0}, p0={p0}")
        U_cell_conserved = primitive_to_conserved(rho0, u0, p0, gamma_val)
        # Fill the entire U_initial array with this constant state
        for i in range(N_cells):
            U_initial[:, i] = U_cell_conserved
          
            
    else:
        raise ValueError(f"Unknown 1D problem_type in initialize_state_1d: {problem_type}")

    return U_initial

