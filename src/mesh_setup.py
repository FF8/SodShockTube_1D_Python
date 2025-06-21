import numpy as np


def setup_mesh(N_cells, domain_length, N_ghost=2):
    """
    Sets up a 1D uniform mesh with ghost cells at both ends.

    Args:
        N_cells (int): Number of PHYSICAL cells.
        domain_length (float): Length of the PHYSICAL domain.
        N_ghost (int): Number of ghost cells to add on EACH side.

    Returns:
        tuple: (dx, cell_centers, cell_interfaces, physical_slice)
            dx (float): Cell width (based on physical domain).
            cell_centers (np.ndarray): Coordinates of ALL cell centers (including ghost cells).
            cell_interfaces (np.ndarray): Coordinates of ALL cell interfaces.
            physical_slice (slice): A slice object to easily access the physical cells.
    """
    # dx is determined by the physical domain and physical cells
    dx = domain_length / N_cells
    
    # Total number of cells in the computational domain
    N_total = N_cells + 2 * N_ghost
    
    # Define the slice that corresponds to the physical cells
    # This is crucial for applying initial conditions and analysis
    physical_slice = slice(N_ghost, N_ghost + N_cells)

    # The entire computational domain, including ghost cells on both sides
    # The first physical interface is at x=0. It has index N_ghost.
    # The last physical interface is at x=domain_length. It has index N_ghost + N_cells.
    # So we create N_total + 1 interfaces.
    start_interface = 0.0 - N_ghost * dx
    end_interface = domain_length + N_ghost * dx
    cell_interfaces = np.linspace(start_interface, end_interface, N_total + 1)
    
    # Cell centers are the midpoints of the interfaces
    cell_centers = cell_interfaces[:-1] + dx / 2.0
    
    print(f"Mesh Setup: N_physical={N_cells}, N_ghost={N_ghost}, N_total={N_total}, dx={dx}")
    
    return dx, cell_centers, cell_interfaces, physical_slice