import numpy as np

def setup_mesh(N_cells, domain_length):
    """
    Sets up a simple 1D uniform mesh.

    Args:
        N_cells (int): Number of cells.
        domain_length (float): Total length of the domain.

    Returns:
        tuple: (dx, cell_centers, cell_interfaces)
               dx (float): Cell width.
               cell_centers (np.ndarray): Coordinates of cell centers.
               cell_interfaces (np.ndarray): Coordinates of cell interfaces.
    """
    dx = domain_length / N_cells
    # Cell interfaces: N_cells + 1 points
    cell_interfaces = np.linspace(0, domain_length, N_cells + 1)
    # Cell centers: N_cells points
    cell_centers = cell_interfaces[:-1] + dx / 2.0
    
    print(f"1D Mesh Setup: N_cells={N_cells}, domain_length={domain_length}, dx={dx}")
    
    return dx, cell_centers, cell_interfaces