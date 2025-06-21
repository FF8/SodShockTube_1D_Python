import numpy as np

def minmod_limiter(a, b):
    """
    Minmod slope limiter.
    Returns 0 if a and b have different signs.
    Otherwise, returns the value (a or b) that has the smaller absolute magnitude.
    Applied component-wise if a and b are arrays.

    Args:
        a (np.ndarray or float): First slope/difference.
        b (np.ndarray or float): Second slope/difference.

    Returns:
        np.ndarray or float: The limited slope/difference.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Condition for different signs (or one is zero): a * b <= 0
    # If signs are the same, pick the one with smaller magnitude.
    # np.where(condition, value_if_true, value_if_false)
    return np.where(a * b <= 0, 
                    0.0,  # Slope is zero if signs differ or one is zero
                    np.where(np.abs(a) < np.abs(b), a, b)) # Else, pick the one with smaller magnitude

def calculate_muscl_slopes(U_current, N_ghost, limiter_func=minmod_limiter):
    """
    Calculates the limited slopes for each cell on a grid with ghost cells.
    This version correctly handles the ghost cell layout.

    Args:
        U_current (np.ndarray): Array of current conserved states, including ghost cells.
        N_ghost (int): Number of ghost cells on each side.
        limiter_func (callable): The slope limiter function to use.

    Returns:
        np.ndarray: Array of limited slopes for all cells.
    """
    N_total = U_current.shape[1]
    slopes_s = np.zeros_like(U_current)

    # Loop from the cell before the first physical cell (i.e., the innermost ghost cell)
    # up to the cell before the last cell in the entire array.
    # This is necessary to compute slopes for ALL cells that are needed for reconstruction
    # at the physical interfaces.
    for i in range(N_ghost - 1, N_total - 1):
        # Difference between current cell and left neighbor
        delta_L = U_current[:, i] - U_current[:, i-1]
        # Difference between right neighbor and current cell
        delta_R = U_current[:, i+1] - U_current[:, i]
        
        slopes_s[:, i] = limiter_func(delta_L, delta_R)

    return slopes_s