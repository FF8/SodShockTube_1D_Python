# Functions related to spatial discretization schemes, e.g., MUSCL.

import numpy as np
import constants 

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

def calculate_muscl_slopes(U_current, N_cells, limiter_func=minmod_limiter):
    """
    Calculates the limited slopes for each cell for MUSCL reconstruction.

    Args:
        U_current (np.ndarray): Array of current cell-averaged conserved states (shape: 3, N_cells).
        N_cells (int): Number of cells.
        limiter_func (callable): The slope limiter function to use 
                                 (e.g., minmod_limiter, van_leer_limiter).

    Returns:
        np.ndarray: Array of limited slopes (shape: 3, N_cells).
    """
    slopes_s = np.zeros_like(U_current) # Initialize slopes to zero

    if N_cells < 3:
        # Not enough cells for centered differences needed by most limiters for interior.
        # Slopes remain zero, effectively first-order.
        return slopes_s

    # Handle interior cells
    for i in range(1, N_cells - 1):
        # Difference between current cell and left neighbor
        delta_L = U_current[:, i] - U_current[:, i-1]
        # Difference between right neighbor and current cell
        delta_R = U_current[:, i+1] - U_current[:, i]
        
        slopes_s[:, i] = limiter_func(delta_L, delta_R) # Applied component-wise

    return slopes_s