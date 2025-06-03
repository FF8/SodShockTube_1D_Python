import numpy as np
import constants 

def conserved_to_primitive_1d(U_1D, gamma_val):
    """
    Converts 1D conserved variables U = [rho, rho*u, E] to primitive variables.
    U_1D can be a single state vector (1D array of 3 elements) or an array of states (3, N_cells).
    """
    is_single_state = U_1D.ndim == 1

    rho = U_1D[0] if is_single_state else U_1D[0, :]
    rho_u = U_1D[1] if is_single_state else U_1D[1, :]
    E_total = U_1D[2] if is_single_state else U_1D[2, :]

    safe_rho = np.maximum(rho, constants.EPSILON) # Avoid division by zero

    u = rho_u / safe_rho

    # p = (gamma - 1) * (E - 0.5 * rho * u^2)
    # Use original rho for kinetic energy term for consistency
    kinetic_energy_density = 0.5 * rho * u**2 
    internal_energy_density = E_total - kinetic_energy_density
    
    safe_gamma_minus_1 = gamma_val - 1.0
    # Ensure denominator for pressure calculation is safe if gamma_val is 1.0
    if abs(safe_gamma_minus_1) < constants.EPSILON:
        safe_gamma_minus_1 = np.sign(safe_gamma_minus_1 + constants.EPSILON) * constants.EPSILON if safe_gamma_minus_1 != 0 else constants.EPSILON
        
    p = safe_gamma_minus_1 * internal_energy_density
    p = np.maximum(p, constants.EPSILON) # Ensure pressure is positive

    return safe_rho, u, p # Return safe_rho

def primitive_to_conserved_1d(rho, u, p, gamma_val):
    """
    Converts 1D primitive variables (rho, u, p) to conserved variables.
    Inputs can be scalars or NumPy arrays (for vectorized operations).
    U = [rho, rho*u, E]
    """
    rho_u = rho * u
    
    safe_gamma_minus_1 = gamma_val - 1.0
    if abs(safe_gamma_minus_1) < constants.EPSILON:
        safe_gamma_minus_1 = np.sign(safe_gamma_minus_1 + constants.EPSILON) * constants.EPSILON if safe_gamma_minus_1 != 0 else constants.EPSILON

    # E = p/(gamma-1) + 0.5 * rho * u^2
    E_total = p / safe_gamma_minus_1 + 0.5 * rho * u**2

    if isinstance(rho, np.ndarray): # If inputs are arrays, stack them
        return np.vstack((rho, rho_u, E_total))
    else: # If inputs are scalars
        return np.array([rho, rho_u, E_total])

def calculate_flux_1d(U_1D, gamma_val):
    """
    Calculates the physical flux vector F for 1D Euler equations.
    F(U) = [rho*u, rho*u^2 + p, (E+p)*u]^T
    U_1D can be a single state vector or an array of states (3, N_cells).
    """
    # Get E directly from conserved variables (it's U_1D[2])
    is_single_state = U_1D.ndim == 1
    E_total = U_1D[2] if is_single_state else U_1D[2, :]
    
    rho, u, p = conserved_to_primitive_1d(U_1D, gamma_val)

    F_flux = np.zeros_like(U_1D)
    F_flux[0] = rho * u
    F_flux[1] = rho * u**2 + p
    F_flux[2] = (E_total + p) * u
    return F_flux
