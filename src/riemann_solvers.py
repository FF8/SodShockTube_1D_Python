# riemann_solvers.py
import numpy as np
from utils import conserved_to_primitive, calculate_flux 
# from constants import EPSILON 
import constants 

def hll_solver(U_L_input, U_R_input, gamma_val):

    rho_L, u_L, p_L = conserved_to_primitive(U_L_input, gamma_val)
    rho_R, u_R, p_R = conserved_to_primitive(U_R_input, gamma_val)

    F_L = calculate_flux(U_L_input, gamma_val)
    F_R = calculate_flux(U_R_input, gamma_val)

    a_L = np.sqrt(gamma_val * p_L / rho_L)
    a_R = np.sqrt(gamma_val * p_R / rho_R)

    S_L = min(u_L - a_L, u_R - a_R) # Using Simple Eigenvalue Bounds
    S_R = max(u_L + a_L, u_R + a_R)

    if 0 <= S_L:
        F_HLL = F_L
    elif S_L < 0 < S_R:
        denominator = S_R - S_L
        # Check for denominator effectively zero, though physics should prevent for distinct L/R states
        if np.abs(denominator) < 1e-9:
            # This implies S_L is very close to S_R, meaning a very narrow or non-existent fan.
            # Could indicate identical states or issues. Simple average as fallback.
            F_HLL = 0.5 * (F_L + F_R)
        else:
            F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R_input - U_L_input)) / denominator
    elif S_R <= 0:
        F_HLL = F_R
    else: # Should not be reached if S_L, S_R are 
        F_HLL = 0.5 * (F_L + F_R) # Fallback for unexpected configuration
    return F_HLL


def hllc_solver(U_L_input, U_R_input, gamma_val, 
                wave_speed_method=constants.HLLC_WAVE_SPEED_SIMPLE):
    """
    HLLC (Harten-Lax-van Leer-Contact) Riemann solver for 1D Euler equations.
    Inputs U_L_input, U_R_input are 3-component [rho, rho*u, E].
    """
    # Step 0: Get primitive variables and physical fluxes
    rho_L, u_L, p_L = conserved_to_primitive(U_L_input, gamma_val)
    F_L = calculate_flux(U_L_input, gamma_val)

    rho_R, u_R, p_R = conserved_to_primitive(U_R_input, gamma_val)
    F_R = calculate_flux(U_R_input, gamma_val)

    # Sound speeds
    a_L = np.sqrt(gamma_val * p_L / (rho_L + constants.EPSILON)) # Added EPSILON
    a_R = np.sqrt(gamma_val * p_R / (rho_R + constants.EPSILON)) # Added EPSILON

    # Step 1: Estimate wave speeds S_L and S_R
    if wave_speed_method == constants.HLLC_WAVE_SPEED_SIMPLE:
        S_L = min(u_L - a_L, u_R - a_R)
        S_R = max(u_L + a_L, u_R + a_R)
    elif wave_speed_method == constants.HLLC_WAVE_SPEED_ROE:
        # Calculate Roe Averages
        sqrt_rho_L = np.sqrt(rho_L)
        sqrt_rho_R = np.sqrt(rho_R)

        roe_den_sqrt_rho_sum = sqrt_rho_L + sqrt_rho_R
        
        # Handle potential division by zero if both densities are very small
        if abs(roe_den_sqrt_rho_sum) < constants.EPSILON:
            u_roe = (u_L + u_R) / 2.0
            # A simple fallback for a_roe if densities are too low for proper enthalpy average
            if rho_L > constants.EPSILON and rho_R < constants.EPSILON * rho_L : a_roe = a_L
            elif rho_R > constants.EPSILON and rho_L < constants.EPSILON * rho_R : a_roe = a_R
            else: a_roe = (a_L + a_R) / 2.0
        else:
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / roe_den_sqrt_rho_sum

            # Specific enthalpy h = (E + p) / rho
            # U_L_input[2] is E_L, U_R_input[2] is E_R
            h_L = (U_L_input[2] + p_L) / (rho_L + constants.EPSILON)
            h_R = (U_R_input[2] + p_R) / (rho_R + constants.EPSILON)
            h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) / roe_den_sqrt_rho_sum

            a_roe_sq_arg = h_roe - 0.5 * u_roe**2
            # Ensure argument for sqrt is non-negative for a_roe
            a_roe = np.sqrt(max(constants.EPSILON, (gamma_val - 1.0) * a_roe_sq_arg))
        
        # Einfeldt's speeds
        S_L = u_roe - a_roe
        S_R = u_roe + a_roe
    else:
        raise ValueError(f"Unknown wave_speed_method for HLLC: {wave_speed_method}")

    # Step 2: Estimate contact wave speed S_M
    den_SM = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    num_SM = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)

    if abs(den_SM) < constants.EPSILON:
        if np.allclose(U_L_input, U_R_input, atol=constants.EPSILON):
            return F_L 
        S_M = (u_L + u_R) / 2.0 # Fallback for S_M
        # print(f"Warning: Denominator for S_M near zero. S_M set to avg vel: {S_M}")
    else:
        S_M = num_SM / den_SM # Using Toro, Eq. 10.38
    
    # Optional: Check/Clamp S_M to be between S_L and S_R for robustness
    #S_L_eff, S_R_eff = min(S_L, S_R), max(S_L, S_R) # Ensure S_L_eff <= S_R_eff
    #if not (S_L_eff - constants.EPSILON <= S_M <= S_R_eff + constants.EPSILON):
    #    print(f"Warning: S_M={S_M} out of bounds [S_L={S_L}, S_R={S_R}]. Clamping or reverting to HLL might be needed.")
    #    S_M = max(S_L_eff, min(S_M, S_R_eff)) # Simple clamping


    # Step 3: Calculate star region pressure p_star
    # Using R-state based formula (Toro pg. 325, also derived from pK + rhoK(uK-SK)(uK-SM))
    p_star = rho_R * (u_R - S_R) * (S_M - u_R) + p_R 
    p_star = max(constants.EPSILON, p_star) # Ensure positivity

    U_L_star, U_R_star = np.zeros(3), np.zeros(3)

    # Calculate U_L_star (rho_L_star, momentum_L_star, E_L_star)
    if abs(S_L - S_M) < constants.EPSILON: # If S_L is the contact speed
        U_L_star = np.copy(U_L_input) # Star state is initial state
    else:
        safe_den_L_star = S_L - S_M
        # Robust division
        if abs(safe_den_L_star) < constants.EPSILON: 
            safe_den_L_star = np.sign(safe_den_L_star + constants.EPSILON) * constants.EPSILON if safe_den_L_star !=0 else constants.EPSILON
        
        rho_L_star_val = rho_L * (S_L - u_L) / safe_den_L_star
        rho_L_star_val = max(constants.EPSILON, rho_L_star_val)
        U_L_star[0] = rho_L_star_val
        U_L_star[1] = rho_L_star_val * S_M
        # Energy will be set using p_star later
    
    # Calculate U_R_star (rho_R_star, momentum_R_star, E_R_star)
    if abs(S_R - S_M) < constants.EPSILON: # If S_R is the contact speed
        U_R_star = np.copy(U_R_input) # Star state is initial state
    else:
        safe_den_R_star = S_R - S_M
        if abs(safe_den_R_star) < constants.EPSILON:
            safe_den_R_star = np.sign(safe_den_R_star + constants.EPSILON) * constants.EPSILON if safe_den_R_star !=0 else constants.EPSILON

        rho_R_star_val = rho_R * (S_R - u_R) / safe_den_R_star
        rho_R_star_val = max(constants.EPSILON, rho_R_star_val)
        U_R_star[0] = rho_R_star_val
        U_R_star[1] = rho_R_star_val * S_M
        # Energy will be set using p_star later
    
    # Calculate E_L_star and E_R_star using p_star and derived/copied rho_K_star
    safe_gamma_minus_1 = gamma_val - 1.0
    if abs(safe_gamma_minus_1) < constants.EPSILON: 
        safe_gamma_minus_1 = np.sign(safe_gamma_minus_1+constants.EPSILON) * constants.EPSILON if safe_gamma_minus_1 !=0 else constants.EPSILON

    U_L_star[2] = p_star / safe_gamma_minus_1 + 0.5 * U_L_star[0] * S_M**2 
    U_R_star[2] = p_star / safe_gamma_minus_1 + 0.5 * U_R_star[0] * S_M**2

    # Step 4: Calculate HLLC flux based on regions
    F_HLLC = np.zeros(3) # Initialize
    if 0 <= S_L:
        F_HLLC = F_L
    elif S_L < 0 <= S_M: # Toro uses S_L <= 0 <= S_M
        F_HLLC = F_L + S_L * (U_L_star - U_L_input)
    elif S_M < 0 < S_R:  # Toro uses S_M <= 0 <= S_R
        F_HLLC = F_R + S_R * (U_R_star - U_R_input)
    elif S_R <= 0:
        F_HLLC = F_R
    else: 
        # This fallback should ideally not be hit if S_L <= S_M <= S_R
        # and the conditions above correctly cover all cases of 0 relative to S_L, S_M, S_R.
        # print(f"Warning: HLLC solver entered unexpected region (S_L={S_L:.3f}, S_M={S_M:.3f}, S_R={S_R:.3f}). Defaulting to HLL-like average.")
        safe_den_SR_SL = S_R - S_L
        if abs(safe_den_SR_SL) < constants.EPSILON:
            F_HLLC = 0.5 * (F_L + F_R)
        else:
            F_HLLC = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R_input - U_L_input)) / safe_den_SR_SL
            
    return F_HLLC
