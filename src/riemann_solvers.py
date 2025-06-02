# riemann_solvers.py
import numpy as np
from .utils import conserved_to_primitive, calculate_flux 
# from constants import EPSILON 
# import constants 

def hll_solver(U_L_input, U_R_input, gamma_val):

    rho_L, u_L, p_L = conserved_to_primitive(U_L_input, gamma_val)
    rho_R, u_R, p_R = conserved_to_primitive(U_R_input, gamma_val)

    F_L = calculate_flux(U_L_input, gamma_val)
    F_R = calculate_flux(U_R_input, gamma_val)

    a_L = np.sqrt(gamma_val * p_L / rho_L)
    a_R = np.sqrt(gamma_val * p_R / rho_R)

    S_L = min(u_L - a_L, u_R - a_R)
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


def hllc_solver(U_L_input, U_R_input, gamma_val, wave_speed_method=constants.HLLC_WAVE_SPEED_SIMPLE):
    # ...
    pass
