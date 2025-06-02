import numpy as np

def euler_step(U_current, dt, RHS_function, **rhs_args):
    """
    Performs a single Forward Euler time step.

    Args:
        U_current (np.ndarray): The current state vector array.
        dt (float): The time step size.
        RHS_function (callable): The function that computes dU/dt (the RHS).
                                 It should accept U_current and other necessary 
                                 arguments via **rhs_args.
        **rhs_args: Additional arguments required by the RHS_function 
                    (e.g., N_cells, dx, gamma_val, scheme, Riemann solver choice, BCs, etc.).

    Returns:
        np.ndarray: The state vector array at the next time level (U_next).
    """
    RHS_n = RHS_function(U_current, **rhs_args)
    U_next = U_current + dt * RHS_n
    return U_next