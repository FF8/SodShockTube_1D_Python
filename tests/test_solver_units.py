import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))



# --- Imports (must come AFTER the sys.path modification) ---
# Now that 'src' is on the path, we can import its modules directly.
import utils
import schemes_spatial
import riemann_solvers 

# --- Test Data and Fixtures ---
# Use a fixture to provide consistent test data to multiple tests
@pytest.fixture
def basic_states():
    """Provides a standard set of left/right states for testing."""
    gamma = 1.4
    # Primitives: rho, u, p
    prim_L = np.array([1.0, 0.75, 1.0])
    prim_R = np.array([0.125, 0.0, 0.1])
    # Corresponding conserved states
    U_L = utils.primitive_to_conserved(*prim_L, gamma)
    U_R = utils.primitive_to_conserved(*prim_R, gamma)
    return {
        "gamma": gamma,
        "prim_L": prim_L, "prim_R": prim_R,
        "U_L": U_L, "U_R": U_R
    }

# --- Unit Tests for Utility Functions ---

def test_conversion_roundtrip(basic_states):
    """
    Tests that converting primitive -> conserved -> primitive returns the original values.
    """
    # GIVEN: A known primitive state
    gamma = basic_states["gamma"]
    prim_L_original = basic_states["prim_L"]
    
    # WHEN: We perform a round-trip conversion
    U_L = utils.primitive_to_conserved(*prim_L_original, gamma)
    prim_L_new = utils.conserved_to_primitive(U_L, gamma)
    
    # THEN: The result should be very close to the original
    assert np.allclose(prim_L_original, prim_L_new)


def test_minmod_limiter():
    """Tests the three main cases of the minmod limiter."""
    # Case 1: Same sign, a has smaller magnitude
    assert schemes_spatial.minmod_limiter(2, 5) == 2
    # Case 2: Different signs
    assert schemes_spatial.minmod_limiter(2, -5) == 0
    # Case 3: One value is zero
    assert schemes_spatial.minmod_limiter(0, 5) == 0


# --- Unit Tests for Riemann Solvers ---

def test_hllc_consistency(basic_states):
    """
    Tests that if U_L == U_R, the HLLC flux equals the analytical flux.
    This is a fundamental property of a consistent Riemann solver.
    """
    # GIVEN: Identical left and right states
    gamma = basic_states["gamma"]
    U_L = basic_states["U_L"]
    
    # WHEN: We calculate the analytical flux and the HLLC flux
    analytical_flux = utils.calculate_flux(U_L, gamma)
    hllc_flux = riemann_solvers.hllc_solver(U_L, U_L, gamma)
    
    # THEN: The two fluxes must be approximately equal
    assert np.allclose(analytical_flux, hllc_flux)
