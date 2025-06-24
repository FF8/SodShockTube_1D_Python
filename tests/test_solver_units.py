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