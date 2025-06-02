# constants.py

# Numerical tolerance
EPSILON = 1e-9  # Or your preferred small number

# Boundary Condition Types
BC_TRANSMISSIVE = "transmissive"
BC_REFLECTIVE = "reflective"

# HLLC Wave Speed Estimation Methods
HLLC_WAVE_SPEED_SIMPLE = "simple_bounds"
HLLC_WAVE_SPEED_ROE = "roe_einfeldt"

# Spatial Scheme Options
SCHEME_FIRST_ORDER = "first_order"
SCHEME_MUSCL = "muscl"

# Time Integrator Options
INTEGRATOR_EULER = "euler"
INTEGRATOR_SSPRK2 = "ssprk2"

# Riemann Solver Options
SOLVER_HLL = "hll"
SOLVER_HLLC = "hllc"

# EOS Types (Anticipating future use)
EOS_IDEAL_GAS = "ideal_gas"
# Add other constants as your project grows
