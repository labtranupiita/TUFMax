"""
TUFMax Physical Constants and Validation Thresholds.

This module centralizes all physical constants, geodetic parameters, and empirical thresholds
used throughout the TUFMax library. By defining these in a single location, we ensure:
    1. Dimensional consistency via astropy.units.Quantity.
    2. Easy updates when standards change (e.g., WGS84 -> WGS84-G1150).
    3. Elimination of "magic numbers" scattered across the codebase.

Constants are categorized into Fundamental, Geophysical, and Empirical Thresholds.
"""

import astropy.units as u
import astropy.constants as const
import numpy as np

# -----------------------------------------------------------------------------
# 0. Unit Registration & Helpers
# -----------------------------------------------------------------------------

# Ensure SFU (Solar Flux Unit) is available. 
# In some astropy versions, it's not in the default namespace.
# 1 SFU = 10^-22 W m^-2 Hz^-1
if not hasattr(u, 'SFU'):
    try:
        # Try to import from physical units if available
        u.SFU = u.def_unit('SFU', 1e-22 * u.W / (u.m**2 * u.Hz))
    except (AttributeError, TypeError):
        # Fallback definition if def_unit fails or isn't needed
        pass 

# -----------------------------------------------------------------------------
# 1. Fundamental Physical Constants (SI Units)
# Wrapped from astropy.constants for convenience and explicit typing.
# -----------------------------------------------------------------------------

#: Speed of light in vacuum (exact)
C_LIGHT = const.c.to(u.m / u.s)

#: Elementary charge (positive)
E_CHARGE = const.e.to(u.C)

#: Electron mass
M_ELECTRON = const.m_e.to(u.kg)

#: Proton mass
M_PROTON = const.m_p.to(u.kg)

#: Neutron mass
M_NEUTRON = const.m_n.to(u.kg)

#: Atomic mass unit (unified)
AMU = const.u.to(u.kg)

#: Vacuum permeability
MU_0 = const.mu0.to(u.N / u.A**2)

#: Vacuum permittivity
EPSILON_0 = const.eps0.to(u.F / u.m)

#: Boltzmann constant
K_BOLTZMANN = const.k_B.to(u.J / u.K)

#: Avogadro constant
N_AVOGADRO = const.N_A.to(1 / u.mol)

#: Gravitational constant
G_GRAVITY = const.G.to(u.m**3 / (u.kg * u.s**2))

# -----------------------------------------------------------------------------
# 2. Geophysical Parameters (Earth Model: WGS84)
# Standard values for geometry, gravity, and magnetic references.
# -----------------------------------------------------------------------------

#: Earth equatorial radius (WGS84 semi-major axis)
EARTH_RADIUS_EQ = 6378.137 * u.km

#: Earth polar radius (WGS84 semi-minor axis)
EARTH_RADIUS_POL = 6356.752 * u.km

#: Earth mean radius (volumetric equivalent for simple approximations)
EARTH_RADIUS_MEAN = 6371.0 * u.km

#: Earth flattening factor (WGS84)
EARTH_FLATTENING = 1 / 298.257223563

#: Earth standard gravitational parameter (GM)
EARTH_GM = 3.986004418e14 * u.m**3 / u.s**2

#: Earth rotation rate (omega)
EARTH_ROTATION_RATE = 7.2921159e-5 * u.rad / u.s

#: Magnetic dipole moment (approximate centered dipole for fallbacks)
EARTH_MAG_DIPOLE_MOMENT = 7.79e22 * u.A * u.m**2

#: Reference altitude for homopause (transition from mixing to diffusion)
HOMOPAUSE_ALTITUDE = 100.0 * u.km

# -----------------------------------------------------------------------------
# 3. Species Masses (for Neutral Atmosphere & Plasma)
# Pre-calculated in kg for performance, derived from AMU.
# -----------------------------------------------------------------------------

NEUTRAL_MASSES = {
    'N2': 28.0134 * AMU,
    'O2': 31.9988 * AMU,
    'O': 15.9994 * AMU,
    'He': 4.0026 * AMU,
    'H': 1.0078 * AMU,
    'Ar': 39.948 * AMU,
    'N': 14.0067 * AMU,
    'NO': 30.0061 * AMU,
    'CO2': 44.0095 * AMU,
}

ION_MASSES = {
    'O+': 15.9994 * AMU,
    'H+': 1.0078 * AMU,
    'He+': 4.0026 * AMU,
    'N+': 14.0067 * AMU,
    'NO+': 30.0061 * AMU,
    'O2+': 31.9988 * AMU,
    'N2+': 28.0134 * AMU,
}

# Charge states for ions (multiples of elementary charge)
ION_CHARGES = {
    'O+': 1, 'H+': 1, 'He+': 1, 'N+': 1, 
    'NO+': 1, 'O2+': 1, 'N2+': 1,
    'O++': 2 # Example of doubly charged ion if needed
}

# -----------------------------------------------------------------------------
# 4. Empirical Validation Thresholds (Sanity Check Limits)
# Used by the Centralized Validator to detect anomalies.
# -----------------------------------------------------------------------------

#: Minimum allowed elevation angle for Line of Sight (degrees)
MIN_ELEVATION_ANGLE = 0.0 * u.deg

#: Maximum allowed age for TLE data before warning (days)
MAX_TLE_AGE_WARNING = 3.0 * u.day

#: Maximum allowed age for TLE data before critical error (days)
MAX_TLE_AGE_CRITICAL = 30.0 * u.day

#: Historical range for F10.7 solar flux (SFU) - Lower bound
# Define SFU explicitly if not present in u namespace to avoid AttributeError
_SFU_UNIT = getattr(u, 'SFU', 1e-22 * u.W / (u.m**2 * u.Hz))
F107_MIN_HISTORICAL = 50.0 * _SFU_UNIT

#: Historical range for F10.7 solar flux (SFU) - Upper bound
F107_MAX_HISTORICAL = 400.0 * _SFU_UNIT

#: Tolerance for mass density conservation check (fractional, 5%)
MASS_CONSERVATION_TOLERANCE = 0.05

#: Minimum temperature threshold for neutral atmosphere (Kelvin)
MIN_TEMP_NEUTRAL = 1.0 * u.K

#: Maximum gradient threshold for plasma discontinuity detection (fractional change per km)
# If density changes by >50% between adjacent points in transition zone
PLASMA_GRADIENT_THRESHOLD = 0.5 / u.km 

#: Minimum electron density threshold (m^-3) to avoid division by zero in some calcs
MIN_ELECTRON_DENSITY = 1e-6 / u.m**3

#: Smoothing window size for active correction in transition zones (number of points)
SMOOTHING_WINDOW_SIZE = 3

# -----------------------------------------------------------------------------
# 5. Unit Conversion Helpers (Commonly Used)
# -----------------------------------------------------------------------------

#: Conversion factor from SFU (Solar Flux Units) to SI (W/m^2/Hz)
# 1 SFU = 10^-22 W m^-2 Hz^-1
SFU_TO_SI = 1e-22 * u.W / (u.m**2 * u.Hz)

#: Conversion from km/s to m/s
KM_S_TO_M_S = 1000.0 * u.m / u.km / u.s
