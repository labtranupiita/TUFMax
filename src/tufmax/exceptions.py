"""
TUFMax Custom Exceptions Hierarchy.

This module defines the standard exceptions and warnings used throughout the TUFMax library.
It enables precise error handling, robust logging, and the implementation of graceful 
degradation strategies (fallbacks) when physical models fail or data is missing.

All custom exceptions inherit from TUFMaxError to allow global catching via:
    try:
        ...
    except TUFMaxError as e:
        logger.error(f"TUFMax failed: {e}")
"""

from typing import Optional


class TUFMaxError(Exception):
    """
    Base exception class for all TUFMax-specific errors.
    
    Any critical failure within the library should raise an exception derived 
    from this class. It ensures that TUFMax errors can be distinguished from 
    standard Python errors or third-party library errors.
    """
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class TUFMaxConfigurationError(TUFMaxError):
    """
    Raised when there is a critical misconfiguration in the TUFMax setup.
    
    Examples:
        - Missing essential environment variables.
        - Incompatible configuration flags (e.g., requesting high-fidelity mode without required dependencies).
        - Invalid file paths for configuration or output.
    """
    pass


class TUFMaxInputError(TUFMaxError):
    """
    Raised when user-provided input data is invalid or malformed.
    
    Examples:
        - Negative altitude values where not physically possible.
        - Malformed date/time strings that cannot be parsed.
        - Coordinates outside valid ranges (e.g., latitude > 90).
    """
    pass


class TUFMaxGeometryError(TUFMaxError):
    """
    Raised when geometric calculations fail or yield impossible configurations.
    
    Examples:
        - Line-of-Sight (LoS) intersects the Earth's surface (blocked visibility).
        - Satellite position vector falls inside the Earth's radius.
        - Invalid transformation between coordinate systems resulting in NaNs.
    """
    pass


class TUFMaxOrbitalPropagationError(TUFMaxError):
    """
    Raised specifically during orbital mechanics calculations (TLE/Keplerian propagation).
    
    Examples:
        - SGP4 propagation fails due to expired TLE epoch (>30 days).
        - Keplerian elements result in hyperbolic orbit when elliptical was expected.
        - Failure to converge on Eccentric Anomaly solution.
    """
    pass


class TUFMaxValidationError(TUFMaxError):
    """
    Raised when the Centralized Validator (Sanity Check) detects physical inconsistencies.
    
    This is the primary exception used by the 'validator' module to halt execution 
    when data violates physical laws (e.g., negative density, temperature <= 0 K).
    
    Examples:
        - Electron density ($n_e$) < 0.
        - Neutral temperature ($T_n$) <= 0 K.
        - Sum of species densities does not match total mass density within tolerance.
    """
    pass


class TUFMaxDataNotFoundError(TUFMaxError):
    """
    Raised when required external data (indices, files, network resources) cannot be retrieved.
    
    Examples:
        - Failed download of solar/geomagnetic indices (F10.7, Kp) from NOAA/SWPC.
        - Missing local coefficient files for IGRF or MSIS models.
        - Network timeout while fetching real-time TEC maps.
    """
    pass


class TUFMaxModelInitializationError(TUFMaxError):
    """
    Raised when a physics model engine fails to initialize or load.
    
    Examples:
        - Failure to load compiled Fortran libraries for NRLMSISE-00.
        - Version mismatch in SpacePy preventing IRBEM initialization.
        - Memory allocation failure when loading large GCM grids.
    """
    pass


class TUFMaxPhysicsError(TUFMaxError):
    """
    Raised when a calculation yields a result that violates fundamental physical laws.
    
    Distinct from ValidationError (which checks inputs), this checks outputs of complex derivations.
    
    Examples:
        - Conductivity tensor has negative eigenvalues (unstable medium).
        - Refractive index implies superluminal phase velocity in plasma without justification.
        - Collision frequency exceeds theoretical limits.
    """
    pass


# -----------------------------------------------------------------------------
# Non-Critical Warnings (For Graceful Degradation / Fallbacks)
# -----------------------------------------------------------------------------

class TUFMaxDegradationWarning(Warning):
    """
    Warning raised when the system switches to a lower-fidelity model (Fallback).
    
    Unlike exceptions, this warning does NOT halt execution. It informs the user 
    and the Logger that the results are valid but derived from a simplified model 
    due to the unavailability of higher-fidelity data or tools.
    
    Examples:
        - Switching from NRLMSIS 2.1 to COESA76 due to missing solar indices.
        - Using a dipole magnetic field approximation because IGRF failed.
        - Propagating TLE beyond recommended 3-day window.
    """
    def __init__(self, message: str, fallback_model: str = "Unknown", original_model: str = "Unknown"):
        super().__init__(message)
        self.fallback_model = fallback_model
        self.original_model = original_model

    def __str__(self):
        return (f"{super().__str__()} "
                f"[Fallback: {self.fallback_model} replaced {self.original_model}]")
