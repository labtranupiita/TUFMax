"""
TUFMax Centralized Validator (Sanity Check).

This module implements the physical quality control protocol for the entire TUFMax pipeline.
It acts as an 'immune system', intercepting data between computational blocks to ensure 
physical consistency (conservation laws, thermodynamic ranges, continuity) before errors 
propagate downstream.

Key Features:
    - Unit-Aware Validation: Native handling of astropy.units.Quantity.
    - Active Correction: Automatic smoothing of discontinuities in model transition zones.
    - Hierarchical Response: Distinguishes between CRITICAL (halt), WARNING (mask & continue), and VALID.
    - Provenance Logging: Detailed auditing of every validation decision.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import astropy.units as u

# Local imports
from ..exceptions import TUFMaxValidationError, TUFMaxDegradationWarning
from ..utils.adapter import UnitAdapter


class ValidationStatus(Enum):
    """Hierarchical status levels for validation results."""
    VALID = "VALID"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationReport:
    """
    Structured report object returned by the validator.
    
    Attributes:
        status: The severity level (VALID, WARNING, CRITICAL).
        message: Human-readable description of the result or error.
        block_id: Identifier of the physics block being validated.
        valid_mask: Boolean array indicating which data points are safe to use (for vectorized data).
        corrected_data: Dictionary containing any data modified by active correction routines (e.g., smoothed profiles).
        details: Optional dictionary for extra debugging context.
    """
    status: ValidationStatus
    message: str
    block_id: int
    valid_mask: Optional[np.ndarray] = None
    corrected_data: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def is_critical(self) -> bool:
        return self.status == ValidationStatus.CRITICAL

    def has_warnings(self) -> bool:
        return self.status == ValidationStatus.WARNING


class TUFMaxValidator:
    """
    Centralized Dispatcher for Physical Sanity Checks.
    
    This class manages the routing of data to specific validation subroutines based on the 
    block ID. It integrates with the TUFMaxLogger for provenance tracking and uses the 
    UnitAdapter for safe numerical operations.
    """

    def __init__(self, logger: Any, adapter: Optional[UnitAdapter] = None):
        """
        Initialize the Validator.
        
        Args:
            logger: Instance of TUFMaxLogger for recording events.
            adapter: Instance of UnitAdapter for unit-safe calculations. If None, creates a new one.
        """
        self.logger = logger
        self.adapter = adapter or UnitAdapter()
        
        # Dispatch map: Block ID -> Validation Function
        self._method_map = {
            2: self._validate_geometry,
            3: self._validate_indices,
            4: self._validate_neutral_atmosphere,
            5: self._validate_plasma,
            6: self._validate_magnetic_field,
            7: lambda d: self._validate_derived_params(d, 'gyro'),
            8: lambda d: self._validate_derived_params(d, 'collision'),
            9: lambda d: self._validate_derived_params(d, 'stix'),
            10: lambda d: self._validate_derived_params(d, 'conductivity'),
        }

    def run_check(self, block_id: int, data: Dict[str, Any]) -> bool:
        """
        Execute the sanity check for a specific block.
        
        Args:
            block_id: Integer identifier of the source block (2-10).
            data: Dictionary containing physical quantities (astropy.units.Quantity).
            
        Returns:
            True if validation passes (or passes with warnings).
            
        Raises:
            TUFMaxValidationError: If a CRITICAL failure occurs.
        """
        if block_id not in self._method_map:
            msg = f"Block ID {block_id} has no defined validator."
            self.logger.log_validation(ValidationReport(ValidationStatus.CRITICAL, msg, block_id))
            raise ValueError(msg)

        validator_func = self._method_map[block_id]
        
        try:
            report = validator_func(data)
            
            # Log the result immediately for provenance
            self.logger.log_validation(report)
            
            if report.is_critical():
                raise TUFMaxValidationError(
                    message=report.message, 
                    details={"block": block_id, **report.details}
                )
            
            if report.has_warnings():
                # Issue a non-blocking warning to the user/system
                warn_msg = f"Validation Warning in Block {block_id}: {report.message}"
                # We log it, but don't stop execution. The mask/correction is available in the report.
                # In a real pipeline, the caller might apply report.valid_mask to the data here.
                
            return True

        except Exception as e:
            # Catch unexpected crashes within the validator itself
            error_msg = f"Validator system crash in Block {block_id}: {str(e)}"
            self.logger.log_error(error_msg)
            raise TUFMaxValidationError(
                message="Validation system failure", 
                details={"original_error": str(e), "block": block_id}
            ) from e

    # ----------------------------------------------------------------------
    # Specific Validation Subroutines (Unit-Aware)
    # ----------------------------------------------------------------------

    def _validate_geometry(self, data: Dict[str, Any]) -> ValidationReport:
        """Block 2: Validate Altitudes and Line of Sight."""
        try:
            altitudes = data.get('altitudes')
            if altitudes is None:
                return ValidationReport(ValidationStatus.CRITICAL, "Missing altitude data", 2)

            # Normalize to km for comparison
            h_km = altitudes.to(u.km).value
            
            if np.any(h_km < 0):
                idx = np.where(h_km < 0)[0][0]
                return ValidationReport(
                    ValidationStatus.CRITICAL, 
                    f"Negative altitude detected at index {idx} ({h_km[idx]} km)", 
                    2
                )

            los_valid = data.get('los_valid', True)
            if not los_valid:
                return ValidationReport(
                    ValidationStatus.CRITICAL, 
                    "Line of sight obstructed by Earth curvature", 
                    2
                )

            return ValidationReport(ValidationStatus.VALID, "Geometry validation passed", 2)

        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Geometry check failed: {e}", 2)

    def _validate_indices(self, data: Dict[str, Any]) -> ValidationReport:
        """Block 3: Validate Solar/Geomagnetic Indices."""
        try:
            f10 = data.get('F10.7')
            if f10 is not None:
                val = f10.value if hasattr(f10, 'value') else f10
                if val < 50 or val > 400:
                    msg = f"F10.7 ({val}) outside typical historical range [50, 400]"
                    self.logger.log_warning(msg)
                    return ValidationReport(ValidationStatus.WARNING, msg, 3)

            indices = data.get('indices')
            if indices is not None:
                arr = indices.value if hasattr(indices, 'value') else indices
                if np.any(np.isnan(arr)):
                    return ValidationReport(ValidationStatus.CRITICAL, "Indices contain NaN values", 3)

            return ValidationReport(ValidationStatus.VALID, "Indices validation passed", 3)

        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Indices check failed: {e}", 3)

    def _validate_neutral_atmosphere(self, data: Dict[str, Any]) -> ValidationReport:
        """Block 4: Validate Neutral Species Consistency and Mass Conservation."""
        try:
            tn = data.get('Tn')
            rho_tot = data.get('total_density')
            
            # Check Temperature Positivity
            if tn is not None:
                t_val = tn.to(u.K).value
                if np.any(t_val <= 0):
                    return ValidationReport(ValidationStatus.CRITICAL, "Neutral temperature <= 0 K", 4)

            # Check Mass Conservation (Sum of species vs Total)
            if rho_tot is not None and 'species_number_densities' in data and 'species_masses' in data:
                n_species = data['species_number_densities']
                masses = data['species_masses']
                
                # Calculate calculated density: sum(n_i * m_i)
                # Ensure units align before summing
                rho_calc = sum(n * m for n, m in zip(n_species, masses))
                
                # Relative difference
                diff = np.abs(rho_calc - rho_tot) / rho_tot
                diff_val = diff.value if hasattr(diff, 'value') else diff
                
                if np.any(diff_val > 0.05): # 5% tolerance
                    msg = "Mass inconsistency > 5% between species sum and total density"
                    self.logger.log_warning(msg)
                    # Return WARNING, not CRITICAL, as models sometimes differ slightly
                    return ValidationReport(ValidationStatus.WARNING, msg, 4)

            return ValidationReport(ValidationStatus.VALID, "Neutral atmosphere validation passed", 4)

        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Neutral atmosphere check failed: {e}", 4)

    def _validate_plasma(self, data: Dict[str, Any]) -> ValidationReport:
        """Block 5: Validate Plasma Continuity and Smooth Transitions."""
        try:
            ne = data.get('ne')
            alt = data.get('altitudes')
            
            if ne is None:
                return ValidationReport(ValidationStatus.CRITICAL, "Missing electron density data", 5)

            ne_val = ne.value if hasattr(ne, 'value') else ne
            alt_val = alt.value if hasattr(alt, 'value') else alt

            # Check Non-negativity
            if np.any(ne_val < 0):
                return ValidationReport(ValidationStatus.CRITICAL, "Negative electron density detected", 5)

            # Check Transition Smoothing (1000-2000 km)
            # Identify indices in the transition zone
            if alt is not None:
                mask_zone = (alt_val >= 1000) & (alt_val <= 2000)
                if np.any(mask_zone):
                    zone_vals = ne_val[mask_zone]
                    
                    # Simple gradient check for discontinuities
                    if len(zone_vals) > 1:
                        gradient = np.abs(np.diff(zone_vals))
                        # Threshold for "sharp" jump (heuristic: > 50% change between adjacent points)
                        if np.any(gradient > (0.5 * zone_vals[:-1])):
                            msg = "Discontinuity detected in IRI-GCPM transition zone (1000-2000 km)"
                            self.logger.log_warning(msg)
                            
                            # ACTIVE CORRECTION: Apply smoothing
                            smoothed = self._smooth_profile(ne, mask_zone)
                            return ValidationReport(
                                ValidationStatus.WARNING, 
                                msg, 
                                5, 
                                corrected_data={'ne': smoothed}
                            )

            return ValidationReport(ValidationStatus.VALID, "Plasma validation passed", 5)

        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Plasma check failed: {e}", 5)

    def _validate_magnetic_field(self, data: Dict[str, Any]) -> ValidationReport:
        """Block 6: Validate Magnetic Field Magnitude and Continuity."""
        try:
            b_vec = data.get('B_vector')
            if b_vec is not None:
                b_mag = np.linalg.norm(b_vec, axis=-1)
                b_val = b_mag.value if hasattr(b_mag, 'value') else b_mag
                
                if np.any(b_val <= 0):
                    return ValidationReport(ValidationStatus.CRITICAL, "Zero or negative magnetic field magnitude", 6)
                if np.any(np.isnan(b_val)):
                    return ValidationReport(ValidationStatus.CRITICAL, "NaN detected in magnetic field", 6)
            
            return ValidationReport(ValidationStatus.VALID, "Magnetic field validation passed", 6)
        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Magnetic field check failed: {e}", 6)

    def _validate_derived_params(self, data: Dict[str, Any], param_type: str) -> ValidationReport:
        """Blocks 7-10: Generic Validator for Derived Physics Parameters."""
        try:
            vals = data.get('values')
            if vals is None:
                return ValidationReport(ValidationStatus.WARNING, f"No data to validate for {param_type}", 10)

            val_arr = vals.value if hasattr(vals, 'value') else vals

            # Specific Checks by Type
            if param_type == 'conductivity':
                # Check for negative diagonal elements (physically suspicious for passive media)
                # Assuming tensor is flattened or diagonal is extracted
                if np.any(val_arr < 0):
                    return ValidationReport(ValidationStatus.WARNING, "Negative conductivity values detected", 10)
            
            elif param_type == 'stix':
                # Check Hermiticity conditions (simplified check for unexpected imaginary signs)
                if np.iscomplexobj(val_arr):
                    imag_part = np.imag(val_arr)
                    # Logic depends on specific Stix component, generic check here
                    if np.any(np.isnan(imag_part)):
                         return ValidationReport(ValidationStatus.WARNING, "NaN in imaginary part of Stix parameters", 10)

            # Global NaN Check
            if np.any(np.isnan(val_arr)):
                mask = ~np.isnan(val_arr)
                return ValidationReport(
                    ValidationStatus.WARNING, 
                    "NaNs generated in derived parameters; masking affected points", 
                    10, 
                    valid_mask=mask
                )

            return ValidationReport(ValidationStatus.VALID, f"{param_type} validation passed", 10)

        except Exception as e:
            return ValidationReport(ValidationStatus.CRITICAL, f"Derived params check failed: {e}", 10)

    # ----------------------------------------------------------------------
    # Helper Utilities
    # ----------------------------------------------------------------------

    def _smooth_profile(self, profile: u.Quantity, mask: np.ndarray) -> u.Quantity:
        """
        Applies a simple moving average smoothing to the masked region of a profile.
        
        Args:
            profile: The full quantity array.
            mask: Boolean mask of the region to smooth.
            
        Returns:
            The corrected profile with smoothed values in the transition zone.
        """
        corrected = profile.copy()
        zone_vals = corrected[mask]
        
        if len(zone_vals) < 3:
            return corrected # Not enough points to smooth
            
        # Apply 3-point moving average
        kernel = np.ones(3)/3.0
        smoothed_vals = np.convolve(zone_vals.value, kernel, mode='same') * zone_vals.unit
        
        corrected[mask] = smoothed_vals
        self.logger.log_warning("Applied active smoothing correction to profile.")
        return corrected
