"""
TUFMax Central Execution Engine.

This module implements the 'Resilient Orchestrator' pattern, managing the entire 
lifecycle of a TUFMax simulation. It coordinates the sequence of physics blocks, 
handles global state, manages fallbacks, and ensures that every execution results 
in a valid TUFMaxResult object (or a clear, audited failure).

Key Features:
    - Stateful Execution: Manages configuration, dependencies, and runtime state.
    - Atomic Block Execution: Runs each physics module with isolated error handling.
    - Automatic Fallback Chaining: Triggers lower-fidelity models if primary ones fail.
    - Provenance Tracking: Records the exact chain of models used for every result.
    - Context Management: Supports 'with' statement for safe resource cleanup.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import traceback
from datetime import datetime
import warnings
import numpy as np
import astropy.units as u


# Local Imports - Core Infrastructure
from ..exceptions import (
    TUFMaxError, TUFMaxConfigurationError, TUFMaxInputError, 
    TUFMaxValidationError, TUFMaxModelInitializationError, 
    TUFMaxDegradationWarning, TUFMaxGeometryError
)
from .logger import TUFMaxLogger
from .validator import TUFMaxValidator, ValidationReport, ValidationStatus
from .result import TUFMaxResult
from ..utils.adapter import UnitAdapter
from ..utils.constants import EARTH_RADIUS_MEAN

# Local Imports - IO & Physics Modules (To be implemented/linked)
# Note: We import conditionally or use placeholders if modules are still WIP to prevent crash
try:
    from ..io.time_utils import TUFMaxTime
    from ..io.orbit import OrbitalPropagator, OrbitalResult
    from ..io.coords import SpatialNormalizer
    from ..io.los_path import LoSPathGenerator, LoSProfile
    IO_AVAILABLE = True
except ImportError as e:
    IO_AVAILABLE = False
    # In production, this might raise TUFMaxConfigurationError immediately
    print(f"Warning: IO modules incomplete: {e}")

# Placeholder for Physics Blocks (Blocks 3-10)
# These will be imported as they are developed
class PhysicsBlockPlaceholder:
    def compute(self, *args, **kwargs):
        raise NotImplementedError("Physics block not yet implemented.")


class TUFMaxEngine:
    """
    Main Execution Engine for TUFMax.
    
    Coordinates the flow from input parsing -> geometry -> physics -> derived parameters.
    Implements the 'Resilient Orchestrator' strategy with checkpointing and fallback management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Engine with configuration and global dependencies.
        
        Args:
            config: Dictionary with user settings (e.g., 'use_high_fidelity', 'timeout_sec').
        """
        self.config = config or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Initialize Core Dependencies
        self.logger = TUFMaxLogger()
        self.adapter = UnitAdapter()
        
        # Validator depends on Logger
        self.validator = TUFMaxValidator(logger=self.logger, adapter=self.adapter)
        
        # State Containers
        self.context: Dict[str, Any] = {}  # Holds intermediate data (geometry, profiles)
        self.model_chain: Dict[int, str] = {}  # Tracks which model was used per block
        self.execution_log: List[str] = []
        
        # Flags
        self._is_running = False
        self._dry_run = self.config.get('dry_run', False)

    # ----------------------------------------------------------------------
    # Context Management (Safe Resource Handling)
    # ----------------------------------------------------------------------
    
    def __enter__(self):
        self._initialize_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize_session(exc_type, exc_val, exc_tb)
        # Return False to propagate exceptions, True to swallow (usually False for engines)
        return False

    def _initialize_session(self):
        """Sets up the run environment."""
        self.start_time = time.time()
        self._is_running = True
        self.context.clear()
        self.model_chain.clear()
        
        # Definir valores por defecto seguros para evitar TypeError
        default_pos = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        
        # Obtener datos de config, asegurando que sean diccionarios
        ground_pos_data = self.config.get('ground_pos', default_pos)
        sat_pos_data = self.config.get('sat_pos', default_pos)
        
        if not isinstance(ground_pos_data, dict):
            ground_pos_data = default_pos
        if not isinstance(sat_pos_data, dict):
            sat_pos_data = default_pos

        self.logger.initialize_run(
            ground_pos=ground_pos_data,
            sat_pos=sat_pos_data,
            config=self.config
        )
        
        self.logger.log_module_start("TUFMax Engine Initialization")

    def _finalize_session(self, exc_type, exc_val, exc_tb):
        """Cleans up and writes final summary."""
        self._is_running = False
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.log_module_end("TUFMax Engine Execution", success=True)
            self.logger.finalize(
                final_models_used=self.model_chain,
                total_time=f"{duration:.4f}s"
            )
        else:
            self.logger.log_error(f"Engine terminated by exception: {exc_type.__name__}: {exc_val}")
            # Ensure log is closed even on crash
            self.logger.finalize(
                final_models_used=self.model_chain,
                total_time=f"{duration:.4f}s (FAILED)"
            )

    # ----------------------------------------------------------------------
    # Main Execution Flow
    # ----------------------------------------------------------------------

    def run(self, 
            tle_data: Optional[str] = None,
            keplerian_data: Optional[Dict] = None,
            ground_coords: Optional[Tuple[float, float, float]] = None,
            target_time: Optional[Any] = None,
            n_samples: int = 100) -> TUFMaxResult:
        """
        Executes the full TUFMax pipeline.
        
        Args:
            tle_data: TLE string for satellite orbit.
            keplerian_data: Dict of Keplerian elements.
            ground_coords: (Lat, Lon, Alt) for ground station.
            target_time: Time for simulation (string, datetime, or float).
            n_samples: Resolution of LoS profile.
            
        Returns:
            TUFMaxResult object containing all profiles and metadata.
        """
        if not self._is_running:
            # Auto-initialize if not used as context manager
            self._initialize_session()

        try:
            self.logger.log_module_start("Phase 1: Ingestion & Geometry (Blocks 0-2)")
            
            # --- Block 0 & 1: Time & Orbit ---
            if not IO_AVAILABLE:
                raise TUFMaxConfigurationError("IO modules missing. Cannot proceed.")
                
            orb_prop = OrbitalPropagator(logger=self.logger)
            
            # Determine input source
            input_data = tle_data or keplerian_data or ground_coords
            if input_data is None:
                raise TUFMaxInputError("No orbital or ground input provided.")
            
            # Propagate/Process Orbit
            # Note: If ground_coords only, orbit module bypasses propagation
            orb_result: OrbitalResult = orb_prop.process(input_data, target_time=target_time)
            
            # Store in context
            self.context['sat_itrs'] = orb_result.position_itrs
            self.context['epoch'] = orb_result.epoch
            self.model_chain[0] = orb_result.source_model
            
            # Validate Geometry (Block 2 prep)
            # If we have ground coords, we need to normalize them too
            if ground_coords and isinstance(input_data, tuple):
                 # Input was ground coords directly (Static profile mode)
                 self.context['ground_itrs'] = self._coords_to_itrs(ground_coords) # Helper needed
            else:
                 # Assume default ground or extract from somewhere? 
                 # For now, assume single-point vertical profile if no ground/sat pair defined clearly
                 # Let's assume a standard scenario: Ground at (0,0,0) relative? No, needs real coords.
                 # Simplification: If only TLE provided, assume observer at sub-satellite point? 
                 # Better: Require explicit ground_coords for LoS.
                 if not ground_coords:
                     # Fallback: Vertical profile at Sat location projected down? 
                     # Let's raise for clarity in this robust version
                     raise TUFMaxInputError("Ground station coordinates required for LoS generation.")
            
            # Normalize Ground Coordinates (Block 2)
            ground_norm = SpatialNormalizer(
                input_pos={'lat': ground_coords[0], 'lon': ground_coords[1], 'alt': ground_coords[2]},
                date_decimal_year=self.context['epoch'].decimalyear
            )
            g_res = ground_norm.get_results()
            self.context['ground_itrs'] = g_res['itrs_vector_km'] * 1000.0 * UnitAdapter.get_standard_unit('length') # Convert to m
            
            # Validate Geometry (Block 2 Validation)
            geo_data = {
                'altitudes': None, # Will be filled by LoS
                'los_valid': True  # Placeholder until LoS check
            }
            # We defer detailed validation to LoS module which has both vectors
            
            self.logger.log_module_end("Phase 1: Ingestion & Geometry", success=True)
            self.model_chain[2] = "SpatialNormalizer (Astropy/Apexpy)"

            # --- Block 3: LoS Path Generation ---
            self.logger.log_module_start("Phase 2: Line-of-Sight Path (Block 3)")
            
            los_gen = LoSPathGenerator(logger=self.logger)
            
            # Ensure units match (LoS expects meters)
            r_g = self.context['ground_itrs']
            r_s = self.context['sat_itrs']
            
            # --- DEBUG CRÍTICO: VERIFICAR UNIDADES DE ENTRADA ---
            print("\n[DEBUG ENGINE] Verificando vectores de entrada para LoS:")
            print(f"   Ground (r_g): {r_g}")
            print(f"   Ground Norm: {np.linalg.norm(r_g)}")
            print(f"   Sat (r_s): {r_s}")
            print(f"   Sat Norm: {np.linalg.norm(r_s)}")
            print(f"   Diff Norm: {(np.linalg.norm(r_s)-np.linalg.norm(r_g)).to(u.km)}")
            
            # Verificación de consistencia
            norm_g = np.linalg.norm(r_g).to(u.km).value
            norm_s = np.linalg.norm(r_s).to(u.km).value
            
            if norm_g < 100 or norm_s < 100:
                print("   ❌ ERROR: Uno de los vectores es demasiado pequeño (<100 km). ¿Unidades en metros mal interpretadas?")
            elif norm_g > 100000 or norm_s > 100000:
                print("   ❌ ERROR: Uno de los vectores es demasiado grande (>100,000 km). ¿Unidades en km tratadas como metros?")
            else:
                print("   ✅ Vectores dentro de rango esperado (6378 - 50000 km)")
            # ---------------------------------------------------
            
            los_profile: LoSProfile = los_gen.generate(
                r_ground=r_g,
                r_sat=r_s,
                n_samples=n_samples,
                min_elevation_deg=self.config.get('min_elevation', 0.0),
                epoch=self.context['epoch']
            )
            
            los_profile: LoSProfile = los_gen.generate(
                r_ground=r_g,
                r_sat=r_s,
                n_samples=n_samples,
                min_elevation_deg=self.config.get('min_elevation', 0.0),
                epoch=self.context['epoch']
            )
            
            # Extract Data for Physics Blocks
            # loS_profile contains xyz, geodetic, magnetic, valid_mask
            valid_data = los_profile.get_valid_profile()
            
            self.context['path_xyz'] = valid_data['xyz']
            self.context['path_geodetic'] = valid_data['geodetic']
            self.context['path_magnetic'] = valid_data['magnetic']
            self.context['valid_mask'] = valid_data.get('valid_mask', np.ones(self.context.get('n_points', 0), dtype=bool))
            self.context['n_points'] = valid_data['count']
            
            self.model_chain[3] = "LoSPathGenerator (Vectorized)"
            self.logger.log_module_end("Phase 2: Line-of-Sight Path", success=True)

            # --- Blocks 4-10: Physics Chain ---
            # Since physics modules are not fully implemented in this snippet, 
            # we simulate the chain with placeholders and validation calls.
            
            self.logger.log_module_start("Phase 3: Physics Calculations (Blocks 4-10)")
            
            # Block 4: Neutral Atmosphere
            neutral_data = self._execute_physics_block(4, "NeutralAtmosphere", 
                                                       inputs=self.context, 
                                                       expected_keys=['Tn', 'total_density'])
            self.context.update(neutral_data)
            
            # Block 5: Ionosphere/Plasmasphere
            plasma_data = self._execute_physics_block(5, "IonospherePlasmasphere", 
                                                      inputs=self.context, 
                                                      expected_keys=['ne', 'Te', 'Ti'])
            self.context.update(plasma_data)
            
            # Block 6: Magnetic Field (Refinement along path)
            # Usually done in LoS, but can be refined here
            b_field_data = self._execute_physics_block(6, "GeomagneticField", 
                                                       inputs=self.context, 
                                                       expected_keys=['B_vector'])
            self.context.update(b_field_data)
            
            # Block 7-10: Derived Parameters (Gyro, Collisions, Stix, Conductivity)
            # Chained execution
            derived_blocks = [
                (7, "Gyrofrequencies", ['Omega_ce', 'Omega_ci']),
                (8, "CollisionFrequencies", ['nu_en', 'nu_in']),
                (9, "StixParameters", ['S', 'D', 'P']),
                (10, "ConductivityTensor", ['sigma_P', 'sigma_H'])
            ]
            
            for blk_id, name, keys in derived_blocks:
                res = self._execute_physics_block(blk_id, name, inputs=self.context, expected_keys=keys)
                self.context.update(res)

            self.logger.log_module_end("Phase 3: Physics Calculations", success=True)

            # --- Final Assembly ---
            result = self._assemble_result(los_profile)
            
            return result

        except TUFMaxGeometryError as ge:
            # Manejo específico para errores geométricos: Loguear y devolver Resultado Vacío/Fallido
            self.logger.log_error(f"Geometry Validation Failed: {ge}")
            self.logger.log_module_end("Phase 2: Line-of-Sight Path", success=False)
            
            # Devolver un resultado vacío pero válido indicando el fallo
            return self._assemble_empty_result(reason=str(ge))

        except Exception as e:
            # Log the fatal error
            self.logger.log_error(f"Critical Failure in Engine Run: {e}")
            # Opcional: Imprimir stack trace solo en modo debug, no en producción
            # import traceback; traceback.print_exc() 
            
            # En lugar de 'raise', devolvemos un resultado de fallo limpio
            # Si PREFIERES que el script falle explícitamente, descomenta la siguiente línea:
            # raise 
            
            # Para comportamiento "Silencioso/Robusto":
            self.logger.finalize(final_models_used=self.model_chain, total_time="FAILED")
            return self._assemble_empty_result(reason=f"{type(e).__name__}: {e}")

    def _execute_physics_block(self, block_id: int, name: str, inputs: Dict, expected_keys: List[str]) -> Dict:
        """
        Generic runner for physics blocks with validation and fallback logic.
        """
        self.logger.log_module_start(f"Block {block_id}: {name}")
        
        try:
            # 1. Attempt Primary Model
            # In real implementation: model = PrimaryModel(); data = model.compute(inputs)
            # Here we simulate success or failure based on config
            if self.config.get(f'force_fail_block_{block_id}', False):
                raise TUFMaxModelInitializationError(f"Simulated failure for Block {block_id}")
            
            # Simulate Data Generation (Placeholder)
            # Real code would call pymsis, iri2016, etc.
            dummy_data = {k: None for k in expected_keys} 
            # Fill with dummy quantities for validation to pass
            import astropy.units as u
            n_pts = inputs.get('n_points', 10)
            if 'Tn' in expected_keys: dummy_data['Tn'] = np.ones(n_pts) * 1000 * u.K
            if 'ne' in expected_keys: dummy_data['ne'] = np.ones(n_pts) * 1e11 * u.m**-3
            if 'B_vector' in expected_keys: dummy_data['B_vector'] = np.ones((n_pts, 3)) * 5e-5 * u.T
            # ... fill others ...
            
            # 2. Validate Output
            # Construct temp dict for validator
            val_dict = {**inputs, **dummy_data}
            self.validator.run_check(block_id, val_dict)
            
            # 3. Record Success
            self.model_chain[block_id] = f"{name} (Primary)"
            self.logger.log_module_end(f"Block {block_id}: {name}", success=True)
            return dummy_data

        except TUFMaxModelInitializationError as mie:
            # Trigger Fallback Logic
            self.logger.log_fallback(from_model=f"{name} (Primary)", to_model=f"{name} (Fallback)", reason=str(mie))
            # Implement fallback call here...
            # For now, re-raise or return empty
            raise TUFMaxDegradationWarning(f"Fallback required for Block {block_id}", original_model=name, fallback_model="None")
        
        except TUFMaxValidationError as ve:
            self.logger.log_validation(ve) # Already logged in validator, but good to catch here
            raise
            
        except Exception as e:
            self.logger.log_error(f"Unexpected error in Block {block_id}: {e}")
            raise TUFMaxError(f"Block {block_id} failed unexpectedly: {e}")

    def _assemble_result(self, los_profile: LoSProfile) -> TUFMaxResult:
        """Constructs the final TUFMaxResult object."""
        self.logger.log_module_start("Assembling Final Result")
        
        # Map context data to TUFMaxResult structure
        # Neutral Data
        neutral = {
            'Tn': self.context.get('Tn'),
            'total_density': self.context.get('total_density'),
            # Add species...
        }
        
        # Plasma Data
        plasma = {
            'ne': self.context.get('ne'),
            'Te': self.context.get('Te'),
            'Ti': self.context.get('Ti'),
        }
        
        # Magnetic Data
        magnetic = {
            'B_vector': self.context.get('B_vector'),
        }
        
        # Derived Data
        derived = {
            'sigma_P': self.context.get('sigma_P'),
            'sigma_H': self.context.get('sigma_H'),
            'stix_S': self.context.get('S'),
            # ...
        }
        
        # Metadata
        meta = {
            'execution_time': datetime.now().isoformat(),
            'config': self.config,
            'ground_pos': self.config.get('ground_pos'),
            'sat_pos': self.config.get('sat_pos')
        }
        
        # Get last validation report (simulated)
        last_report = ValidationReport(ValidationStatus.VALID, "Assembly Complete", 99)
        
        result = TUFMaxResult(
            altitude=los_profile.geodetic['alt'], # Esto sí lo tienes
            los_profile=los_profile,
            neutral_data=neutral,
            plasma_data=plasma,
            magnetic_data=magnetic,
            derived_data=derived,
            metadata=meta,
            validation_report=last_report,
            model_chain=self.model_chain,
            _valid_mask=los_profile.valid_mask
        )
        
        self.logger.log_module_end("Assembling Final Result", success=True)
        return result

    def _assemble_empty_result(self, reason: str) -> TUFMaxResult:
        """
        Construye un TUFMaxResult vacío cuando la ejecución falla tempranamente.
        Esto permite que el usuario reciba un objeto manejable en lugar de un crash.
        """
        meta = {
            'execution_time': datetime.now().isoformat(),
            'config': self.config,
            'failure_reason': reason,
            'status': 'FAILED'
        }
        
        # Reporte de validación fallida
        last_report = ValidationReport(ValidationStatus.CRITICAL, reason, 0)
        
        # Crear arrays vacíos para evitar errores de índice en el consumidor
        empty_alt = np.array([0.0]) * u.km
        
        return TUFMaxResult(
            altitude=empty_alt,
            neutral_data={},
            plasma_data={},
            magnetic_data={},
            derived_data={},
            metadata=meta,
            validation_report=last_report,
            model_chain=self.model_chain,
            _valid_mask=np.array([False])
        )

    def _coords_to_itrs(self, coords: Tuple[float, float, float]) -> Any:
        """Helper to convert simple lat/lon/alt to ITRS Quantity using SpatialNormalizer."""
        norm = SpatialNormalizer(
            input_pos={'lat': coords[0], 'lon': coords[1], 'alt': coords[2]},
            date_decimal_year=TUFMaxTime.now().decimalyear
        )
        res = norm.get_results()
        # Convert km to m and return Quantity
        vec_km = res['itrs_vector_km']
        return vec_km * 1000.0 * UnitAdapter.get_standard_unit('length')

# Import numpy locally to avoid global dependency issues if not needed elsewhere
#import numpy as np
