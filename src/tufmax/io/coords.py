"""
TUFMax Spatial Normalization Module (Module 2).
Corrección sistemática de errores de coordenadas y fallbacks.
"""

from __future__ import annotations
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import warnings
import datetime

# Local imports
from ..exceptions import TUFMaxGeometryError, TUFMaxDegradationWarning
from ..utils.constants import EARTH_RADIUS_EQ, EARTH_FLATTENING

# Type aliases
GeodeticInput = Dict[str, float]  # {'lat': deg, 'lon': deg, 'alt': km}
CartesianInput = Union[Tuple[float, float, float], list, np.ndarray]  # [x, y, z] in km
InputType = Union[GeodeticInput, CartesianInput]


class SpatialNormalizer:
    """
    Hierarchical Spatial Normalization Engine.
    """

    def __init__(self, input_pos: InputType, date_decimal_year: float):
        self.date_decimal_year = date_decimal_year
        self.itrs_vector: Optional[np.ndarray] = None
        self.geodetic: Dict[str, float] = {}
        self.magnetic_coords: Dict[str, float] = {
            'MagLat': np.nan, 
            'MLT': np.nan, 
            'ApexHeight': np.nan
        }
        self._original_input = input_pos
        
        if not self._execute_fallback_chain():
            raise TUFMaxGeometryError("All spatial normalization backends failed.")

    def _execute_fallback_chain(self) -> bool:
        if self._attempt_phase_1_astropy():
            return True
        if self._attempt_phase_2_spacepy():
            return True
        if self._attempt_phase_3_analytical():
            return True
        return False

    def _validate_altitude_sanity(self, alt_km: float):
        if alt_km < -float(EARTH_RADIUS_EQ.to('km').value):
            raise TUFMaxGeometryError(
                f"Altitude {alt_km:.2f} km implies position inside Earth's core."
            )

    # ----------------------------------------------------------------------
    # Phase 1: Astropy + Apexpy (CORREGIDO)
    # ----------------------------------------------------------------------
    
    def _attempt_phase_1_astropy(self) -> bool:
        try:
            from astropy.coordinates import EarthLocation, CartesianRepresentation
            from astropy import units as u
            import apexpy

            # Manejo correcto de entrada Geodética
            if isinstance(self._original_input, dict):
                lat = self._original_input['lat'] * u.deg
                lon = self._original_input['lon'] * u.deg
                alt = self._original_input['alt'] * u.km
                
                loc = EarthLocation(lat=lat, lon=lon, height=alt)
                
                self.itrs_vector = np.array([
                    loc.x.to(u.km).value,
                    loc.y.to(u.km).value,
                    loc.z.to(u.km).value
                ])
                
                self.geodetic = {
                    'lat': loc.lat.to(u.deg).value,
                    'lon': loc.lon.to(u.deg).value,
                    'alt': loc.height.to(u.km).value
                }
            
            # Manejo CORREGIDO de entrada Cartesiana (x, y, z en km)
            elif isinstance(self._original_input, (list, tuple, np.ndarray)):
                x, y, z = self._original_input
                # Crear representación cartesiana explícita con unidades
                cart_rep = CartesianRepresentation(x*u.km, y*u.km, z*u.km)
                # Convertir a EarthLocation usando from_geocentric
                loc = EarthLocation.from_geocentric(x*u.km, y*u.km, z*u.km)
                
                self.itrs_vector = np.array([x, y, z])
                self.geodetic = {
                    'lat': loc.lat.to(u.deg).value,
                    'lon': loc.lon.to(u.deg).value,
                    'alt': loc.height.to(u.km).value
                }
            else:
                return False

            self._validate_altitude_sanity(self.geodetic['alt'])

            # Cálculo de coordenadas magnéticas con Apexpy
            apex_obj = apexpy.Apex(date=self.date_decimal_year)
            
            # geo2apex devuelve (apex_lat, apex_lon)
            mag_lat, mag_lon = apex_obj.geo2apex(
                self.geodetic['lat'],
                self.geodetic['lon'],
                self.geodetic['alt']
            )
            
            # Calcular MLT
            dt_obj = self._decimal_year_to_datetime(self.date_decimal_year)
            mlt = apex_obj.mlon2mlt(mag_lon, dt_obj)
            
            self.magnetic_coords = {
                'MagLat': float(mag_lat),
                'MLT': float(mlt),
                'ApexHeight': float(self.geodetic['alt'])
            }
            
            return True

        except ImportError:
            return False
        except Exception as e:
            warnings.warn(f"Astropy/Apexpy phase failed: {e}. Trying fallback...", TUFMaxDegradationWarning)
            return False

    # ----------------------------------------------------------------------
    # Phase 2: SpacePy (CORREGIDO)
    # ----------------------------------------------------------------------

    def _attempt_phase_2_spacepy(self) -> bool:
        try:
            import spacepy.coordinates as spc
            from spacepy.time import Ticktock
            
            # Si la entrada es diccionario (Geodética), es directo
            if isinstance(self._original_input, dict):
                vals = np.array([[
                    self._original_input['lat'],
                    self._original_input['lon'],
                    self._original_input['alt']
                ]])
                cvals = spc.Coords(vals, 'GDZ', 'sph')
            
            # Si es cartesiana, SpacePy necesita GDZ directamente o una conversión previa
            # Como SpacePy no convierte ITRS->GDZ fácilmente sin ticks, usamos analítica rápida aquí
            elif isinstance(self._original_input, (list, tuple, np.ndarray)):
                x, y, z = self._original_input
                # Conversión analítica rápida ECEF -> GDZ para alimentar a SpacePy
                r_xy = np.sqrt(x**2 + y**2)
                r = np.sqrt(r_xy**2 + z**2)
                if r < 100: return False
                
                a = float(EARTH_RADIUS_EQ.to('km').value)
                f = float(EARTH_FLATTENING.value)
                e_sq = 2*f - f**2
                
                phi = np.arctan(z / (r_xy * (1 - e_sq)))
                for _ in range(5):
                    N = a / np.sqrt(1 - e_sq * np.sin(phi)**2)
                    h = r_xy / np.cos(phi) - N
                    phi_new = np.arctan(z / (r_xy * (1 - e_sq * N / (N + h))))
                    if abs(phi_new - phi) < 1e-9: break
                    phi = phi_new
                
                lat = np.degrees(phi)
                lon = np.degrees(np.arctan2(y, x))
                h_km = r_xy / np.cos(phi) - N
                
                vals = np.array([[lat, lon, h_km]])
                cvals = spc.Coords(vals, 'GDZ', 'sph')
            else:
                return False

            dt_obj = self._decimal_year_to_datetime(self.date_decimal_year)
            ticks = Ticktock([dt_obj.isoformat()], 'ISO')
            cvals.ticks = ticks

            # Obtener vector ITRS desde SpacePy
            itrs_car = cvals.convert('ITRS', 'car')
            self.itrs_vector = itrs_car.data[0]
            
            # Obtener geodésicas confirmadas
            geo_sph = cvals.convert('GEO', 'sph') # SpacePy GEO es similar a GDZ en esfera
            # Mejor usar los datos originales si vinieron de GDZ directo
            if isinstance(self._original_input, dict):
                self.geodetic = {
                    'lat': self._original_input['lat'],
                    'lon': self._original_input['lon'],
                    'alt': self._original_input['alt']
                }
            else:
                self.geodetic = {
                    'lat': cvals.data[0][0],
                    'lon': cvals.data[0][1],
                    'alt': cvals.data[0][2]
                }
            
            self._validate_altitude_sanity(self.geodetic['alt'])
            
            # Cálculo MLT manual robusto
            jd = ticks.jd[0]
            T = (jd - 2451545.0) / 36525.0
            gmst_sec = (24110.54841 + 8640184.812866*T + 0.093104*T**2) % 86400.0
            gmst_deg = (gmst_sec / 86400.0) * 360.0
            mlt = ((self.geodetic['lon'] + gmst_deg) / 15.0) % 24.0
            
            self.magnetic_coords = {
                'MagLat': np.nan,
                'MLT': float(mlt),
                'ApexHeight': np.nan
            }
            
            return True

        except ImportError:
            return False
        except Exception as e:
            warnings.warn(f"SpacePy phase failed: {e}. Trying analytical fallback...", TUFMaxDegradationWarning)
            return False

    # ----------------------------------------------------------------------
    # Phase 3: Analytical WGS84 (CORREGIDO Y COMPLETO)
    # ----------------------------------------------------------------------

    def _attempt_phase_3_analytical(self) -> bool:
        try:
            # Caso A: Entrada Geodética -> ITRS
            if isinstance(self._original_input, dict):
                phi_deg = self._original_input['lat']
                lam_deg = self._original_input['lon']
                h_km = self._original_input['alt']
                
                self._validate_altitude_sanity(h_km)
                
                a = float(EARTH_RADIUS_EQ.to('km').value)
                f = float(EARTH_FLATTENING.value)
                e_sq = 2*f - f**2
                
                phi_rad = np.radians(phi_deg)
                lam_rad = np.radians(lam_deg)
                
                N = a / np.sqrt(1 - e_sq * np.sin(phi_rad)**2)
                
                x = (N + h_km) * np.cos(phi_rad) * np.cos(lam_rad)
                y = (N + h_km) * np.cos(phi_rad) * np.sin(lam_rad)
                z = (N * (1 - e_sq) + h_km) * np.sin(phi_rad)
                
                self.itrs_vector = np.array([x, y, z])
                self.geodetic = {'lat': phi_deg, 'lon': lam_deg, 'alt': h_km}
                self.magnetic_coords = {'MagLat': np.nan, 'MLT': np.nan, 'ApexHeight': np.nan}
                return True

            # Caso B: Entrada Cartesiana -> Geodética (Iterativo)
            elif isinstance(self._original_input, (list, tuple, np.ndarray)):
                x, y, z = self._original_input
                r_xy = np.sqrt(x**2 + y**2)
                r = np.sqrt(r_xy**2 + z**2)
                
                if r < 100: return False

                a = float(EARTH_RADIUS_EQ.to('km').value)
                f = float(EARTH_FLATTENING.value)
                e_sq = 2*f - f**2
                
                phi = np.arctan(z / (r_xy * (1 - e_sq)))
                
                for _ in range(5):
                    N = a / np.sqrt(1 - e_sq * np.sin(phi)**2)
                    h = r_xy / np.cos(phi) - N
                    phi_new = np.arctan(z / (r_xy * (1 - e_sq * N / (N + h))))
                    if abs(phi_new - phi) < 1e-9: break
                    phi = phi_new
                
                phi_deg = np.degrees(phi)
                lam_deg = np.degrees(np.arctan2(y, x))
                h_km = r_xy / np.cos(phi) - N
                
                self._validate_altitude_sanity(h_km)
                
                self.itrs_vector = np.array([x, y, z])
                self.geodetic = {'lat': phi_deg, 'lon': lam_deg, 'alt': h_km}
                self.magnetic_coords = {'MagLat': np.nan, 'MLT': np.nan, 'ApexHeight': np.nan}
                
                return True
            
            return False

        except Exception:
            return False

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _decimal_year_to_datetime(self, dec_year: float) -> datetime.datetime:
        year = int(dec_year)
        day_frac = dec_year - year
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        day_of_year = day_frac * (days_in_year - 1)
        
        start_of_year = datetime.datetime(year, 1, 1)
        return start_of_year + datetime.timedelta(days=day_of_year)

    def get_results(self) -> Dict[str, Any]:
        return {
            'itrs_vector_km': self.itrs_vector,
            'geodetic': self.geodetic,
            'magnetic': self.magnetic_coords
        }
