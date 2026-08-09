#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruebitas.py - Archivo de pruebas e investigación preliminar para TUFMax

Librería LaTeX para cálculo de parámetros de las ecuaciones de Maxwell
en la atmósfera terrestre (0 km - Órbita Geoestacionaria ~36,000 km)

Documentación recopilada de herramientas para modelado de parámetros ambientales:
- Índices solares: F10.7, S10, M10, Y10, F30
- Índices geomagnéticos: Ap, Kp, Dst
"""

# =============================================================================
# HERRAMIENTAS PARA MODELADO DE PARÁMETROS AMBIENTALES
# =============================================================================
# Ordenadas de mayor a menor precisión térmica según documentación oficial

# -----------------------------------------------------------------------------
# 1. PYATMOS / SET (Máxima Precisión Térmica)
# -----------------------------------------------------------------------------
"""
PYATMOS - Package for Atmospheric Density and Temperature Models
Versión instalada: 1.2.7
Autor: Chunxiao Li
Repositorio: https://github.com/lcx366/ATMOS

MODELOS DISPONIBLES EN PYATMOS:
================================

1.1 NRLMSISE-00 (NRL Mass Spectrometer and Incoherent Scatter Radar Extended)
------------------------------------------------------------------------------
Rango de altitudes: 0 - 2000 km (nivel del mar hasta exósfera)
Tipo: Semi-empírico, global
Uso principal: Predicción de decaimiento orbital por arrastre atmosférico

PARÁMETROS DE ENTRADA:
- t (str): Tiempo en formato UTC, ej. '2020-07-22 22:18:45'
- location (tuple/list): [latitud, longitud, altitud] en [grados, grados, km]
- swdata (dict): Datos de clima espacial (F10.7, Ap) leídos de archivo SW-All.csv
- aphmode (bool, opcional, default=True): Usar índice geomagnético de 3h

PARÁMETROS DE SALIDA (objeto ATMOS):
- rho (float): Densidad de masa total [kg/m^3]
- T (tuple): Temperatura local [K]
- nd (dict): Densidad numérica de componentes [1/m^3]:
    * He (Helio)
    * O (Oxígeno atómico)
    * O2 (Oxígeno molecular)
    * N (Nitrógeno atómico)
    * N2 (Nitrógeno molecular)
    * Ar (Argón)
    * H (Hidrógeno)
    * ANM O (Oxígeno anómalo)

EJEMPLO DE USO:
```python
from pyatmos import nrlmsise00, download_sw_nrlmsise00, read_sw_nrlmsise00

# Descargar datos de clima espacial
swfile = download_sw_nrlmsise00()  # Descarga de www.celestrak.com
swdata = read_sw_nrlmsise00(swfile)

# Configurar tiempo y ubicación
t = '2020-07-22 22:18:45'
lat, lon, alt = 25, 102, 600  # grados, grados, km

# Ejecutar modelo
nrl00 = nrlmsise00(t, (lat, lon, alt), swdata)

# Acceder a resultados
print(f"Densidad: {nrl00.rho} kg/m^3")
print(f"Temperatura: {nrl00.T} K")
print(f"Especies: {nrl00.nd}")  # dict con 8 especies
```

NOTAS:
- Requiere conexión a internet para descargar archivo SW-All.csv la primera vez
- El archivo se cachea localmente para uso posterior
- Índice F10.7 diario y promedio de 3 meses
- Índice Ap de 3 horas

--------------------------------------------------------------------------------

1.2 JB2008 (Jacchia-Bowman 2008)
------------------------------------------------------------------------------
Rango de altitudes: 90 - 2500 km
Tipo: Empírico, global
Uso principal: Predicción de decaimiento orbital con alta precisión durante
             tormentas geomagnéticas agudas

CARACTERÍSTICAS ESPECIALES:
- Vincula longitudes de onda específicas a profundidades ópticas
- Superior precisión durante tormentas geomagnéticas agudas
- Utiliza índices solares SET (Space Environment Technologies)

PARÁMETROS DE ENTRADA:
- t (str): Tiempo en formato UTC
- location (tuple/list): [latitud, longitud, altitud] en [grados, grados, km]
- swdata (dict): Datos de clima espacial SET incluyendo:
    * F10.7 (índice solar radio)
    * S10 (índice UV Mg II)
    * M10 (índice UV medio)
    * Y10 (índice UV lejano)
    * Ap (índice geomagnético)
    * Dst (convertido a dTc)

PARÁMETROS DE SALIDA (objeto ATMOS):
- rho (float): Densidad de masa total [kg/m^3]
- T (tuple): Temperatura local [K]

NOTA: JB2008 NO proporciona composición química detallada, solo densidad total y temperatura

EJEMPLO DE USO:
```python
from pyatmos import jb2008, download_sw_jb2008, read_sw_jb2008

# Descargar datos de clima espacial SET
swfile = download_sw_jb2008()  # Descarga de sol.spacenvironment.net
swdata = read_sw_jb2008(swfile)

# Configurar tiempo y ubicación
t = '2020-07-22 22:18:45'
lat, lon, alt = 25, 102, 600

# Ejecutar modelo
jb08 = jb2008(t, (lat, lon, alt), swdata)

print(f"Densidad: {jb08.rho} kg/m^3")
print(f"Temperatura: {jb08.T} K")
```

ARCHIVOS REQUERIDOS:
- SOLFSMY.TXT: Datos solares diarios
- DTCFILE.TXT: Datos de temperatura y correcciones

--------------------------------------------------------------------------------

1.3 COESA76 (U.S. Committee on Extension to the Standard Atmosphere 1976)
------------------------------------------------------------------------------
Rango de altitudes: -0.611 - 1000 km (geométricas)
Tipo: Estándar atmosférico estático (no varía con tiempo/actividad solar)
Uso: Referencia estándar, condiciones promedio

PARÁMETROS DE ENTRADA:
- alts (list/array): Lista de altitudes geométricas o geopotenciales [km]
- alt_type (str, opcional): 'geometric' (default) o 'geopotential'

PARÁMETROS DE SALIDA (objeto ATMOS):
- rho (array): Densidades [kg/m^3]
- T (array): Temperaturas [K]
- P (array): Presiones [Pa]

EJEMPLO DE USO:
```python
from pyatmos import coesa76

alts = [0, 100, 500, 1000]  # km
result = coesa76(alts)

print(f"Densidades: {result.rho} kg/m^3")
print(f"Temperaturas: {result.T} K")
print(f"Presiones: {result.P} Pa")
```

NOTAS:
- No requiere índices solares ni geomagnéticos
- Modelo estático basado en promedios climáticos
- Para altitudes fuera del rango, extrapola valores

--------------------------------------------------------------------------------

RESUMEN DE MODELOS PYATMOS:
===========================
| Modelo      | Rango (km) | Especies | Índices Solares | Índices Geo | Uso Principal          |
|-------------|------------|----------|-----------------|-------------|------------------------|
| NRLMSISE-00 | 0-2000     | 8        | F10.7           | Ap (3h)     | General, composición   |
| JB2008      | 90-2500    | 0*       | F10.7,S10,M10,Y10| Ap, Dst    | Tormentas, precisión   |
| COESA76     | -0.6-1000  | 0        | Ninguno         | Ninguno     | Estándar estático      |

* JB2008 no devuelve especies individuales, solo densidad total
"""

# -----------------------------------------------------------------------------
# 2. SPACEPY (Estándar para Investigación de Gran Altitud)
# -----------------------------------------------------------------------------
"""
SPACEPY - Python-based Space Physics Library
Versión instalada: 0.7.0
Institución: Los Alamos National Laboratory
Documentación: https://spacepy.github.io/

MÓDULO PRINCIPAL: spacepy.omni
================================
Propósito: Acceder a datos validados de NASA/GSFC OMNIWeb
Rango de validez: 100 km - Órbita Geoestacionaria (>36,000 km)
Uso: Análisis de misiones, continuidad en plasma térmico (H+, He+, O+)

PARÁMETROS DE ENTRADA:
- ticks (Ticktock o array de datetime): Tiempos para los cuales se requieren datos
- dbase (str, opcional): Base de datos a utilizar
    * 'QDhourly' (default): Qin-Denton hourly data
    * 'OMNI2hourly': OMNI2 hourly data
    * 'Mergedhourly': Merged hourly data

PARÁMETROS DE SALIDA (SpaceData - tipo diccionario):
ÍNDICES GEOMAGNÉTICOS:
- Kp: Índice planetario de 3 horas (0-9)
- Dst: Índice de tormenta de anillo [nT]
- akp3: Índice Kp convertido a escala lineal

ÍNDICES SOLARES/VIENTO SOLAR:
- F10.7: Índice de radio solar (disponible en algunas bases)
- dens: Densidad del viento solar [partículas/cm^3]
- velo: Velocidad del viento solar [km/s]
- Pdyn: Presión dinámica del viento solar [nPa]
- ByIMF, BzIMF: Componentes del campo magnético interplanetario [nT]
- Bz1-Bz6, W1-W6: Componentes adicionales del IMF

OTROS:
- G1, G2, G3: Escalas de tormentas geomagnéticas
- Year, DOY, Hr: Información temporal
- Qbits: Indicador de calidad de datos (0=promedio, 2=medido)

EJEMPLO DE USO:
```python
import spacepy.time as spt
import spacepy.omni as om

# Crear objeto Ticktock con tiempos deseados
ticks = spt.Ticktock(['2020-02-02T12:00:00', '2020-02-02T12:10:00'], 'ISO')

# Obtener datos OMNI
d = om.get_omni(ticks, dbase='QDhourly')

# Acceder a índices específicos
print(f"Kp: {d['Kp']}")
print(f"Dst: {d['Dst']} nT")
print(f"F10.7: {d.get('F10.7', 'No disponible')}")
print(f"Densidad viento solar: {d['dens']} part/cm^3")
print(f"Velocidad viento solar: {d['velo']} km/s")

# Verificar calidad de datos
print(f"Calidad (Qbits): {d['Qbits']}")
```

NOTAS IMPORTANTES:
- Requiere descarga inicial de datos: spacepy.toolbox.update(QDomni=True)
- Los datos se almacenan localmente en formato HDF5
- Interpolación automática a la resolución temporal deseada
- Qbits=2: valor bien determinado (medido)
- Qbits=1: conexión parcial a mediciones
- Qbits=0: basado en promedios (menor confianza)

MÓDULO ADICIONAL: spacepy.plasmapy (para plasmasfera)
- Proporciona modelos de densidad de plasma H+, He+, O+
- Útil para extensiones hacia GEO
"""

# -----------------------------------------------------------------------------
# 3. IRI-PLAS (Asimilación de Contenido Total de Electrones)
# -----------------------------------------------------------------------------
"""
IRI-PLAS - International Reference Ionosphere with Plasmaspheric Extension
Versión disponible: Versión Fortran (requiere compilación)
Extensión de IRI con asimilación de datos TEC en tiempo real

Rango de altitudes: 50 - 20,200 km (extiende hasta la plasmasfera superior)
Tipo: Empírico con asimilación de datos
Uso: Corrección de perfiles ionosféricos con mediciones TEC

CARACTERÍSTICAS PRINCIPALES:
- Utiliza drivers estándar de IRI (F10.7, ap)
- Corrige activamente mediante asimilación de TEC (GIM-TEC)
- Escala perfiles foF2 y hmF2 para coincidir con TEC medido
- Extiende modelo de ionosfera a plasmasfera

PARÁMETROS DE ENTRADA ESTÁNDAR:
- Fecha/hora (UTC)
- Latitud geográfica [grados]
- Longitud geográfica [grados]
- Altitud [km]
- F10.7: Índice solar diario y promedio
- ap: Índice geomagnético
- Datos GIM-TEC (opcionales, para asimilación)

PARÁMETROS DE SALIDA:
- Ne: Densidad electrónica [m^-3]
- foF2: Frecuencia crítica de la capa F2 [MHz]
- hmF2: Altura máxima de la capa F2 [km]
- TEC: Contenido total de electrones [TECU]
- Ti: Temperatura de iones [K]
- Te: Temperatura de electrones [K]
- Composición iónica: O+, NO+, O2+, H+, He+

NOTA DE IMPLEMENTACIÓN:
- La versión Python (iri2016) requiere compilador Fortran
- Alternativa: Usar web API de IRI Online Calculator
- IRI-Plas es una extensión comercial/investigación del IRI estándar

REFERENCIA:
- Bilitza et al., "International Reference Ionosphere 2016", 2022
"""

# -----------------------------------------------------------------------------
# 4. AUTO-NVIS (Nowcasting en Tiempo Real)
# -----------------------------------------------------------------------------
"""
AUTO-NVIS - Automated Nowcasting via Ionospheric Sounding
Tipo: Sistema automatizado de monitoreo en tiempo real
Fuente de datos: NOAA SWPC (Space Weather Prediction Center) endpoints

Rango de altitudes: 60 - 600 km (ionosfera dinámica)
Actualización: Nivel de minutos
Uso principal: Monitoreo de condiciones ionosféricas actuales

CARACTERÍSTICAS PRINCIPALES:
- Obtiene índices directamente de endpoints de NOAA SWPC
- Actualización casi en tiempo real (minutos)
- Enfocado en ionosfera dinámica

PARÁMETROS MONITOREADOS:
- Flux de Rayos X (Escala R): R1-R5 (Minor a Extreme)
- Flux de Protones (Escala S): S1-S5 (Minor a Extreme)
- Índice Kp (Escala G): G1-G5 (Minor a Extreme)
- foF2, hmF2 en tiempo real desde ionosondas

ESCALAS DE CLIMA ESPACIAL NOAA:
R-Scale (Radio Blackouts) - Rayos X:
  R1 (Minor) a R5 (Extreme)

S-Scale (Solar Radiation Storms) - Protones:
  S1 (Minor) a S5 (Extreme)

G-Scale (Geomagnetic Storms) - Kp:
  G1 (Minor, Kp=5) a G5 (Extreme, Kp=9)

IMPLEMENTACIÓN:
- No hay paquete Python oficial "auto-nvis"
- Implementación custom requerida usando:
  * requests para HTTP GET a NOAA SWPC
  * Parsing de archivos TXT/XML
  * Endpoints: https://services.swpc.noaa.gov/

EJEMPLO CONCEPTUAL:
```python
import requests
from datetime import datetime

# Ejemplo de endpoint NOAA (URLs reales pueden variar)
url_kp = "https://services.swpc.noaa.gov/products/noaa-indices.json"
url_xray = "https://services.swpc.noaa.gov/products/goes-xray-flux.json"

response = requests.get(url_kp)
data = response.json()

# Extraer Kp más reciente
kp_actual = data[-1]['kp']
escala_g = data[-1]['derived_planetary_max_index']

print(f"Kp actual: {kp_actual}")
print(f"Escala G: {escala_g}")
```

NOTA: Auto-NVIS es más un sistema/concepto que una librería Python específica
"""

# -----------------------------------------------------------------------------
# 5. IRI2016 (Estándar Climatológico)
# -----------------------------------------------------------------------------
"""
IRI2016 - International Reference Ionosphere 2016
Versión Python: iri2016 1.11.1
Autores: International Union of Radio Science (URSI)
Documentación: https://github.com/aburrell/iri2016

Rango de altitudes: 60 - 2000 km (perfiles de temperatura hasta 3000 km)
Tipo: Empírico, climatológico (promedios mensuales)
Uso: Estándar internacional para ionosfera

PARÁMETROS DE ENTRADA:
- time (datetime o str): Fecha y hora UTC
- altkmrange (Sequence[float]): Lista/rango de altitudes [km]
- glat (float): Latitud geográfica [grados]
- glon (float): Longitud geográfica [grados]
- Opcionalmente:
  * F10.7: Índice solar (usa archivo apf107.dat si no se provee)
  * R12: Índice solar suavizado de 12 meses
  * ap: Índice geomagnético diario

ARCHIVOS DE ÍNDICES:
- apf107.dat: Contiene F10.7 diario y promedio de 12 meses
- Actualizado semi-anualmente
- Incluido automáticamente con el paquete

PARÁMETROS DE SALIDA (xarray.Dataset):
- Ne: Densidad de electrones [m^-3]
- foF2: Frecuencia crítica F2 [MHz]
- hmF2: Altura máxima F2 [km]
- foF1, hmF1: Parámetros capa F1
- foE, hmE: Parámetros capa E
- fbE: Frecuencia base de la capa E
- foEs, fbEs: Parámetros capa Es (sporádica)
- hmEs: Altura capa Es
- Te: Temperatura de electrones [K] (hasta 3000 km)
- Ti: Temperatura de iones [K]
- Ni: Densidad de iones principales [m^-3]:
  * O+ (Oxígeno ionizado)
  * NO+ (Óxido nítrico ionizado)
  * O2+ (Oxígeno molecular ionizado)
  * H+ (Hidrógeno ionizado, arriba de ~500 km)
  * He+ (Helio ionizado, arriba de ~500 km)

EJEMPLO DE USO:
```python
from iri2016 import IRI
import datetime

# Configurar parámetros
time = datetime.datetime(2020, 7, 22, 22, 18, 45)
altkmrange = [100, 200, 300, 400, 500]  # km
glat = 25.0  # latitud
glon = 102.0  # longitud

# Ejecutar modelo
result = IRI(time, altkmrange, glat, glon)

# Acceder a resultados
print(result)
print(f"Densidad electrónica: {result.Ne.values}")
print(f"foF2: {result.foF2.values} MHz")
print(f"hmF2: {result.hmF2.values} km")
print(f"Temperatura electrones: {result.Te.values} K")
```

REQUISITOS TÉCNICOS:
- Requiere compilador Fortran para construir el driver IRI
- Dependencias: numpy, pandas, xarray, matplotlib (para plots)

LIMITACIONES:
- Basado en promedios mensuales (no captura variabilidad diaria extrema)
- Menor precisión durante tormentas geomagnéticas intensas
- No asimila datos en tiempo real (a diferencia de IRI-Plas)

REFERENCIA:
- Bilitza, D., et al. "The International Reference Ionosphere 2016."
  Radio Science, 2022.
"""

# =============================================================================
# TABLA COMPARATIVA RESUMEN
# =============================================================================
"""
┌───────────────┬──────────────┬──────────┬───────────────┬──────────────┬─────────────────┐
│ Herramienta   │ Rango (km)   │ Precisión│ Índices       │ Salida       │ Uso Recomendado │
│               │              │          │ Requeridos    │ Principal    │                 │
├───────────────┼──────────────┼──────────┼───────────────┼──────────────┼─────────────────┤
│ PYATMOS/JB2008│ 90-2500      │ MÁXIMA   │ F10.7,S10,    │ rho, T       │ Tormentas       │
│               │              │          │ M10,Y10,Ap,Dst│              │ geomagnéticas   │
├───────────────┼──────────────┼──────────┼───────────────┼──────────────┼─────────────────┤
│ SpacePy/OMNI  │ 100-GEO      │ ALTA     │ Kp, Dst,      │ Todos índices│ Análisis de     │
│               │ (>36,000)    │          │ F10.7, dens,  │ + viento     │ misiones, GEO   │
│               │              │          │ velo, Pdyn    │ solar        │                 │
├───────────────┼──────────────┼──────────┼───────────────┼──────────────┼─────────────────┤
│ IRI-Plas      │ 50-20,200    │ ALTA     │ F10.7, ap +   │ Ne, TEC,     │ Ionosfera con   │
│               │              │ (con TEC)│ asimilación   │ foF2, hmF2,  │ asimilación TEC │
│               │              │          │ TEC           │ composición  │                 │
├───────────────┼──────────────┼──────────┼───────────────┼──────────────┼─────────────────┤
│ Auto-NVIS     │ 60-600       │ MEDIA    │ Kp, R-scale,  │ Índices      │ Nowcasting,     │
│               │              │ (tiempo  │ S-scale       │ en tiempo    │ alertas         │
│               │              │  real)   │               │ real         │ tempranas       │
├───────────────┼──────────────┼──────────┼───────────────┼──────────────┼─────────────────┤
│ IRI2016       │ 60-2000      │ MEDIA    │ F10.7, R12,   │ Ne, Te, Ti,  │ Climatología,   │
│               │ (Te hasta    │ (climato-│ ap            │ composición  │ promedios       │
│               │ 3000 km)     │ lógico)  │               │ iónica       │ mensuales       │
└───────────────┴──────────────┴──────────┴───────────────┴──────────────┴─────────────────┘
"""

# =============================================================================
# RECOMENDACIONES PARA TUFMax
# =============================================================================
"""
Para la librería TUFMax (0 km - GEO ~36,000 km), se recomienda:

1. ESTRATEGIA HÍBRIDA POR RANGOS DE ALTITUD:
   
   0-90 km:
   - USAR: COESA76 (pyatmos)
   - Razón: Único modelo que cubre desde nivel del mar
   - Parámetros: rho, T, P
   
   90-2000 km:
   - USAR: NRLMSISE-00 (pyatmos) como principal
   - Razón: Cubre todo el rango, proporciona 8 especies químicas
   - Alternativa: JB2008 para eventos de tormentas geomagnéticas
   - Parámetros: rho, T, [He, O, O2, N, N2, Ar, H, ANM O]
   
   2000-20,200 km:
   - USAR: IRI-Plas o IRI2016
   - Razón: Extensión a plasmasfera, densidad electrónica
   - Parámetros: Ne, Te, Ti, [O+, NO+, O2+, H+, He+]
   
   100-GEO (>36,000 km):
   - USAR: SpacePy.omni para índices + modelos de plasma extendidos
   - Razón: Validado hasta GEO, continuidad de plasma térmico
   - Parámetros: Kp, Dst, F10.7, dens, velo, Pdyn del viento solar

2. OBTENCIÓN DE ÍNDICES SOLARES Y GEOMAGNÉTICOS:
   
   Para máxima precisión (JB2008):
   - USAR: pyatmos.download_sw_jb2008()
   - Fuente: Space Environment Technologies (SET)
   - Índices: F10.7, S10, M10, Y10, Ap, Dst
   
   Para uso general (NRLMSISE-00):
   - USAR: pyatmos.download_sw_nrlmsise00()
   - Fuente: Celestrak (NASA/GSFC)
   - Índices: F10.7, Ap
   
   Para análisis de misiones:
   - USAR: spacepy.omni.get_omni()
   - Fuente: OMNIWeb validado
   - Índices: Kp, Dst, F10.7, parámetros de viento solar

3. IMPLEMENTACIÓN SUGERIDA:
   
   ```python
   # Pseudocódigo para TUFMax
   
   def calcular_parametros_atmosfericos(tiempo, lat, lon, alt):
       
       if alt < 90:  # Baja atmósfera
           resultado = coesa76([alt])
           return {'rho': resultado.rho, 'T': resultado.T, 'P': resultado.P}
       
       elif 90 <= alt < 2000:  # Atmósfera media/alta
           swdata = load_swdata()  # Precargar una vez
           resultado = nrlmsise00(tiempo, (lat, lon, alt), swdata)
           return {
               'rho': resultado.rho,
               'T': resultado.T,
               'especies': resultado.nd  # 8 especies
           }
       
       elif 2000 <= alt < 20200:  # Plasmasfera inferior
           resultado = IRI(tiempo, [alt], lat, lon)
           return {
               'Ne': resultado.Ne,
               'Te': resultado.Te,
               'Ti': resultado.Ti,
               'composicion_ionica': resultado.Ni
           }
       
       else:  # GEO y más allá
           ticks = crear_ticktock(tiempo)
           indices = omni.get_omni(ticks)
           return {
               'Kp': indices['Kp'],
               'Dst': indices['Dst'],
               'F10.7': indices.get('F10.7'),
               'viento_solar': {
                   'dens': indices['dens'],
                   'velo': indices['velo'],
                   'Pdyn': indices['Pdyn']
               }
           }
   ```

4. NOTAS FINALES:
   
   - Auto-NVIS e IRI-Plas no tienen paquetes Python oficiales listos
   - Se requiere implementación custom o uso de APIs web
   - Para producción, considerar caché local de índices solares/geomagnéticos
   - Validar siempre Qbits en SpacePy para calidad de datos
   - IRI2016 requiere compilador Fortran (dependencia adicional)
"""

if __name__ == "__main__":
    print(__doc__)
    
    # Ejemplos rápidos de verificación
    print("\n" + "="*70)
    print("VERIFICACIÓN DE PAQUETES INSTALADOS")
    print("="*70)
    
    try:
        import pyatmos
        print(f"✓ pyatmos v{pyatmos.__version__ if hasattr(pyatmos, '__version__') else 'N/A'}")
    except ImportError:
        print("✗ pyatmos no instalado")
    
    try:
        import spacepy
        print(f"✓ spacepy v{spacepy.__version__}")
    except ImportError:
        print("✗ spacepy no instalado")
    
    try:
        import iri2016
        print(f"✓ iri2016 v{'N/A'}")
    except ImportError:
        print("✗ iri2016 no instalado")
    
    print("\n" + "="*70)
    print("Fin del archivo de pruebas e investigación")
    print("="*70)
