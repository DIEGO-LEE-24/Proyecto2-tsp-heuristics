"""
TSP Utilities Package
Paquete de utilidades para el Problema del Viajante de Comercio
"""

from .data_generator import TSPDataGenerator, CityPattern
from .visualizer import TSPVisualizer

__all__ = [
    'TSPDataGenerator',
    'CityPattern',
    'TSPVisualizer'
]

__version__ = '1.0.0'
__author__ = 'Lee Sang-cheol'
__email__ = 'lee.sangcheol@example.com'

# Información del paquete
PACKAGE_INFO = {
    'name': 'TSP Utilities',
    'version': __version__,
    'description': 'Herramientas de generación de datos y visualización para TSP',
    'modules': {
        'data_generator': 'Generación de ciudades con diferentes patrones',
        'visualizer': 'Visualización de tours y análisis de resultados'
    }
}