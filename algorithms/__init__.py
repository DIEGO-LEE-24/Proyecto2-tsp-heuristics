"""
TSP Heuristic Algorithms Package
Paquete de algoritmos heurísticos para el Problema del Viajante de Comercio
"""

from .greedy import GreedyTSP
from .genetic import GeneticAlgorithmTSP
from .simulated_annealing import SimulatedAnnealingTSP, CoolingSchedule
from .hybrid import HybridTSP

__all__ = [
    'GreedyTSP',
    'GeneticAlgorithmTSP', 
    'SimulatedAnnealingTSP',
    'CoolingSchedule',
    'HybridTSP'
]

__version__ = '1.0.0'
__author__ = 'Lee Sang-cheol'
__email__ = 'lee.sangcheol@example.com'

# Información del paquete
PACKAGE_INFO = {
    'name': 'TSP Heuristic Algorithms',
    'version': __version__,
    'description': 'Implementación de algoritmos heurísticos para resolver el TSP',
    'algorithms': {
        'greedy': 'Algoritmos voraces (Nearest Neighbor, Cheapest Insertion)',
        'genetic': 'Algoritmo Genético con operadores especializados',
        'simulated_annealing': 'Recocido Simulado con esquemas de enfriamiento adaptativos',
        'hybrid': 'Algoritmos híbridos combinando múltiples estrategias'
    }
}