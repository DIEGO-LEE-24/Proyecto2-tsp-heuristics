"""
Generador de datos para pruebas de TSP
Incluye generación aleatoria y patrones específicos
"""

import numpy as np
import json
import math
from typing import List, Tuple, Optional, Dict
from enum import Enum


class CityPattern(Enum):
    """Patrones de distribución de ciudades"""
    RANDOM = "random"
    CIRCLE = "circle"
    GRID = "grid"
    CLUSTERED = "clustered"
    SPIRAL = "spiral"
    GAUSSIAN = "gaussian"


class TSPDataGenerator:
    """Generador de datos para problemas TSP"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializar generador
        
        Args:
            seed: Semilla para reproducibilidad
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
    
    def generate_random_cities(self, n: int, 
                              min_coord: float = 0, 
                              max_coord: float = 100,
                              seed: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Generar ciudades con distribución aleatoria uniforme
        
        Args:
            n: Número de ciudades
            min_coord: Coordenada mínima
            max_coord: Coordenada máxima
            seed: Semilla específica para esta generación
            
        Returns:
            Lista de coordenadas (x, y)
        """
        if seed is not None:
            np.random.seed(seed)
        
        cities = []
        for _ in range(n):
            x = np.random.uniform(min_coord, max_coord)
            y = np.random.uniform(min_coord, max_coord)
            cities.append((x, y))
        
        return cities
    
    def generate_circle_cities(self, n: int, 
                              radius: float = 50,
                              center: Tuple[float, float] = (50, 50)) -> List[Tuple[float, float]]:
        """
        Generar ciudades en un círculo
        
        Args:
            n: Número de ciudades
            radius: Radio del círculo
            center: Centro del círculo
            
        Returns:
            Lista de coordenadas (x, y)
        """
        cities = []
        angle_step = 2 * math.pi / n
        
        for i in range(n):
            angle = i * angle_step
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            cities.append((x, y))
        
        return cities
    
    def generate_grid_cities(self, rows: int, cols: int,
                            spacing: float = 10,
                            origin: Tuple[float, float] = (0, 0)) -> List[Tuple[float, float]]:
        """
        Generar ciudades en una cuadrícula
        
        Args:
            rows: Número de filas
            cols: Número de columnas
            spacing: Espaciado entre ciudades
            origin: Origen de la cuadrícula
            
        Returns:
            Lista de coordenadas (x, y)
        """
        cities = []
        
        for i in range(rows):
            for j in range(cols):
                x = origin[0] + j * spacing
                y = origin[1] + i * spacing
                cities.append((x, y))
        
        return cities
    
    def generate_clustered_cities(self, n: int,
                                 n_clusters: int = 3,
                                 cluster_std: float = 5,
                                 area_size: float = 100) -> List[Tuple[float, float]]:
        """
        Generar ciudades en clusters
        
        Args:
            n: Número total de ciudades
            n_clusters: Número de clusters
            cluster_std: Desviación estándar dentro de cada cluster
            area_size: Tamaño del área total
            
        Returns:
            Lista de coordenadas (x, y)
        """
        cities = []
        cities_per_cluster = n // n_clusters
        remaining = n % n_clusters
        
        # Generar centros de clusters
        cluster_centers = self.generate_random_cities(n_clusters, 
                                                      cluster_std * 2, 
                                                      area_size - cluster_std * 2)
        
        for i, center in enumerate(cluster_centers):
            # Determinar número de ciudades en este cluster
            cluster_size = cities_per_cluster
            if i < remaining:
                cluster_size += 1
            
            # Generar ciudades alrededor del centro
            for _ in range(cluster_size):
                x = np.random.normal(center[0], cluster_std)
                y = np.random.normal(center[1], cluster_std)
                
                # Asegurar que está dentro del área
                x = np.clip(x, 0, area_size)
                y = np.clip(y, 0, area_size)
                
                cities.append((x, y))
        
        return cities
    
    def generate_spiral_cities(self, n: int,
                              rotations: float = 3,
                              max_radius: float = 50,
                              center: Tuple[float, float] = (50, 50)) -> List[Tuple[float, float]]:
        """
        Generar ciudades en espiral
        
        Args:
            n: Número de ciudades
            rotations: Número de rotaciones de la espiral
            max_radius: Radio máximo de la espiral
            center: Centro de la espiral
            
        Returns:
            Lista de coordenadas (x, y)
        """
        cities = []
        
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            angle = 2 * math.pi * rotations * t
            radius = max_radius * t
            
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            
            cities.append((x, y))
        
        return cities
    
    def generate_gaussian_cities(self, n: int,
                                mean: Tuple[float, float] = (50, 50),
                                std: float = 20) -> List[Tuple[float, float]]:
        """
        Generar ciudades con distribución gaussiana
        
        Args:
            n: Número de ciudades
            mean: Media de la distribución
            std: Desviación estándar
            
        Returns:
            Lista de coordenadas (x, y)
        """
        cities = []
        
        for _ in range(n):
            x = np.random.normal(mean[0], std)
            y = np.random.normal(mean[1], std)
            cities.append((x, y))
        
        return cities
    
    def generate_cities(self, n: int, 
                       pattern: CityPattern = CityPattern.RANDOM,
                       **kwargs) -> List[Tuple[float, float]]:
        """
        Generar ciudades según un patrón específico
        
        Args:
            n: Número de ciudades
            pattern: Patrón de distribución
            **kwargs: Parámetros adicionales para el patrón
            
        Returns:
            Lista de coordenadas (x, y)
        """
        if pattern == CityPattern.RANDOM:
            return self.generate_random_cities(n, **kwargs)
        elif pattern == CityPattern.CIRCLE:
            return self.generate_circle_cities(n, **kwargs)
        elif pattern == CityPattern.GRID:
            # Calcular filas y columnas para n ciudades
            rows = int(math.sqrt(n))
            cols = math.ceil(n / rows)
            cities = self.generate_grid_cities(rows, cols, **kwargs)
            return cities[:n]  # Truncar si hay extras
        elif pattern == CityPattern.CLUSTERED:
            return self.generate_clustered_cities(n, **kwargs)
        elif pattern == CityPattern.SPIRAL:
            return self.generate_spiral_cities(n, **kwargs)
        elif pattern == CityPattern.GAUSSIAN:
            return self.generate_gaussian_cities(n, **kwargs)
        else:
            return self.generate_random_cities(n, **kwargs)
    
    def calculate_distance_matrix(self, cities: List[Tuple[float, float]],
                                 distance_type: str = "euclidean") -> np.ndarray:
        """
        Calcular matriz de distancias
        
        Args:
            cities: Lista de coordenadas
            distance_type: Tipo de distancia ("euclidean", "manhattan", "geo")
            
        Returns:
            Matriz de distancias
        """
        n = len(cities)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if distance_type == "euclidean":
                    dist = self._euclidean_distance(cities[i], cities[j])
                elif distance_type == "manhattan":
                    dist = self._manhattan_distance(cities[i], cities[j])
                elif distance_type == "geo":
                    dist = self._geo_distance(cities[i], cities[j])
                else:
                    dist = self._euclidean_distance(cities[i], cities[j])
                
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def _euclidean_distance(self, city1: Tuple[float, float], 
                          city2: Tuple[float, float]) -> float:
        """Distancia euclidiana"""
        dx = city1[0] - city2[0]
        dy = city1[1] - city2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _manhattan_distance(self, city1: Tuple[float, float], 
                          city2: Tuple[float, float]) -> float:
        """Distancia Manhattan"""
        return abs(city1[0] - city2[0]) + abs(city1[1] - city2[1])
    
    def _geo_distance(self, city1: Tuple[float, float], 
                     city2: Tuple[float, float]) -> float:
        """
        Distancia geográfica (asumiendo coordenadas lat/lon)
        Usa la fórmula de Haversine
        """
        # Radio de la Tierra en km
        R = 6371
        
        lat1, lon1 = math.radians(city1[0]), math.radians(city1[1])
        lat2, lon2 = math.radians(city2[0]), math.radians(city2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def save_to_file(self, cities: List[Tuple[float, float]], 
                    filename: str,
                    metadata: Optional[Dict] = None):
        """
        Guardar ciudades en archivo JSON
        
        Args:
            cities: Lista de ciudades
            filename: Nombre del archivo
            metadata: Metadatos adicionales
        """
        data = {
            'n_cities': len(cities),
            'cities': cities,
            'seed': self.seed
        }
        
        if metadata:
            data['metadata'] = metadata
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        Cargar ciudades desde archivo JSON
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            cities: Lista de ciudades
            metadata: Metadatos del archivo
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        cities = [tuple(city) for city in data['cities']]
        metadata = data.get('metadata', {})
        
        return cities, metadata
    
    def generate_benchmark_set(self, output_dir: str = "data/"):
        """
        Generar conjunto de benchmarks estándar
        
        Args:
            output_dir: Directorio de salida
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        benchmarks = [
            # Pequeñas
            {'n': 10, 'pattern': CityPattern.RANDOM, 'name': 'random_10'},
            {'n': 15, 'pattern': CityPattern.CIRCLE, 'name': 'circle_15'},
            {'n': 20, 'pattern': CityPattern.GRID, 'name': 'grid_20'},
            
            # Medianas
            {'n': 30, 'pattern': CityPattern.RANDOM, 'name': 'random_30'},
            {'n': 50, 'pattern': CityPattern.CLUSTERED, 'name': 'clustered_50'},
            {'n': 75, 'pattern': CityPattern.SPIRAL, 'name': 'spiral_75'},
            
            # Grandes
            {'n': 100, 'pattern': CityPattern.RANDOM, 'name': 'random_100'},
            {'n': 150, 'pattern': CityPattern.GAUSSIAN, 'name': 'gaussian_150'},
            {'n': 200, 'pattern': CityPattern.CLUSTERED, 'name': 'clustered_200'},
        ]
        
        for benchmark in benchmarks:
            cities = self.generate_cities(benchmark['n'], benchmark['pattern'])
            
            metadata = {
                'pattern': benchmark['pattern'].value,
                'description': f"{benchmark['n']} cities with {benchmark['pattern'].value} distribution"
            }
            
            filename = os.path.join(output_dir, f"{benchmark['name']}.json")
            self.save_to_file(cities, filename, metadata)
            
            print(f"Generated: {filename}")
    
    def generate_tsplib_format(self, cities: List[Tuple[float, float]], 
                              name: str = "TSP_INSTANCE") -> str:
        """
        Generar formato TSPLIB
        
        Args:
            cities: Lista de ciudades
            name: Nombre de la instancia
            
        Returns:
            String en formato TSPLIB
        """
        tsplib = []
        tsplib.append(f"NAME: {name}")
        tsplib.append(f"TYPE: TSP")
        tsplib.append(f"DIMENSION: {len(cities)}")
        tsplib.append("EDGE_WEIGHT_TYPE: EUC_2D")
        tsplib.append("NODE_COORD_SECTION")
        
        for i, (x, y) in enumerate(cities, 1):
            tsplib.append(f"{i} {x:.6f} {y:.6f}")
        
        tsplib.append("EOF")
        
        return "\n".join(tsplib)
    
    def create_asymmetric_matrix(self, cities: List[Tuple[float, float]], 
                                asymmetry_factor: float = 0.2) -> np.ndarray:
        """
        Crear matriz de distancias asimétrica (para TSP asimétrico)
        
        Args:
            cities: Lista de ciudades
            asymmetry_factor: Factor de asimetría (0 = simétrico, 1 = muy asimétrico)
            
        Returns:
            Matriz de distancias asimétrica
        """
        n = len(cities)
        dist_matrix = self.calculate_distance_matrix(cities)
        
        # Hacer la matriz asimétrica
        for i in range(n):
            for j in range(i + 1, n):
                # Agregar asimetría aleatoria
                factor = 1 + np.random.uniform(-asymmetry_factor, asymmetry_factor)
                dist_matrix[i][j] *= factor
                
                factor = 1 + np.random.uniform(-asymmetry_factor, asymmetry_factor)
                dist_matrix[j][i] *= factor
        
        return dist_matrix