"""
Algoritmos Voraces (Greedy) para TSP
- Vecino más cercano (Nearest Neighbor)
- Inserción más barata (Cheapest Insertion)
- Mejora 2-opt
"""

import numpy as np
from typing import List, Tuple


class GreedyTSP:
    """Implementación de algoritmos voraces para TSP"""
    
    def __init__(self, dist_matrix: np.ndarray):
        """
        Inicializar con matriz de distancias
        
        Args:
            dist_matrix: Matriz de distancias entre ciudades
        """
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
    
    def nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Algoritmo del Vecino Más Cercano
        
        Args:
            start_city: Ciudad inicial (índice)
            
        Returns:
            tour: Lista con el orden de visita de ciudades
            total_cost: Costo total del tour
        """
        visited = [False] * self.n_cities
        tour = [start_city]
        visited[start_city] = True
        current_city = start_city
        total_cost = 0
        
        # Visitar n-1 ciudades
        for _ in range(self.n_cities - 1):
            min_distance = float('inf')
            nearest_city = -1
            
            # Encontrar la ciudad no visitada más cercana
            for j in range(self.n_cities):
                if not visited[j] and self.dist_matrix[current_city][j] < min_distance:
                    min_distance = self.dist_matrix[current_city][j]
                    nearest_city = j
            
            if nearest_city != -1:
                tour.append(nearest_city)
                visited[nearest_city] = True
                total_cost += min_distance
                current_city = nearest_city
        
        # Regresar a la ciudad inicial
        total_cost += self.dist_matrix[current_city][start_city]
        tour.append(start_city)
        
        return tour, total_cost
    
    def cheapest_insertion(self) -> Tuple[List[int], float]:
        """
        Algoritmo de Inserción Más Barata
        
        Returns:
            tour: Lista con el orden de visita de ciudades
            total_cost: Costo total del tour
        """
        # Inicializar con el par de ciudades más alejadas
        max_dist = 0
        city1, city2 = 0, 1
        
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                if self.dist_matrix[i][j] > max_dist:
                    max_dist = self.dist_matrix[i][j]
                    city1, city2 = i, j
        
        # Tour inicial con 2 ciudades
        tour = [city1, city2, city1]
        unvisited = set(range(self.n_cities)) - {city1, city2}
        
        # Agregar ciudades una por una
        while unvisited:
            best_city = -1
            best_position = -1
            min_increase = float('inf')
            
            # Para cada ciudad no visitada
            for city in unvisited:
                # Encontrar la mejor posición para insertar
                for i in range(len(tour) - 1):
                    # Calcular el incremento en distancia
                    increase = (self.dist_matrix[tour[i]][city] +
                               self.dist_matrix[city][tour[i + 1]] -
                               self.dist_matrix[tour[i]][tour[i + 1]])
                    
                    if increase < min_increase:
                        min_increase = increase
                        best_city = city
                        best_position = i + 1
            
            # Insertar la ciudad en la mejor posición
            tour.insert(best_position, best_city)
            unvisited.remove(best_city)
        
        # Calcular costo total
        total_cost = sum(self.dist_matrix[tour[i]][tour[i + 1]] 
                        for i in range(len(tour) - 1))
        
        return tour, total_cost
    
    def two_opt_improvement(self, tour: List[int]) -> Tuple[List[int], float]:
        """
        Mejora 2-opt para un tour existente
        
        Args:
            tour: Tour inicial
            
        Returns:
            improved_tour: Tour mejorado
            total_cost: Costo del tour mejorado
        """
        improved = True
        best_tour = tour.copy()
        
        while improved:
            improved = False
            
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue
                    
                    # Crear nuevo tour con segmento invertido
                    new_tour = best_tour.copy()
                    new_tour[i:j] = reversed(best_tour[i:j])
                    
                    # Calcular cambio en costo
                    current_cost = self._calculate_tour_cost(best_tour)
                    new_cost = self._calculate_tour_cost(new_tour)
                    
                    if new_cost < current_cost:
                        best_tour = new_tour
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_tour, self._calculate_tour_cost(best_tour)
    
    def three_opt_improvement(self, tour: List[int], max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        Mejora 3-opt para un tour existente
        
        Args:
            tour: Tour inicial
            max_iterations: Número máximo de iteraciones
            
        Returns:
            improved_tour: Tour mejorado
            total_cost: Costo del tour mejorado
        """
        best_tour = tour.copy()
        best_cost = self._calculate_tour_cost(best_tour)
        
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(len(tour) - 3):
                for j in range(i + 1, len(tour) - 2):
                    for k in range(j + 1, len(tour) - 1):
                        # Generar todas las reconexiones posibles
                        segments = [
                            best_tour[:i+1],
                            best_tour[i+1:j+1],
                            best_tour[j+1:k+1],
                            best_tour[k+1:]
                        ]
                        
                        # Probar diferentes reconexiones
                        reconnections = [
                            segments[0] + segments[1] + segments[2] + segments[3],  # Original
                            segments[0] + segments[1] + segments[2][::-1] + segments[3],
                            segments[0] + segments[1][::-1] + segments[2] + segments[3],
                            segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3],
                            segments[0] + segments[2] + segments[1] + segments[3],
                            segments[0] + segments[2] + segments[1][::-1] + segments[3],
                            segments[0] + segments[2][::-1] + segments[1] + segments[3],
                            segments[0] + segments[2][::-1] + segments[1][::-1] + segments[3]
                        ]
                        
                        for new_tour in reconnections[1:]:  # Saltar el original
                            new_cost = self._calculate_tour_cost(new_tour)
                            if new_cost < best_cost:
                                best_tour = new_tour
                                best_cost = new_cost
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_tour, best_cost
    
    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """
        Calcular el costo total de un tour
        
        Args:
            tour: Lista con el orden de visita
            
        Returns:
            Costo total del tour
        """
        if len(tour) < 2:
            return 0
            
        # Asegurarse de que el tour es cerrado
        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]
            
        return sum(self.dist_matrix[tour[i]][tour[i + 1]] 
                  for i in range(len(tour) - 1))
    
    def multiple_start_nearest_neighbor(self) -> Tuple[List[int], float]:
        """
        Ejecutar Vecino Más Cercano desde múltiples ciudades iniciales
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
        """
        best_tour = None
        best_cost = float('inf')
        
        for start_city in range(self.n_cities):
            tour, cost = self.nearest_neighbor(start_city)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        return best_tour, best_cost
    
    def savings_algorithm(self) -> Tuple[List[int], float]:
        """
        Algoritmo de Ahorros (Clarke-Wright)
        
        Returns:
            tour: Lista con el orden de visita de ciudades
            total_cost: Costo total del tour
        """
        # Calcular matriz de ahorros
        savings = []
        depot = 0  # Ciudad inicial/depósito
        
        for i in range(1, self.n_cities):
            for j in range(i + 1, self.n_cities):
                saving = (self.dist_matrix[depot][i] + 
                         self.dist_matrix[depot][j] - 
                         self.dist_matrix[i][j])
                savings.append((saving, i, j))
        
        # Ordenar por ahorro descendente
        savings.sort(reverse=True)
        
        # Inicializar rutas individuales
        routes = [[depot, i, depot] for i in range(1, self.n_cities)]
        
        # Combinar rutas basándose en ahorros
        for saving, i, j in savings:
            # Encontrar rutas que contengan i y j
            route_i = None
            route_j = None
            
            for route in routes:
                if i in route[1:-1]:  # Excluir depósito
                    route_i = route
                if j in route[1:-1]:
                    route_j = route
            
            # Si están en diferentes rutas y se pueden combinar
            if route_i and route_j and route_i != route_j:
                # Verificar si i y j están en los extremos
                if route_i[1] == i and route_j[-2] == j:
                    # Combinar rutas
                    new_route = route_j[:-1] + route_i[1:]
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(new_route)
                elif route_i[-2] == i and route_j[1] == j:
                    # Combinar rutas
                    new_route = route_i[:-1] + route_j[1:]
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(new_route)
        
        # Si quedan múltiples rutas, combinarlas
        if len(routes) > 1:
            combined_tour = [depot]
            for route in routes:
                combined_tour.extend(route[1:-1])
            combined_tour.append(depot)
        else:
            combined_tour = routes[0]
        
        total_cost = self._calculate_tour_cost(combined_tour)
        
        return combined_tour, total_cost
    
    def calculate_distance(self, tour: List[int]) -> float:
        """
        Calcular distancia total de un tour (método público)
        
        Args:
            tour: Tour a evaluar
            
        Returns:
            Distancia total
        """
        return self._calculate_tour_cost(tour)