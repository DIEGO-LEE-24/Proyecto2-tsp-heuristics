"""
Recocido Simulado (Simulated Annealing) para TSP
Incluye diferentes esquemas de enfriamiento y operadores de vecindario
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional, Callable
from enum import Enum


class CoolingSchedule(Enum):
    """Tipos de esquemas de enfriamiento"""
    GEOMETRIC = "geometric"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    CAUCHY = "cauchy"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"


class SimulatedAnnealingTSP:
    """Implementación de Recocido Simulado para TSP"""
    
    def __init__(self, dist_matrix: np.ndarray,
                 initial_temp: float = 1000,
                 final_temp: float = 0.1,
                 alpha: float = 0.995,
                 cooling_schedule: CoolingSchedule = CoolingSchedule.GEOMETRIC,
                 max_iterations: Optional[int] = None,
                 verbose: bool = True):
        """
        Inicializar Recocido Simulado
        
        Args:
            dist_matrix: Matriz de distancias
            initial_temp: Temperatura inicial
            final_temp: Temperatura final
            alpha: Factor de enfriamiento (para esquema geométrico)
            cooling_schedule: Tipo de esquema de enfriamiento
            max_iterations: Número máximo de iteraciones
            verbose: Si mostrar progreso
        """
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.cooling_schedule = cooling_schedule
        self.verbose = verbose
        
        # Calcular iteraciones máximas si no se especifica
        if max_iterations is None:
            self.max_iterations = self._estimate_iterations()
        else:
            self.max_iterations = max_iterations
        
        # Para tracking
        self.temperature_history = []
        self.cost_history = []
        self.acceptance_history = []
        
    def _estimate_iterations(self) -> int:
        """
        Estimar número de iteraciones basado en temperatura y alpha
        
        Returns:
            Número estimado de iteraciones
        """
        if self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            # Calcular iteraciones necesarias para alcanzar temp_final
            iterations = int(math.log(self.final_temp / self.initial_temp) / 
                           math.log(self.alpha))
        else:
            # Estimación general
            iterations = 100 * self.n_cities
        
        return max(iterations, 1000)
    
    def calculate_cost(self, tour: List[int]) -> float:
        """
        Calcular costo total de un tour
        
        Args:
            tour: Tour a evaluar
            
        Returns:
            Costo total del tour
        """
        cost = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            cost += self.dist_matrix[from_city][to_city]
        return cost
    
    def generate_initial_solution(self, method: str = "random") -> List[int]:
        """
        Generar solución inicial
        
        Args:
            method: Método de generación ("random", "nearest", "greedy")
            
        Returns:
            Tour inicial
        """
        if method == "random":
            tour = list(range(self.n_cities))
            random.shuffle(tour)
            return tour
        
        elif method == "nearest":
            # Usar vecino más cercano para solución inicial
            from .greedy import GreedyTSP
            greedy = GreedyTSP(self.dist_matrix)
            tour, _ = greedy.nearest_neighbor()
            # Remover el último elemento (ciudad repetida)
            return tour[:-1]
        
        elif method == "greedy":
            # Usar inserción más barata
            from .greedy import GreedyTSP
            greedy = GreedyTSP(self.dist_matrix)
            tour, _ = greedy.cheapest_insertion()
            return tour[:-1]
        
        else:
            # Por defecto, aleatorio
            tour = list(range(self.n_cities))
            random.shuffle(tour)
            return tour
    
    def two_opt_neighbor(self, tour: List[int]) -> List[int]:
        """
        Generar vecino usando intercambio 2-opt
        
        Args:
            tour: Tour actual
            
        Returns:
            Tour vecino
        """
        new_tour = tour.copy()
        i = random.randint(0, self.n_cities - 1)
        j = random.randint(0, self.n_cities - 1)
        
        if i != j:
            if i > j:
                i, j = j, i
            # Invertir segmento
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
        
        return new_tour
    
    def three_opt_neighbor(self, tour: List[int]) -> List[int]:
        """
        Generar vecino usando intercambio 3-opt
        
        Args:
            tour: Tour actual
            
        Returns:
            Tour vecino
        """
        new_tour = tour.copy()
        
        # Seleccionar 3 puntos aleatorios
        points = sorted(random.sample(range(self.n_cities), 3))
        i, j, k = points
        
        # Elegir una de las posibles reconexiones 3-opt
        options = [
            # Opción 1: invertir primer segmento
            lambda t: t[:i] + t[i:j][::-1] + t[j:k] + t[k:],
            # Opción 2: invertir segundo segmento
            lambda t: t[:i] + t[i:j] + t[j:k][::-1] + t[k:],
            # Opción 3: invertir ambos segmentos
            lambda t: t[:i] + t[i:j][::-1] + t[j:k][::-1] + t[k:],
            # Opción 4: intercambiar segmentos
            lambda t: t[:i] + t[j:k] + t[i:j] + t[k:],
            # Opción 5: intercambiar e invertir primer segmento
            lambda t: t[:i] + t[j:k] + t[i:j][::-1] + t[k:],
            # Opción 6: intercambiar e invertir segundo segmento
            lambda t: t[:i] + t[j:k][::-1] + t[i:j] + t[k:],
        ]
        
        # Elegir una opción aleatoria
        transformation = random.choice(options)
        return transformation(new_tour)
    
    def swap_neighbor(self, tour: List[int]) -> List[int]:
        """
        Generar vecino intercambiando dos ciudades
        
        Args:
            tour: Tour actual
            
        Returns:
            Tour vecino
        """
        new_tour = tour.copy()
        i = random.randint(0, self.n_cities - 1)
        j = random.randint(0, self.n_cities - 1)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    def insertion_neighbor(self, tour: List[int]) -> List[int]:
        """
        Generar vecino moviendo una ciudad a otra posición
        
        Args:
            tour: Tour actual
            
        Returns:
            Tour vecino
        """
        new_tour = tour.copy()
        i = random.randint(0, self.n_cities - 1)
        j = random.randint(0, self.n_cities - 1)
        
        city = new_tour.pop(i)
        new_tour.insert(j, city)
        
        return new_tour
    
    def generate_neighbor(self, tour: List[int], 
                         temperature: float) -> List[int]:
        """
        Generar vecino adaptativo según temperatura
        
        Args:
            tour: Tour actual
            temperature: Temperatura actual
            
        Returns:
            Tour vecino
        """
        # A alta temperatura, usar movimientos más disruptivos
        # A baja temperatura, usar movimientos más conservadores
        
        temp_ratio = temperature / self.initial_temp
        
        if temp_ratio > 0.7:
            # Alta temperatura: movimientos grandes
            if random.random() < 0.5:
                return self.three_opt_neighbor(tour)
            else:
                return self.insertion_neighbor(tour)
        elif temp_ratio > 0.3:
            # Temperatura media: balance
            return self.two_opt_neighbor(tour)
        else:
            # Baja temperatura: movimientos pequeños
            if random.random() < 0.7:
                return self.two_opt_neighbor(tour)
            else:
                return self.swap_neighbor(tour)
    
    def acceptance_probability(self, current_cost: float, 
                              new_cost: float, 
                              temperature: float) -> float:
        """
        Calcular probabilidad de aceptación
        
        Args:
            current_cost: Costo de la solución actual
            new_cost: Costo de la nueva solución
            temperature: Temperatura actual
            
        Returns:
            Probabilidad de aceptación
        """
        if new_cost < current_cost:
            return 1.0
        
        if temperature <= 0:
            return 0.0
        
        try:
            delta = new_cost - current_cost
            probability = math.exp(-delta / temperature)
            return min(probability, 1.0)
        except OverflowError:
            return 0.0
    
    def update_temperature(self, iteration: int, 
                          current_temp: float) -> float:
        """
        Actualizar temperatura según el esquema de enfriamiento
        
        Args:
            iteration: Iteración actual
            current_temp: Temperatura actual
            
        Returns:
            Nueva temperatura
        """
        if self.cooling_schedule == CoolingSchedule.GEOMETRIC:
            return current_temp * self.alpha
        
        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            return self.initial_temp - (iteration * 
                   (self.initial_temp - self.final_temp) / self.max_iterations)
        
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return self.initial_temp / (1 + math.log(1 + iteration))
        
        elif self.cooling_schedule == CoolingSchedule.CAUCHY:
            return self.initial_temp / (1 + iteration)
        
        elif self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return self.initial_temp * math.exp(-iteration * 0.001)
        
        elif self.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # Enfriamiento adaptativo basado en tasa de aceptación
            if len(self.acceptance_history) > 100:
                recent_acceptance = np.mean(self.acceptance_history[-100:])
                if recent_acceptance > 0.8:
                    # Demasiada aceptación, enfriar más rápido
                    return current_temp * 0.95
                elif recent_acceptance < 0.2:
                    # Poca aceptación, enfriar más lento
                    return current_temp * 0.999
                else:
                    # Tasa de aceptación normal
                    return current_temp * self.alpha
            else:
                return current_temp * self.alpha
        
        else:
            return current_temp * self.alpha
    
    def run(self, initial_solution: Optional[List[int]] = None) -> Tuple[List[int], float, List[float]]:
        """
        Ejecutar algoritmo de recocido simulado
        
        Args:
            initial_solution: Solución inicial (opcional)
            
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            history: Historial de costos
        """
        # Generar solución inicial
        if initial_solution is None:
            current_tour = self.generate_initial_solution("random")
        else:
            current_tour = initial_solution.copy()
        
        current_cost = self.calculate_cost(current_tour)
        
        # Mejor solución encontrada
        best_tour = current_tour.copy()
        best_cost = current_cost
        
        # Variables de control
        temperature = self.initial_temp
        iteration = 0
        consecutive_no_improvement = 0
        
        if self.verbose:
            print(f"Temperatura inicial: {temperature:.2f}")
            print(f"Costo inicial: {current_cost:.2f}")
        
        # Bucle principal
        while temperature > self.final_temp and iteration < self.max_iterations:
            # Generar vecino
            neighbor_tour = self.generate_neighbor(current_tour, temperature)
            neighbor_cost = self.calculate_cost(neighbor_tour)
            
            # Calcular probabilidad de aceptación
            prob = self.acceptance_probability(current_cost, neighbor_cost, temperature)
            
            # Decidir si aceptar
            accepted = False
            if random.random() < prob:
                current_tour = neighbor_tour
                current_cost = neighbor_cost
                accepted = True
                
                # Actualizar mejor solución
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
                    consecutive_no_improvement = 0
                    
                    if self.verbose:
                        print(f"  Iteración {iteration}: Nueva mejor solución = {best_cost:.2f} "
                              f"(T={temperature:.2f})")
                else:
                    consecutive_no_improvement += 1
            else:
                consecutive_no_improvement += 1
            
            # Guardar historial
            self.temperature_history.append(temperature)
            self.cost_history.append(best_cost)
            self.acceptance_history.append(1 if accepted else 0)
            
            # Actualizar temperatura
            temperature = self.update_temperature(iteration, temperature)
            
            # Reportar progreso
            if self.verbose and iteration % 1000 == 0 and iteration > 0:
                acceptance_rate = np.mean(self.acceptance_history[-1000:])
                print(f"Iteración {iteration}: T={temperature:.2f}, "
                      f"Mejor={best_cost:.2f}, Actual={current_cost:.2f}, "
                      f"Aceptación={acceptance_rate:.2%}")
            
            # Criterio de parada temprana
            if consecutive_no_improvement > self.n_cities * 100:
                if self.verbose:
                    print(f"Parada temprana: sin mejora en {consecutive_no_improvement} iteraciones")
                break
            
            iteration += 1
        
        if self.verbose:
            print(f"\nFinalizado después de {iteration} iteraciones")
            print(f"Mejor solución encontrada: {best_cost:.2f}")
            final_acceptance = np.mean(self.acceptance_history[-100:]) if len(self.acceptance_history) > 100 else 0
            print(f"Tasa de aceptación final: {final_acceptance:.2%}")
        
        return best_tour, best_cost, self.cost_history
    
    def parallel_runs(self, n_runs: int = 4, 
                     n_jobs: int = -1) -> Tuple[List[int], float]:
        """
        Ejecutar múltiples instancias en paralelo
        
        Args:
            n_runs: Número de ejecuciones paralelas
            n_jobs: Número de trabajos paralelos (-1 para todos los cores)
            
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
        """
        from joblib import Parallel, delayed
        
        def single_run(seed):
            random.seed(seed)
            np.random.seed(seed)
            tour, cost, _ = self.run()
            return tour, cost
        
        if self.verbose:
            print(f"Ejecutando {n_runs} instancias en paralelo...")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(single_run)(seed) for seed in range(n_runs)
        )
        
        # Encontrar mejor resultado
        best_tour = None
        best_cost = float('inf')
        
        for tour, cost in results:
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        if self.verbose:
            print(f"Mejor resultado de {n_runs} ejecuciones: {best_cost:.2f}")
        
        return best_tour, best_cost
    
    def reheat(self, tour: List[int], reheat_temp: float, 
               iterations: int) -> Tuple[List[int], float]:
        """
        Recalentar y continuar optimización (para escapar de óptimos locales)
        
        Args:
            tour: Tour actual
            reheat_temp: Temperatura de recalentamiento
            iterations: Iteraciones adicionales
            
        Returns:
            improved_tour: Tour mejorado
            improved_cost: Costo mejorado
        """
        # Guardar configuración original
        original_temp = self.initial_temp
        original_iterations = self.max_iterations
        
        # Configurar para recalentamiento
        self.initial_temp = reheat_temp
        self.max_iterations = iterations
        
        # Ejecutar desde el tour dado
        improved_tour, improved_cost, _ = self.run(initial_solution=tour)
        
        # Restaurar configuración
        self.initial_temp = original_temp
        self.max_iterations = original_iterations
        
        return improved_tour, improved_cost