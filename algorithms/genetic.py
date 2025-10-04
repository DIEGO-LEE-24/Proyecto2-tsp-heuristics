"""
Algoritmo Genético para TSP
Incluye operadores de selección, cruce y mutación especializados
"""

import numpy as np
import random
from typing import List, Tuple, Optional


class GeneticAlgorithmTSP:
    """Implementación de Algoritmo Genético para TSP"""
    
    def __init__(self, dist_matrix: np.ndarray, 
                 population_size: int = 100,
                 generations: int = 1000,
                 mutation_rate: float = 0.02,
                 elite_size: int = 20,
                 tournament_size: int = 5,
                 verbose: bool = True):
        """
        Inicializar algoritmo genético
        
        Args:
            dist_matrix: Matriz de distancias
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Probabilidad de mutación
            elite_size: Tamaño de élite (mejores individuos que pasan directamente)
            tournament_size: Tamaño del torneo para selección
            verbose: Si mostrar progreso
        """
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.verbose = verbose
        
        # Para tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def create_individual(self) -> List[int]:
        """
        Crear un individuo aleatorio (tour)
        
        Returns:
            Tour aleatorio
        """
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour
    
    def create_population(self) -> List[List[int]]:
        """
        Crear población inicial
        
        Returns:
            Lista de individuos (tours)
        """
        population = []
        for _ in range(self.population_size):
            population.append(self.create_individual())
        return population
    
    def calculate_fitness(self, tour: List[int]) -> float:
        """
        Calcular fitness de un individuo (inverso de la distancia)
        
        Args:
            tour: Tour a evaluar
            
        Returns:
            Fitness del tour
        """
        distance = self.calculate_distance(tour)
        return 1.0 / distance if distance > 0 else float('inf')
    
    def calculate_distance(self, tour: List[int]) -> float:
        """
        Calcular distancia total de un tour
        
        Args:
            tour: Tour a evaluar
            
        Returns:
            Distancia total
        """
        distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            distance += self.dist_matrix[from_city][to_city]
        return distance
    
    def rank_population(self, population: List[List[int]]) -> List[Tuple[List[int], float]]:
        """
        Ordenar población por fitness
        
        Args:
            population: Lista de individuos
            
        Returns:
            Lista de tuplas (individuo, fitness) ordenada por fitness
        """
        fitness_results = []
        for individual in population:
            fitness_results.append((individual, self.calculate_fitness(individual)))
        
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def selection_tournament(self, ranked_pop: List[Tuple[List[int], float]]) -> List[int]:
        """
        Selección por torneo
        
        Args:
            ranked_pop: Población ordenada por fitness
            
        Returns:
            Individuo seleccionado
        """
        tournament = random.sample(ranked_pop, min(self.tournament_size, len(ranked_pop)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0].copy()
    
    def selection_roulette_wheel(self, ranked_pop: List[Tuple[List[int], float]]) -> List[int]:
        """
        Selección por ruleta
        
        Args:
            ranked_pop: Población ordenada por fitness
            
        Returns:
            Individuo seleccionado
        """
        total_fitness = sum(ind[1] for ind in ranked_pop)
        selection_probs = [ind[1]/total_fitness for ind in ranked_pop]
        
        # Selección acumulativa
        cumulative = 0
        rand = random.random()
        for i, prob in enumerate(selection_probs):
            cumulative += prob
            if rand < cumulative:
                return ranked_pop[i][0].copy()
        
        return ranked_pop[-1][0].copy()
    
    def crossover_order(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX)
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Hijo generado
        """
        size = len(parent1)
        child = [-1] * size
        
        # Seleccionar segmento aleatorio del padre 1
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        
        # Copiar segmento del padre 1
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Completar con elementos del padre 2 en orden
        current_pos = (end + 1) % size
        parent2_pos = (end + 1) % size
        
        while -1 in child:
            if parent2[parent2_pos] not in child:
                child[current_pos] = parent2[parent2_pos]
                current_pos = (current_pos + 1) % size
            parent2_pos = (parent2_pos + 1) % size
        
        return child
    
    def crossover_pmx(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Partially Mapped Crossover (PMX)
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Hijo generado
        """
        size = len(parent1)
        child = [-1] * size
        
        # Seleccionar segmento aleatorio
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        
        # Copiar segmento del padre 1
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Mapear elementos del padre 2
        for i in range(start, end + 1):
            if parent2[i] not in child:
                # Encontrar posición para insertar
                pos = i
                while child[pos] != -1:
                    # Buscar el elemento que está en child[pos] en parent2
                    element = child[pos]
                    pos = parent2.index(element)
                child[pos] = parent2[i]
        
        # Completar con elementos restantes del padre 2
        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]
        
        return child
    
    def crossover_cycle(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Cycle Crossover (CX)
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Hijo generado
        """
        size = len(parent1)
        child = [-1] * size
        
        # Encontrar ciclos
        cycles = []
        visited = [False] * size
        
        for start in range(size):
            if not visited[start]:
                cycle = []
                pos = start
                
                while pos not in cycle:
                    cycle.append(pos)
                    visited[pos] = True
                    pos = parent1.index(parent2[pos])
                    if pos == start:
                        break
                
                cycles.append(cycle)
        
        # Alternar ciclos entre padres
        for i, cycle in enumerate(cycles):
            parent = parent1 if i % 2 == 0 else parent2
            for pos in cycle:
                child[pos] = parent[pos]
        
        return child
    
    def mutation_swap(self, individual: List[int]) -> List[int]:
        """
        Mutación por intercambio
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Individuo mutado
        """
        if random.random() < self.mutation_rate:
            mutated = individual.copy()
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            return mutated
        return individual
    
    def mutation_inverse(self, individual: List[int]) -> List[int]:
        """
        Mutación por inversión (2-opt)
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Individuo mutado
        """
        if random.random() < self.mutation_rate:
            mutated = individual.copy()
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            
            if i > j:
                i, j = j, i
            
            mutated[i:j+1] = reversed(mutated[i:j+1])
            return mutated
        return individual
    
    def mutation_insertion(self, individual: List[int]) -> List[int]:
        """
        Mutación por inserción
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Individuo mutado
        """
        if random.random() < self.mutation_rate:
            mutated = individual.copy()
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            
            city = mutated.pop(i)
            mutated.insert(j, city)
            return mutated
        return individual
    
    def mutation_scramble(self, individual: List[int]) -> List[int]:
        """
        Mutación por mezcla de subsecuencia
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Individuo mutado
        """
        if random.random() < self.mutation_rate:
            mutated = individual.copy()
            i = random.randint(0, len(individual) - 1)
            j = random.randint(i, len(individual) - 1)
            
            subsequence = mutated[i:j+1]
            random.shuffle(subsequence)
            mutated[i:j+1] = subsequence
            return mutated
        return individual
    
    def create_mating_pool(self, ranked_pop: List[Tuple[List[int], float]]) -> List[List[int]]:
        """
        Crear pool de apareamiento con élite
        
        Args:
            ranked_pop: Población ordenada por fitness
            
        Returns:
            Pool de apareamiento
        """
        mating_pool = []
        
        # Preservar élite
        for i in range(self.elite_size):
            mating_pool.append(ranked_pop[i][0].copy())
        
        # Seleccionar resto de la población
        for _ in range(len(ranked_pop) - self.elite_size):
            mating_pool.append(self.selection_tournament(ranked_pop))
        
        return mating_pool
    
    def breed_population(self, mating_pool: List[List[int]]) -> List[List[int]]:
        """
        Crear nueva generación mediante cruce
        
        Args:
            mating_pool: Pool de apareamiento
            
        Returns:
            Nueva población
        """
        children = []
        
        # Preservar élite
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Crear descendencia
        for _ in range(len(mating_pool) - self.elite_size):
            # Seleccionar padres
            parent1 = random.choice(mating_pool[:50])  # Sesgo hacia mejores individuos
            parent2 = random.choice(mating_pool)
            
            # Aplicar crossover
            if random.random() < 0.8:  # Probabilidad de crossover
                child = self.crossover_order(parent1, parent2)
            else:
                child = parent1.copy() if random.random() < 0.5 else parent2.copy()
            
            children.append(child)
        
        return children
    
    def mutate_population(self, population: List[List[int]]) -> List[List[int]]:
        """
        Aplicar mutación a la población
        
        Args:
            population: Población actual
            
        Returns:
            Población mutada
        """
        mutated_pop = []
        
        for individual in population:
            # Aplicar diferentes tipos de mutación
            if random.random() < 0.5:
                mutated = self.mutation_inverse(individual)
            elif random.random() < 0.7:
                mutated = self.mutation_swap(individual)
            elif random.random() < 0.9:
                mutated = self.mutation_insertion(individual)
            else:
                mutated = self.mutation_scramble(individual)
            
            mutated_pop.append(mutated)
        
        return mutated_pop
    
    def next_generation(self, current_gen: List[List[int]]) -> List[List[int]]:
        """
        Crear siguiente generación
        
        Args:
            current_gen: Generación actual
            
        Returns:
            Nueva generación
        """
        # Ordenar población actual
        ranked_pop = self.rank_population(current_gen)
        
        # Crear pool de apareamiento
        mating_pool = self.create_mating_pool(ranked_pop)
        
        # Crear descendencia
        children = self.breed_population(mating_pool)
        
        # Aplicar mutación
        next_gen = self.mutate_population(children)
        
        return next_gen
    
    def run(self) -> Tuple[List[int], float, List[float]]:
        """
        Ejecutar algoritmo genético
        
        Returns:
            best_tour: Mejor tour encontrado
            best_distance: Distancia del mejor tour
            history: Historial de mejor fitness
        """
        # Crear población inicial
        population = self.create_population()
        
        best_ever_tour = None
        best_ever_distance = float('inf')
        
        # Evolucionar población
        for generation in range(self.generations):
            # Evaluar población
            ranked_pop = self.rank_population(population)
            
            # Obtener mejor individuo
            best_individual = ranked_pop[0][0]
            best_distance = self.calculate_distance(best_individual)
            
            # Actualizar mejor histórico
            if best_distance < best_ever_distance:
                best_ever_distance = best_distance
                best_ever_tour = best_individual.copy()
            
            # Guardar historial
            self.best_fitness_history.append(best_ever_distance)
            avg_fitness = np.mean([1/self.calculate_distance(ind[0]) 
                                  for ind in ranked_pop])
            self.avg_fitness_history.append(avg_fitness)
            
            # Imprimir progreso
            if self.verbose and generation % 100 == 0:
                print(f"Generación {generation}: Mejor = {best_ever_distance:.2f}, "
                      f"Actual = {best_distance:.2f}")
            
            # Crear siguiente generación
            population = self.next_generation(population)
            
            # Diversificación adaptativa
            if generation > 100 and generation % 200 == 0:
                # Si no hay mejora significativa, aumentar diversidad
                recent_improvement = (self.best_fitness_history[-100] - 
                                    self.best_fitness_history[-1])
                
                if recent_improvement < 0.01:  # Poco progreso
                    # Reemplazar parte de la población con individuos aleatorios
                    for i in range(self.population_size // 4):
                        population[-(i+1)] = self.create_individual()
                    
                    if self.verbose:
                        print(f"  → Diversificación aplicada en generación {generation}")
        
        return best_ever_tour, best_ever_distance, self.best_fitness_history
    
    def adaptive_run(self) -> Tuple[List[int], float, List[float]]:
        """
        Ejecutar algoritmo genético con parámetros adaptativos
        
        Returns:
            best_tour: Mejor tour encontrado
            best_distance: Distancia del mejor tour
            history: Historial de mejor fitness
        """
        # Parámetros adaptativos
        adaptive_mutation_rate = self.mutation_rate
        stagnation_counter = 0
        
        # Crear población inicial
        population = self.create_population()
        
        best_ever_tour = None
        best_ever_distance = float('inf')
        
        for generation in range(self.generations):
            # Ajustar tasa de mutación adaptivamente
            if stagnation_counter > 50:
                adaptive_mutation_rate = min(0.2, adaptive_mutation_rate * 1.5)
            else:
                adaptive_mutation_rate = max(0.01, adaptive_mutation_rate * 0.98)
            
            self.mutation_rate = adaptive_mutation_rate
            
            # Evaluar población
            ranked_pop = self.rank_population(population)
            
            # Obtener mejor individuo
            best_individual = ranked_pop[0][0]
            best_distance = self.calculate_distance(best_individual)
            
            # Verificar mejora
            if best_distance < best_ever_distance:
                improvement = best_ever_distance - best_distance
                best_ever_distance = best_distance
                best_ever_tour = best_individual.copy()
                stagnation_counter = 0
                
                if self.verbose:
                    print(f"Generación {generation}: Nueva mejor solución = {best_ever_distance:.2f} "
                          f"(mejora: {improvement:.2f})")
            else:
                stagnation_counter += 1
            
            # Guardar historial
            self.best_fitness_history.append(best_ever_distance)
            
            # Crear siguiente generación
            population = self.next_generation(population)
        
        return best_ever_tour, best_ever_distance, self.best_fitness_history