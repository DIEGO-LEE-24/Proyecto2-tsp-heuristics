"""
Algoritmo Híbrido para TSP
Combina GA + SA + Búsqueda Local (2-opt/3-opt)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .genetic import GeneticAlgorithmTSP
from .simulated_annealing import SimulatedAnnealingTSP, CoolingSchedule
from .greedy import GreedyTSP


class HybridTSP:
    """Implementación de algoritmo híbrido para TSP"""
    
    def __init__(self, dist_matrix: np.ndarray, verbose: bool = True):
        """
        Inicializar algoritmo híbrido
        
        Args:
            dist_matrix: Matriz de distancias
            verbose: Si mostrar progreso
        """
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.verbose = verbose
        
        # Componentes del algoritmo
        self.greedy_solver = GreedyTSP(dist_matrix)
        self.ga_solver = None
        self.sa_solver = None
        
        # Resultados de cada fase
        self.phase_results = {}
        
    def run(self, strategy: str = "sequential") -> Tuple[List[int], float, Dict]:
        """
        Ejecutar algoritmo híbrido
        
        Args:
            strategy: Estrategia de hibridación
                     "sequential": GA → SA → Local Search
                     "parallel": Ejecutar en paralelo y combinar
                     "embedded": SA embebido en GA
                     "adaptive": Cambiar entre algoritmos adaptativamente
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            phase_results: Resultados de cada fase
        """
        if strategy == "sequential":
            return self._run_sequential()
        elif strategy == "parallel":
            return self._run_parallel()
        elif strategy == "embedded":
            return self._run_embedded()
        elif strategy == "adaptive":
            return self._run_adaptive()
        else:
            return self._run_sequential()
    
    def _run_sequential(self) -> Tuple[List[int], float, Dict]:
        """
        Ejecutar algoritmos en secuencia: Greedy → GA → SA → 2-opt
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            phase_results: Resultados de cada fase
        """
        if self.verbose:
            print("\n" + "="*60)
            print("ALGORITMO HÍBRIDO SECUENCIAL")
            print("="*60)
        
        # Fase 1: Solución inicial con algoritmo voraz
        if self.verbose:
            print("\nFase 1: Generando solución inicial con algoritmo voraz...")
        
        greedy_tour, greedy_cost = self.greedy_solver.multiple_start_nearest_neighbor()
        self.phase_results['greedy'] = {
            'tour': greedy_tour,
            'cost': greedy_cost
        }
        
        if self.verbose:
            print(f"  Costo inicial (Greedy): {greedy_cost:.2f}")
        
        # Fase 2: Exploración global con Algoritmo Genético
        if self.verbose:
            print("\nFase 2: Exploración global con Algoritmo Genético...")
        
        self.ga_solver = GeneticAlgorithmTSP(
            self.dist_matrix,
            population_size=100,
            generations=500,
            mutation_rate=0.02,
            elite_size=20,
            verbose=False
        )
        
        # Incluir solución voraz en población inicial
        ga_population = self.ga_solver.create_population()
        ga_population[0] = greedy_tour[:-1]  # Remover ciudad duplicada
        
        ga_tour, ga_cost, ga_history = self.ga_solver.run()
        self.phase_results['genetic'] = {
            'tour': ga_tour,
            'cost': ga_cost,
            'history': ga_history,
            'improvement': (greedy_cost - ga_cost) / greedy_cost * 100
        }
        
        if self.verbose:
            print(f"  Costo después de GA: {ga_cost:.2f} "
                  f"(mejora: {self.phase_results['genetic']['improvement']:.2f}%)")
        
        # Fase 3: Refinamiento con Recocido Simulado
        if self.verbose:
            print("\nFase 3: Refinamiento local con Recocido Simulado...")
        
        self.sa_solver = SimulatedAnnealingTSP(
            self.dist_matrix,
            initial_temp=100,  # Temperatura más baja para refinamiento
            final_temp=0.01,
            alpha=0.99,
            cooling_schedule=CoolingSchedule.ADAPTIVE,
            verbose=False
        )
        
        sa_tour, sa_cost, sa_history = self.sa_solver.run(initial_solution=ga_tour)
        self.phase_results['simulated_annealing'] = {
            'tour': sa_tour,
            'cost': sa_cost,
            'history': sa_history,
            'improvement': (ga_cost - sa_cost) / ga_cost * 100
        }
        
        if self.verbose:
            print(f"  Costo después de SA: {sa_cost:.2f} "
                  f"(mejora: {self.phase_results['simulated_annealing']['improvement']:.2f}%)")
        
        # Fase 4: Optimización final con búsqueda local intensiva
        if self.verbose:
            print("\nFase 4: Optimización final con búsqueda local (2-opt + 3-opt)...")
        
        # Aplicar 2-opt
        final_tour = sa_tour + [sa_tour[0]]  # Hacer el tour cerrado
        opt2_tour, opt2_cost = self.greedy_solver.two_opt_improvement(final_tour)
        
        # Aplicar 3-opt si hay mejora significativa
        if self.n_cities <= 30:  # Solo para instancias pequeñas/medianas
            opt3_tour, opt3_cost = self.greedy_solver.three_opt_improvement(opt2_tour)
            if opt3_cost < opt2_cost:
                final_tour, final_cost = opt3_tour, opt3_cost
            else:
                final_tour, final_cost = opt2_tour, opt2_cost
        else:
            final_tour, final_cost = opt2_tour, opt2_cost
        
        self.phase_results['local_search'] = {
            'tour': final_tour,
            'cost': final_cost,
            'improvement': (sa_cost - final_cost) / sa_cost * 100
        }
        
        if self.verbose:
            print(f"  Costo final: {final_cost:.2f} "
                  f"(mejora: {self.phase_results['local_search']['improvement']:.2f}%)")
            
            total_improvement = (greedy_cost - final_cost) / greedy_cost * 100
            print(f"\n  Mejora total: {total_improvement:.2f}%")
            print(f"  Reducción de distancia: {greedy_cost - final_cost:.2f}")
        
        # Remover ciudad duplicada del tour final
        if final_tour[0] == final_tour[-1]:
            final_tour = final_tour[:-1]
        
        return final_tour, final_cost, self.phase_results
    
    def _run_parallel(self) -> Tuple[List[int], float, Dict]:
        """
        Ejecutar algoritmos en paralelo y combinar resultados
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            phase_results: Resultados de cada fase
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        if self.verbose:
            print("\n" + "="*60)
            print("ALGORITMO HÍBRIDO PARALELO")
            print("="*60)
            print("\nEjecutando algoritmos en paralelo...")
        
        start_time = time.time()
        results = {}
        
        def run_greedy():
            tour, cost = self.greedy_solver.multiple_start_nearest_neighbor()
            # Aplicar 2-opt
            improved_tour, improved_cost = self.greedy_solver.two_opt_improvement(tour)
            return 'greedy', improved_tour, improved_cost
        
        def run_genetic():
            ga = GeneticAlgorithmTSP(
                self.dist_matrix,
                population_size=150,
                generations=1000,
                mutation_rate=0.02,
                verbose=False
            )
            tour, cost, _ = ga.run()
            return 'genetic', tour, cost
        
        def run_sa():
            sa = SimulatedAnnealingTSP(
                self.dist_matrix,
                initial_temp=1000,
                final_temp=0.1,
                alpha=0.995,
                verbose=False
            )
            tour, cost, _ = sa.run()
            return 'sa', tour, cost
        
        # Ejecutar en paralelo
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_greedy),
                executor.submit(run_genetic),
                executor.submit(run_sa)
            ]
            
            for future in as_completed(futures):
                name, tour, cost = future.result()
                results[name] = {'tour': tour, 'cost': cost}
                
                if self.verbose:
                    print(f"  {name.upper()} completado: costo = {cost:.2f}")
        
        # Encontrar mejor resultado
        best_name = min(results.keys(), key=lambda k: results[k]['cost'])
        best_tour = results[best_name]['tour']
        best_cost = results[best_name]['cost']
        
        if self.verbose:
            print(f"\nMejor algoritmo individual: {best_name.upper()} con costo {best_cost:.2f}")
        
        # Fase de combinación: crear población con mejores tours
        if self.verbose:
            print("\nFase de combinación: mezclando mejores soluciones...")
        
        # Crear población élite con los mejores tours
        elite_population = []
        for result in results.values():
            tour = result['tour']
            if isinstance(tour, list) and tour[0] == tour[-1]:
                tour = tour[:-1]  # Remover ciudad duplicada
            elite_population.append(tour)
        
        # Agregar variaciones de los mejores tours
        for _ in range(7):  # Completar población de 10
            base_tour = elite_population[_ % 3].copy()
            # Aplicar mutación suave
            i, j = np.random.choice(len(base_tour), 2, replace=False)
            base_tour[i], base_tour[j] = base_tour[j], base_tour[i]
            elite_population.append(base_tour)
        
        # Mini GA con población élite
        ga_combiner = GeneticAlgorithmTSP(
            self.dist_matrix,
            population_size=10,
            generations=200,
            mutation_rate=0.05,
            elite_size=3,
            verbose=False
        )
        
        # Reemplazar población inicial con élite
        ga_combiner.population = elite_population
        combined_tour, combined_cost, _ = ga_combiner.run()
        
        # Refinamiento final con SA
        sa_refiner = SimulatedAnnealingTSP(
            self.dist_matrix,
            initial_temp=50,
            final_temp=0.01,
            alpha=0.995,
            verbose=False
        )
        
        final_tour, final_cost, _ = sa_refiner.run(initial_solution=combined_tour)
        
        # Aplicar búsqueda local final
        if isinstance(final_tour, list) and final_tour[0] != final_tour[-1]:
            final_tour = final_tour + [final_tour[0]]
        
        final_tour, final_cost = self.greedy_solver.two_opt_improvement(final_tour)
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"  Costo después de combinación: {combined_cost:.2f}")
            print(f"  Costo final: {final_cost:.2f}")
            print(f"  Tiempo total: {elapsed_time:.2f} segundos")
        
        # Guardar resultados
        self.phase_results = {
            'parallel_results': results,
            'combined': {'tour': combined_tour, 'cost': combined_cost},
            'final': {'tour': final_tour, 'cost': final_cost},
            'execution_time': elapsed_time
        }
        
        # Remover ciudad duplicada si existe
        if final_tour[0] == final_tour[-1]:
            final_tour = final_tour[:-1]
        
        return final_tour, final_cost, self.phase_results
    
    def _run_embedded(self) -> Tuple[List[int], float, Dict]:
        """
        GA con SA embebido para refinamiento de individuos élite
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            phase_results: Resultados de cada fase
        """
        if self.verbose:
            print("\n" + "="*60)
            print("ALGORITMO HÍBRIDO EMBEBIDO (GA + SA)")
            print("="*60)
        
        class GeneticWithSA(GeneticAlgorithmTSP):
            """GA modificado con SA embebido"""
            
            def __init__(self, dist_matrix, sa_frequency=50, sa_elite_size=5, **kwargs):
                super().__init__(dist_matrix, **kwargs)
                self.sa_frequency = sa_frequency
                self.sa_elite_size = sa_elite_size
                self.sa_solver = SimulatedAnnealingTSP(
                    dist_matrix,
                    initial_temp=50,
                    final_temp=1,
                    alpha=0.95,
                    verbose=False
                )
            
            def next_generation(self, current_gen):
                """Sobrescribir para incluir SA en individuos élite"""
                # Crear siguiente generación normalmente
                next_gen = super().next_generation(current_gen)
                
                # Cada sa_frequency generaciones, aplicar SA a élite
                if hasattr(self, 'generation_counter'):
                    self.generation_counter += 1
                else:
                    self.generation_counter = 1
                
                if self.generation_counter % self.sa_frequency == 0:
                    # Ordenar población
                    ranked = self.rank_population(next_gen)
                    
                    # Aplicar SA a los mejores individuos
                    for i in range(min(self.sa_elite_size, len(ranked))):
                        elite_tour = ranked[i][0]
                        refined_tour, refined_cost, _ = self.sa_solver.run(
                            initial_solution=elite_tour
                        )
                        next_gen[i] = refined_tour
                    
                    if self.verbose:
                        print(f"  SA aplicado a élite en generación {self.generation_counter}")
                
                return next_gen
        
        # Crear y ejecutar GA híbrido
        hybrid_ga = GeneticWithSA(
            self.dist_matrix,
            population_size=100,
            generations=1000,
            mutation_rate=0.02,
            elite_size=20,
            sa_frequency=100,
            sa_elite_size=5,
            verbose=self.verbose
        )
        
        best_tour, best_cost, history = hybrid_ga.run()
        
        # Refinamiento final
        if self.verbose:
            print("\nRefinamiento final con búsqueda local...")
        
        if isinstance(best_tour, list) and best_tour[0] != best_tour[-1]:
            best_tour = best_tour + [best_tour[0]]
        
        final_tour, final_cost = self.greedy_solver.two_opt_improvement(best_tour)
        
        self.phase_results = {
            'hybrid_ga': {'tour': best_tour, 'cost': best_cost},
            'final': {'tour': final_tour, 'cost': final_cost},
            'history': history
        }
        
        if self.verbose:
            print(f"Costo final: {final_cost:.2f}")
        
        # Remover ciudad duplicada si existe
        if final_tour[0] == final_tour[-1]:
            final_tour = final_tour[:-1]
        
        return final_tour, final_cost, self.phase_results
    
    def _run_adaptive(self) -> Tuple[List[int], float, Dict]:
        """
        Cambiar adaptativamente entre algoritmos según rendimiento
        
        Returns:
            best_tour: Mejor tour encontrado
            best_cost: Costo del mejor tour
            phase_results: Resultados de cada fase
        """
        if self.verbose:
            print("\n" + "="*60)
            print("ALGORITMO HÍBRIDO ADAPTATIVO")
            print("="*60)
        
        # Inicializar con solución voraz
        current_tour, current_cost = self.greedy_solver.nearest_neighbor()
        if current_tour[-1] == current_tour[0]:
            current_tour = current_tour[:-1]
        
        best_tour = current_tour.copy()
        best_cost = current_cost
        
        # Parámetros adaptativos
        no_improvement_count = 0
        max_no_improvement = 3
        algorithm_performance = {'ga': [], 'sa': [], 'local': []}
        
        iteration = 0
        max_iterations = 10
        
        if self.verbose:
            print(f"Costo inicial: {current_cost:.2f}")
        
        while iteration < max_iterations and no_improvement_count < max_no_improvement:
            iteration += 1
            
            # Seleccionar algoritmo basado en rendimiento histórico
            if iteration == 1:
                # Primera iteración: GA para exploración
                selected_algorithm = 'ga'
            elif no_improvement_count >= 2:
                # Sin mejora reciente: intentar algo diferente
                # Elegir el menos usado recientemente
                recent_uses = {
                    'ga': len([a for a in algorithm_performance['ga'][-3:] if a > 0]),
                    'sa': len([a for a in algorithm_performance['sa'][-3:] if a > 0]),
                    'local': len([a for a in algorithm_performance['local'][-3:] if a > 0])
                }
                selected_algorithm = min(recent_uses.keys(), key=recent_uses.get)
            else:
                # Seleccionar basado en rendimiento promedio
                avg_performance = {}
                for alg in ['ga', 'sa', 'local']:
                    if algorithm_performance[alg]:
                        avg_performance[alg] = np.mean(algorithm_performance[alg][-5:])
                    else:
                        avg_performance[alg] = 0
                
                # Agregar algo de exploración aleatoria
                if np.random.random() < 0.2:
                    selected_algorithm = np.random.choice(['ga', 'sa', 'local'])
                else:
                    selected_algorithm = max(avg_performance.keys(), key=avg_performance.get)
            
            if self.verbose:
                print(f"\nIteración {iteration}: Usando {selected_algorithm.upper()}")
            
            # Ejecutar algoritmo seleccionado
            if selected_algorithm == 'ga':
                ga = GeneticAlgorithmTSP(
                    self.dist_matrix,
                    population_size=50,
                    generations=200,
                    mutation_rate=0.03,
                    verbose=False
                )
                # Incluir solución actual en población
                ga_pop = ga.create_population()
                ga_pop[0] = current_tour
                new_tour, new_cost, _ = ga.run()
                
            elif selected_algorithm == 'sa':
                sa = SimulatedAnnealingTSP(
                    self.dist_matrix,
                    initial_temp=100 * (1 + no_improvement_count),  # Aumentar T si hay estancamiento
                    final_temp=0.1,
                    alpha=0.99,
                    verbose=False
                )
                new_tour, new_cost, _ = sa.run(initial_solution=current_tour)
                
            else:  # local search
                tour_with_end = current_tour + [current_tour[0]]
                new_tour, new_cost = self.greedy_solver.two_opt_improvement(tour_with_end)
                if new_tour[-1] == new_tour[0]:
                    new_tour = new_tour[:-1]
            
            # Calcular mejora
            improvement = (current_cost - new_cost) / current_cost * 100
            algorithm_performance[selected_algorithm].append(max(improvement, 0))
            
            # Actualizar si hay mejora
            if new_cost < current_cost:
                current_tour = new_tour
                current_cost = new_cost
                no_improvement_count = 0
                
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
                    
                if self.verbose:
                    print(f"  Mejora encontrada: {current_cost:.2f} ({improvement:.2f}%)")
            else:
                no_improvement_count += 1
                if self.verbose:
                    print(f"  Sin mejora (contador: {no_improvement_count})")
        
        # Refinamiento final intensivo
        if self.verbose:
            print("\nRefinamiento final intensivo...")
        
        # Aplicar SA con temperatura baja
        sa_final = SimulatedAnnealingTSP(
            self.dist_matrix,
            initial_temp=10,
            final_temp=0.001,
            alpha=0.999,
            verbose=False
        )
        final_tour, final_cost, _ = sa_final.run(initial_solution=best_tour)
        
        # Aplicar búsqueda local
        if final_tour[0] != final_tour[-1]:
            final_tour = final_tour + [final_tour[0]]
        final_tour, final_cost = self.greedy_solver.two_opt_improvement(final_tour)
        
        if self.verbose:
            print(f"\nCosto final: {final_cost:.2f}")
            total_improvement = (self.greedy_solver.calculate_distance(current_tour) - final_cost) / self.greedy_solver.calculate_distance(current_tour) * 100
            print(f"Mejora total: {total_improvement:.2f}%")
        
        self.phase_results = {
            'algorithm_performance': algorithm_performance,
            'iterations': iteration,
            'final_cost': final_cost
        }
        
        # Remover ciudad duplicada si existe
        if final_tour[0] == final_tour[-1]:
            final_tour = final_tour[:-1]
        
        return final_tour, final_cost, self.phase_results