#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP Solver - Comparación de Algoritmos Heurísticos
Autor: Lee Sang-cheol
Carné: 2024801079
Curso: Estructuras de Datos y Algoritmos
"""

import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Importar los algoritmos
from algorithms.greedy import GreedyTSP
from algorithms.genetic import GeneticAlgorithmTSP
from algorithms.simulated_annealing import SimulatedAnnealingTSP
from algorithms.hybrid import HybridTSP
from utils.data_generator import TSPDataGenerator
from utils.visualizer import TSPVisualizer


class TSPSolver:
    """Sistema principal para resolver y comparar algoritmos TSP"""
    
    def __init__(self, cities: List[Tuple[float, float]], verbose: bool = True):
        """
        Inicializar el solucionador TSP
        
        Args:
            cities: Lista de coordenadas (x, y) de las ciudades
            verbose: Si mostrar información durante la ejecución
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.verbose = verbose
        self.dist_matrix = self._calculate_distance_matrix()
        self.results = {}
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calcular matriz de distancias euclidianas"""
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.cities[i][0] - self.cities[j][0]
                dy = self.cities[i][1] - self.cities[j][1]
                distance = math.sqrt(dx * dx + dy * dy)
                dist_matrix[i][j] = dist_matrix[j][i] = distance
                
        return dist_matrix
    
    def solve_greedy(self, start_city: int = 0) -> Dict:
        """Resolver usando algoritmo voraz"""
        if self.verbose:
            print("\n" + "="*50)
            print("Ejecutando Algoritmo Voraz (Vecino Más Cercano)...")
            print("="*50)
            
        solver = GreedyTSP(self.dist_matrix)
        start_time = time.time()
        
        # Probar desde diferentes ciudades iniciales y tomar el mejor
        best_tour = None
        best_cost = float('inf')
        
        for start in range(min(5, self.n_cities)):  # Probar desde 5 ciudades diferentes
            tour, cost = solver.nearest_neighbor(start)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                
        # Aplicar mejora 2-opt
        improved_tour, improved_cost = solver.two_opt_improvement(best_tour)
        
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'Greedy + 2-opt',
            'tour': improved_tour,
            'cost': improved_cost,
            'time': execution_time,
            'improvement': (best_cost - improved_cost) / best_cost * 100
        }
        
        if self.verbose:
            print(f"Distancia inicial: {best_cost:.2f}")
            print(f"Distancia después de 2-opt: {improved_cost:.2f}")
            print(f"Mejora: {result['improvement']:.2f}%")
            print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
            
        return result
    
    def solve_genetic(self, population_size: int = 100, 
                     generations: int = 1000,
                     mutation_rate: float = 0.02) -> Dict:
        """Resolver usando algoritmo genético"""
        if self.verbose:
            print("\n" + "="*50)
            print("Ejecutando Algoritmo Genético...")
            print("="*50)
            print(f"Población: {population_size}, Generaciones: {generations}")
            
        solver = GeneticAlgorithmTSP(
            self.dist_matrix,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elite_size=20,
            tournament_size=5,
            verbose=self.verbose
        )
        
        start_time = time.time()
        best_tour, best_cost, history = solver.run()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'Genetic Algorithm',
            'tour': best_tour,
            'cost': best_cost,
            'time': execution_time,
            'history': history,
            'parameters': {
                'population_size': population_size,
                'generations': generations,
                'mutation_rate': mutation_rate
            }
        }
        
        if self.verbose:
            print(f"Mejor distancia encontrada: {best_cost:.2f}")
            print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
            
        return result
    
    def solve_simulated_annealing(self, initial_temp: float = 1000,
                                  final_temp: float = 0.1,
                                  alpha: float = 0.995) -> Dict:
        """Resolver usando recocido simulado"""
        if self.verbose:
            print("\n" + "="*50)
            print("Ejecutando Recocido Simulado...")
            print("="*50)
            print(f"Temperatura inicial: {initial_temp}, α: {alpha}")
            
        solver = SimulatedAnnealingTSP(
            self.dist_matrix,
            initial_temp=initial_temp,
            final_temp=final_temp,
            alpha=alpha,
            verbose=self.verbose
        )
        
        start_time = time.time()
        best_tour, best_cost, history = solver.run()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'Simulated Annealing',
            'tour': best_tour,
            'cost': best_cost,
            'time': execution_time,
            'history': history,
            'parameters': {
                'initial_temp': initial_temp,
                'final_temp': final_temp,
                'alpha': alpha
            }
        }
        
        if self.verbose:
            print(f"Mejor distancia encontrada: {best_cost:.2f}")
            print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
            
        return result
    
    def solve_hybrid(self) -> Dict:
        """Resolver usando algoritmo híbrido (GA + SA + 2-opt)"""
        if self.verbose:
            print("\n" + "="*50)
            print("Ejecutando Algoritmo Híbrido (GA + SA + 2-opt)...")
            print("="*50)
            
        solver = HybridTSP(self.dist_matrix, verbose=self.verbose)
        
        start_time = time.time()
        best_tour, best_cost, phase_results = solver.run()
        execution_time = time.time() - start_time
        
        result = {
            'algorithm': 'Hybrid (GA + SA + 2-opt)',
            'tour': best_tour,
            'cost': best_cost,
            'time': execution_time,
            'phase_results': phase_results
        }
        
        if self.verbose:
            print(f"Mejor distancia encontrada: {best_cost:.2f}")
            print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
            
        return result
    
    def compare_all_algorithms(self) -> Dict:
        """Ejecutar y comparar todos los algoritmos"""
        if self.verbose:
            print("\n" + "#"*60)
            print(f"# COMPARACIÓN DE ALGORITMOS TSP - {self.n_cities} CIUDADES")
            print("#"*60)
            
        # Ejecutar cada algoritmo
        self.results['greedy'] = self.solve_greedy()
        self.results['genetic'] = self.solve_genetic()
        self.results['simulated_annealing'] = self.solve_simulated_annealing()
        self.results['hybrid'] = self.solve_hybrid()
        
        # Generar resumen comparativo
        self._print_comparison_summary()
        
        return self.results
    
    def _print_comparison_summary(self):
        """Imprimir resumen comparativo de resultados"""
        print("\n" + "="*60)
        print("RESUMEN COMPARATIVO")
        print("="*60)
        
        # Encontrar el mejor resultado
        best_algorithm = min(self.results.items(), 
                            key=lambda x: x[1]['cost'])
        
        print(f"\n{'Algoritmo':<25} {'Distancia':<12} {'Tiempo (s)':<12} {'Gap (%)':<10}")
        print("-"*60)
        
        for name, result in self.results.items():
            gap = ((result['cost'] - best_algorithm[1]['cost']) / 
                   best_algorithm[1]['cost'] * 100)
            print(f"{result['algorithm']:<25} {result['cost']:<12.2f} "
                  f"{result['time']:<12.4f} {gap:<10.2f}")
            
        print("-"*60)
        print(f"\nMEJOR SOLUCIÓN: {best_algorithm[1]['algorithm']}")
        print(f"Distancia: {best_algorithm[1]['cost']:.2f}")
        print("="*60)
    
    def visualize_results(self, save_plots: bool = True):
        """Visualizar todos los resultados"""
        visualizer = TSPVisualizer(self.cities)
        
        # Crear visualización comparativa
        fig = plt.figure(figsize=(15, 10))
        
        # Configurar subplots
        positions = [(2, 3, i+1) for i in range(len(self.results))]
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = fig.add_subplot(positions[idx][0], positions[idx][1], 
                               positions[idx][2])
            visualizer.plot_tour(result['tour'], ax=ax, 
                               title=f"{result['algorithm']}\n"
                                     f"Distancia: {result['cost']:.2f}")
        
        # Gráfico de convergencia para GA y SA
        ax_conv = fig.add_subplot(2, 3, 5)
        if 'genetic' in self.results and 'history' in self.results['genetic']:
            ax_conv.plot(self.results['genetic']['history'], 
                        label='GA', color='blue')
        if 'simulated_annealing' in self.results and 'history' in self.results['simulated_annealing']:
            ax_conv.plot(self.results['simulated_annealing']['history'], 
                        label='SA', color='red')
        ax_conv.set_xlabel('Iteración')
        ax_conv.set_ylabel('Distancia')
        ax_conv.set_title('Convergencia')
        ax_conv.legend()
        ax_conv.grid(True, alpha=0.3)
        
        # Gráfico de barras comparativo
        ax_bar = fig.add_subplot(2, 3, 6)
        algorithms = [r['algorithm'] for r in self.results.values()]
        costs = [r['cost'] for r in self.results.values()]
        times = [r['time'] for r in self.results.values()]
        
        x = np.arange(len(algorithms))
        ax_bar.bar(x, costs, color='skyblue')
        ax_bar.set_xlabel('Algoritmo')
        ax_bar.set_ylabel('Distancia')
        ax_bar.set_title('Comparación de Distancias')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([a.split()[0] for a in algorithms], rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/comparison_{self.n_cities}cities_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\nGráfico guardado en: {filename}")
        
        plt.show()
    
    def save_results(self, filename: str = None):
        """Guardar resultados en archivo JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/tsp_results_{self.n_cities}cities_{timestamp}.json"
        
        # Preparar datos para JSON (convertir numpy arrays a listas)
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                'algorithm': result['algorithm'],
                'cost': result['cost'],
                'time': result['time'],
                'tour': result['tour'].tolist() if isinstance(result['tour'], np.ndarray) else result['tour']
            }
        
        with open(filename, 'w') as f:
            json.dump({
                'n_cities': self.n_cities,
                'cities': self.cities,
                'results': json_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nResultados guardados en: {filename}")


def main():
    """Función principal para demostración"""
    
    # Configuración
    print("\n" + "="*60)
    print("TSP SOLVER - COMPARACIÓN DE ALGORITMOS HEURÍSTICOS")
    print("="*60)
    
    # Generar datos de prueba
    print("\nGenerando ciudades aleatorias...")
    generator = TSPDataGenerator()
    
    # Probar con diferentes tamaños
    test_sizes = [20, 50, 100]
    
    for n_cities in test_sizes:
        print(f"\n{'='*60}")
        print(f"Prueba con {n_cities} ciudades")
        print(f"{'='*60}")
        
        # Generar ciudades
        cities = generator.generate_random_cities(n_cities, seed=42)
        
        # Crear solucionador
        solver = TSPSolver(cities, verbose=True)
        
        # Comparar algoritmos
        results = solver.compare_all_algorithms()
        
        # Visualizar resultados (solo para casos pequeños)
        if n_cities <= 50:
            solver.visualize_results(save_plots=True)
        
        # Guardar resultados
        solver.save_results()
        
        # Pequeña pausa entre pruebas
        if n_cities != test_sizes[-1]:
            input("\nPresiona Enter para continuar con la siguiente prueba...")


if __name__ == "__main__":
    main()