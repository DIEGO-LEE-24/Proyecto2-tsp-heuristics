"""
Benchmark tool for TSP algorithms
Mide y compara el rendimiento de diferentes algoritmos
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import argparse

from algorithms.greedy import GreedyTSP
from algorithms.genetic import GeneticAlgorithmTSP
from algorithms.simulated_annealing import SimulatedAnnealingTSP
from algorithms.hybrid import HybridTSP
from utils.data_generator import TSPDataGenerator, CityPattern


class TSPBenchmark:
    """Herramienta de benchmark para algoritmos TSP"""
    
    def __init__(self, output_dir: str = "results/benchmarks/"):
        """
        Inicializar benchmark
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = output_dir
        self.results = []
        
        # Crear directorio si no existe
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def benchmark_algorithm(self, algorithm_name: str, 
                           dist_matrix: np.ndarray,
                           iterations: int = 1) -> Dict:
        """
        Medir rendimiento de un algoritmo
        
        Args:
            algorithm_name: Nombre del algoritmo
            dist_matrix: Matriz de distancias
            iterations: Número de iteraciones para promediar
            
        Returns:
            Diccionario con resultados
        """
        times = []
        costs = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            if algorithm_name == "greedy":
                solver = GreedyTSP(dist_matrix)
                tour, cost = solver.nearest_neighbor()
                tour, cost = solver.two_opt_improvement(tour)
                
            elif algorithm_name == "genetic":
                solver = GeneticAlgorithmTSP(
                    dist_matrix,
                    population_size=100,
                    generations=500,
                    verbose=False
                )
                tour, cost, _ = solver.run()
                
            elif algorithm_name == "simulated_annealing":
                solver = SimulatedAnnealingTSP(
                    dist_matrix,
                    initial_temp=1000,
                    final_temp=0.1,
                    alpha=0.995,
                    verbose=False
                )
                tour, cost, _ = solver.run()
                
            elif algorithm_name == "hybrid":
                solver = HybridTSP(dist_matrix, verbose=False)
                tour, cost, _ = solver.run()
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            execution_time = time.time() - start_time
            times.append(execution_time)
            costs.append(cost)
        
        return {
            'algorithm': algorithm_name,
            'n_cities': len(dist_matrix),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'min_cost': np.min(costs),
            'iterations': iterations
        }
    
    def run_benchmark_suite(self, city_sizes: List[int],
                          algorithms: List[str],
                          iterations: int = 5,
                          patterns: List[CityPattern] = None) -> pd.DataFrame:
        """
        Ejecutar suite completa de benchmarks
        
        Args:
            city_sizes: Lista de tamaños de ciudades a probar
            algorithms: Lista de algoritmos a probar
            iterations: Iteraciones por prueba
            patterns: Patrones de distribución de ciudades
            
        Returns:
            DataFrame con todos los resultados
        """
        if patterns is None:
            patterns = [CityPattern.RANDOM]
        
        print("\n" + "="*60)
        print("INICIANDO BENCHMARK SUITE")
        print("="*60)
        
        generator = TSPDataGenerator(seed=42)
        results = []
        
        total_tests = len(city_sizes) * len(algorithms) * len(patterns)
        current_test = 0
        
        for pattern in patterns:
            for n_cities in city_sizes:
                # Generar ciudades
                cities = generator.generate_cities(n_cities, pattern)
                dist_matrix = generator.calculate_distance_matrix(cities)
                
                print(f"\n[{pattern.value}] Testing with {n_cities} cities...")
                
                for algorithm in algorithms:
                    current_test += 1
                    print(f"  [{current_test}/{total_tests}] {algorithm}...", end='')
                    
                    # Ajustar parámetros según tamaño
                    if algorithm == "hybrid" and n_cities > 50:
                        print(" (skipped - too large)", end='')
                        continue
                    
                    try:
                        result = self.benchmark_algorithm(
                            algorithm, dist_matrix, iterations
                        )
                        result['pattern'] = pattern.value
                        results.append(result)
                        print(f" ✓ (avg: {result['avg_time']:.2f}s, cost: {result['avg_cost']:.2f})")
                    except Exception as e:
                        print(f" ✗ (error: {str(e)})")
        
        # Convertir a DataFrame
        df = pd.DataFrame(results)
        
        # Guardar resultados
        self.save_results(df)
        
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Guardar resultados en archivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar CSV
        csv_file = f"{self.output_dir}/benchmark_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nResultados guardados en: {csv_file}")
        
        # Guardar JSON
        json_file = f"{self.output_dir}/benchmark_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
    
    def plot_results(self, df: pd.DataFrame, save: bool = True):
        """
        Generar gráficos de resultados
        
        Args:
            df: DataFrame con resultados
            save: Si guardar los gráficos
        """
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tiempo vs Tamaño del problema
        ax1 = axes[0, 0]
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm]
            ax1.plot(data['n_cities'], data['avg_time'], 
                    marker='o', label=algorithm, linewidth=2)
        ax1.set_xlabel('Número de Ciudades')
        ax1.set_ylabel('Tiempo de Ejecución (s)')
        ax1.set_title('Escalabilidad: Tiempo vs Tamaño')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Calidad de Solución vs Tamaño
        ax2 = axes[0, 1]
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm]
            ax2.plot(data['n_cities'], data['avg_cost'], 
                    marker='s', label=algorithm, linewidth=2)
        ax2.set_xlabel('Número de Ciudades')
        ax2.set_ylabel('Distancia Total')
        ax2.set_title('Calidad de Solución vs Tamaño')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade-off Tiempo vs Calidad
        ax3 = axes[1, 0]
        for n in df['n_cities'].unique():
            data = df[df['n_cities'] == n]
            ax3.scatter(data['avg_time'], data['avg_cost'], 
                       label=f'{n} ciudades', s=100, alpha=0.7)
            
            # Anotar algoritmos
            for _, row in data.iterrows():
                ax3.annotate(row['algorithm'][:3], 
                           (row['avg_time'], row['avg_cost']),
                           fontsize=8, alpha=0.7)
        
        ax3.set_xlabel('Tiempo de Ejecución (s)')
        ax3.set_ylabel('Distancia Total')
        ax3.set_title('Trade-off: Tiempo vs Calidad')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # 4. Tabla de Resumen
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Crear tabla resumen
        summary = df.groupby('algorithm').agg({
            'avg_time': 'mean',
            'avg_cost': 'mean',
            'n_cities': 'max'
        }).round(2)
        
        table = ax4.table(cellText=summary.values,
                         colLabels=['Avg Time (s)', 'Avg Cost', 'Max Cities'],
                         rowLabels=summary.index,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax4.set_title('Resumen de Rendimiento', pad=20)
        
        plt.suptitle('Benchmark Results - TSP Algorithms', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/benchmark_plots_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Gráficos guardados en: {filename}")
        
        plt.show()
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generar reporte en texto
        
        Args:
            df: DataFrame con resultados
            
        Returns:
            Reporte en texto
        """
        report = []
        report.append("="*60)
        report.append("BENCHMARK REPORT - TSP ALGORITHMS")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen por algoritmo
        report.append("ALGORITHM PERFORMANCE SUMMARY")
        report.append("-"*40)
        
        for algorithm in df['algorithm'].unique():
            data = df[df['algorithm'] == algorithm]
            report.append(f"\n{algorithm.upper()}:")
            report.append(f"  Avg Time: {data['avg_time'].mean():.3f}s")
            report.append(f"  Avg Cost: {data['avg_cost'].mean():.2f}")
            report.append(f"  Time Std Dev: {data['avg_time'].std():.3f}s")
            report.append(f"  Tested on: {data['n_cities'].unique().tolist()} cities")
        
        # Mejor algoritmo por categoría
        report.append("\n" + "="*40)
        report.append("BEST PERFORMERS")
        report.append("-"*40)
        
        # Más rápido
        fastest = df.loc[df['avg_time'].idxmin()]
        report.append(f"Fastest: {fastest['algorithm']} ({fastest['avg_time']:.3f}s)")
        
        # Mejor calidad
        best_quality = df.loc[df['avg_cost'].idxmin()]
        report.append(f"Best Quality: {best_quality['algorithm']} (cost: {best_quality['avg_cost']:.2f})")
        
        # Mejor balance
        df['score'] = df['avg_cost'] * df['avg_time']  # Simple score
        best_balance = df.loc[df['score'].idxmin()]
        report.append(f"Best Balance: {best_balance['algorithm']}")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nReporte guardado en: {report_file}")
        
        return report_text


def main():
    """Función principal para ejecutar benchmarks"""
    parser = argparse.ArgumentParser(description='TSP Algorithm Benchmarking Tool')
    parser.add_argument('--instances', type=str, default='small',
                       help='Instance sizes: small, medium, large, or comma-separated numbers')
    parser.add_argument('--algorithms', type=str, default='all',
                       help='Algorithms to test: all, greedy, ga, sa, hybrid')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per test')
    parser.add_argument('--patterns', type=str, default='random',
                       help='City patterns: random, circle, grid, clustered')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Configurar tamaños de instancia
    if args.instances == 'small':
        city_sizes = [10, 15, 20, 25, 30]
    elif args.instances == 'medium':
        city_sizes = [40, 50, 60, 75]
    elif args.instances == 'large':
        city_sizes = [100, 150, 200]
    elif args.instances == 'all':
        city_sizes = [10, 20, 30, 50, 75, 100]
    else:
        city_sizes = [int(x) for x in args.instances.split(',')]
    
    # Configurar algoritmos
    if args.algorithms == 'all':
        algorithms = ['greedy', 'genetic', 'simulated_annealing']
        if max(city_sizes) <= 50:
            algorithms.append('hybrid')
    else:
        algorithms = args.algorithms.split(',')
    
    # Configurar patrones
    pattern_map = {
        'random': CityPattern.RANDOM,
        'circle': CityPattern.CIRCLE,
        'grid': CityPattern.GRID,
        'clustered': CityPattern.CLUSTERED
    }
    patterns = [pattern_map.get(p, CityPattern.RANDOM) 
                for p in args.patterns.split(',')]
    
    # Ejecutar benchmark
    benchmark = TSPBenchmark()
    
    print(f"\nConfiguration:")
    print(f"  City sizes: {city_sizes}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Patterns: {[p.value for p in patterns]}")
    
    # Ejecutar benchmarks
    df = benchmark.run_benchmark_suite(
        city_sizes=city_sizes,
        algorithms=algorithms,
        iterations=args.iterations,
        patterns=patterns
    )
    
    # Generar reporte
    report = benchmark.generate_report(df)
    print("\n" + report)
    
    # Generar gráficos si se solicita
    if args.plot:
        benchmark.plot_results(df)
    
    print("\n✅ Benchmark completado!")


if __name__ == "__main__":
    main()