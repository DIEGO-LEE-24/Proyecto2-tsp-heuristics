#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo TSP con ciudades reales de Costa Rica
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.costa_rica_data import (
    get_costa_rica_cities, 
    create_costa_rica_distance_matrix,
    COSTA_RICA_CITIES,
    COSTA_RICA_SUBSETS
)
from algorithms.greedy import GreedyTSP
from algorithms.genetic import GeneticAlgorithmTSP
from algorithms.simulated_annealing import SimulatedAnnealingTSP
from algorithms.hybrid import HybridTSP


def test_costa_rica_tsp(subset='valle_central', algorithm='all'):
    """
    Prueba TSP con ciudades reales de Costa Rica
    
    Args:
        subset: 'valle_central', 'principales', 'pacifico', 'caribe', 'completo'
        algorithm: 'greedy', 'ga', 'sa', 'hybrid', 'all'
    """
    print("\n" + "="*60)
    print("TSP - CIUDADES DE COSTA RICA")
    print("="*60)
    
    # Obtener datos
    dist_matrix, all_cities = create_costa_rica_distance_matrix()
    cities_indices = COSTA_RICA_SUBSETS[subset]
    
    # Filtrar ciudades seleccionadas
    n = len(cities_indices)
    selected_dist_matrix = np.zeros((n, n))
    selected_cities = []
    selected_names = []
    
    for i, idx1 in enumerate(cities_indices):
        selected_cities.append(all_cities[idx1])
        selected_names.append(COSTA_RICA_CITIES[idx1][2])
        for j, idx2 in enumerate(cities_indices):
            selected_dist_matrix[i][j] = dist_matrix[idx1][idx2]
    
    print(f"\nRuta: {subset.upper()}")
    print(f"Ciudades ({n}):")
    for i, name in enumerate(selected_names):
        print(f"  {i}: {name}")
    
    print("\n" + "-"*60)
    results = {}
    
    # Algoritmo Voraz
    if algorithm in ['greedy', 'all']:
        print("\nAlgoritmo Voraz (Vecino Más Cercano):")
        solver = GreedyTSP(selected_dist_matrix)
        tour, cost = solver.nearest_neighbor()
        improved_tour, improved_cost = solver.two_opt_improvement(tour)
        
        results['greedy'] = {
            'tour': improved_tour,
            'cost': improved_cost,
            'cities': [selected_names[i] for i in improved_tour[:-1]]
        }
        
        print(f"  Distancia: {improved_cost:.2f} km")
        print("  Ruta óptima:")
        for i in improved_tour[:-1]:
            print(f"    → {selected_names[i]}")
    
    # Algoritmo Genético
    if algorithm in ['ga', 'all'] and n <= 20:
        print("\nAlgoritmo Genético:")
        ga = GeneticAlgorithmTSP(
            selected_dist_matrix,
            population_size=100,
            generations=500,
            verbose=False
        )
        tour, cost, _ = ga.run()
        
        results['ga'] = {
            'tour': tour + [tour[0]],
            'cost': cost,
            'cities': [selected_names[i] for i in tour]
        }
        
        print(f"  Distancia: {cost:.2f} km")
    
    # Recocido Simulado
    if algorithm in ['sa', 'all']:
        print("\nRecocido Simulado:")
        sa = SimulatedAnnealingTSP(
            selected_dist_matrix,
            initial_temp=1000,
            final_temp=0.1,
            alpha=0.995,
            verbose=False
        )
        tour, cost, _ = sa.run()
        
        results['sa'] = {
            'tour': tour + [tour[0]],
            'cost': cost,
            'cities': [selected_names[i] for i in tour]
        }
        
        print(f"  Distancia: {cost:.2f} km")
    
    # Visualización en mapa
    visualize_costa_rica_route(selected_cities, results, selected_names)
    
    return results


def visualize_costa_rica_route(cities, results, names):
    """
    Visualiza las rutas en un mapa de Costa Rica
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 8))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (alg_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Extraer coordenadas
        lats = [city[0] for city in cities]
        lons = [city[1] for city in cities]
        
        # Dibujar Costa Rica (aproximado)
        # Límites aproximados del país
        ax.set_xlim(-85.8, -82.5)
        ax.set_ylim(8.0, 11.2)
        
        # Dibujar ciudades
        ax.scatter(lons, lats, c='red', s=100, zorder=5)
        
        # Etiquetas de ciudades
        for i, name in enumerate(names):
            ax.annotate(name.split('(')[0], (lons[i], lats[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
        
        # Dibujar tour
        tour = result['tour']
        for i in range(len(tour)-1):
            city1 = cities[tour[i]]
            city2 = cities[tour[i+1]]
            ax.plot([city1[1], city2[1]], [city1[0], city2[0]], 
                   'b-', alpha=0.6, linewidth=2)
            
            # Flecha de dirección
            mid_lon = (city1[1] + city2[1]) / 2
            mid_lat = (city1[0] + city2[0]) / 2
            dlon = city2[1] - city1[1]
            dlat = city2[0] - city1[0]
            ax.arrow(mid_lon - dlon*0.1, mid_lat - dlat*0.1,
                    dlon*0.2, dlat*0.2,
                    head_width=0.05, head_length=0.05,
                    fc='blue', ec='blue', alpha=0.5)
        
        # Ciudad inicial
        start = cities[tour[0]]
        ax.scatter(start[1], start[0], c='green', s=200, marker='s', 
                  zorder=6, label='Inicio')
        
        ax.set_title(f"{alg_name.upper()}\nDistancia: {result['cost']:.2f} km")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle("Optimización de Rutas en Costa Rica", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/costa_rica_tsp.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_costa_rica():
    """
    Demo completo con diferentes rutas
    """
    print("\n" + "="*60)
    print("DEMO: TSP CON CIUDADES DE COSTA RICA")
    print("="*60)
    
    # Test 1: Valle Central (8 ciudades)
    print("\n1. VALLE CENTRAL (San José y alrededores)")
    test_costa_rica_tsp('valle_central', 'all')
    
    # Test 2: Capitales de provincia (7 ciudades)
    print("\n2. CAPITALES DE PROVINCIA")
    test_costa_rica_tsp('principales', 'greedy')
    
    # Test 3: Ruta Pacífica
    print("\n3. RUTA PACÍFICA")
    test_costa_rica_tsp('pacifico', 'sa')
    
    print("\n" + "="*60)
    print("Demo completado! Ver results/costa_rica_tsp.png")
    print("="*60)


if __name__ == "__main__":
    # Ejecutar demo
    demo_costa_rica()
    
    # O prueba específica
    # test_costa_rica_tsp('valle_central', 'all')
    # test_costa_rica_tsp('principales', 'greedy')
    # test_costa_rica_tsp('completo', 'sa')