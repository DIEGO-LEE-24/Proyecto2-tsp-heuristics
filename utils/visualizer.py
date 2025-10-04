"""
Visualizador para problemas y soluciones de TSP
Incluye visualización estática y animada
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from typing import List, Tuple, Optional, Dict
import seaborn as sns


class TSPVisualizer:
    """Visualizador para TSP"""
    
    def __init__(self, cities: List[Tuple[float, float]]):
        """
        Inicializar visualizador
        
        Args:
            cities: Lista de coordenadas de ciudades
        """
        self.cities = cities
        self.n_cities = len(cities)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_cities(self, ax: Optional[plt.Axes] = None, 
                   title: str = "Cities Distribution",
                   show_labels: bool = True) -> plt.Axes:
        """
        Plotear solo las ciudades
        
        Args:
            ax: Axes de matplotlib
            title: Título del gráfico
            show_labels: Si mostrar etiquetas de ciudades
            
        Returns:
            Axes con el plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        
        # Plotear ciudades
        ax.scatter(x, y, c='red', s=200, alpha=0.6, edgecolors='darkred', 
                  linewidth=2, zorder=5)
        
        # Agregar etiquetas
        if show_labels and self.n_cities <= 30:
            for i, (xi, yi) in enumerate(zip(x, y)):
                ax.annotate(str(i), (xi, yi), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_tour(self, tour: List[int], 
                 ax: Optional[plt.Axes] = None,
                 title: str = "TSP Tour",
                 show_distance: bool = True,
                 show_labels: bool = True,
                 color: str = 'blue',
                 style: str = 'solid') -> plt.Axes:
        """
        Plotear un tour completo
        
        Args:
            tour: Lista con el orden de visita
            ax: Axes de matplotlib
            title: Título del gráfico
            show_distance: Si mostrar distancia total
            show_labels: Si mostrar etiquetas
            color: Color de las líneas
            style: Estilo de línea
            
        Returns:
            Axes con el plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotear ciudades primero
        self.plot_cities(ax, title="", show_labels=show_labels)
        
        # Asegurar que el tour esté cerrado
        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]
        
        # Plotear las conexiones
        for i in range(len(tour) - 1):
            city1 = self.cities[tour[i]]
            city2 = self.cities[tour[i + 1]]
            
            ax.plot([city1[0], city2[0]], [city1[1], city2[1]], 
                   color=color, linestyle=style, linewidth=2, 
                   alpha=0.6, zorder=1)
            
            # Agregar flecha para mostrar dirección
            if self.n_cities <= 20:
                mid_x = (city1[0] + city2[0]) / 2
                mid_y = (city1[1] + city2[1]) / 2
                dx = city2[0] - city1[0]
                dy = city2[1] - city1[1]
                ax.arrow(mid_x - dx*0.1, mid_y - dy*0.1, 
                        dx*0.2, dy*0.2,
                        head_width=1.5, head_length=1, 
                        fc=color, ec=color, alpha=0.5, zorder=2)
        
        # Marcar ciudad inicial
        start_city = self.cities[tour[0]]
        ax.scatter(start_city[0], start_city[1], 
                  c='green', s=300, marker='s', 
                  edgecolors='darkgreen', linewidth=3, 
                  zorder=6, label='Start')
        
        # Calcular y mostrar distancia total
        if show_distance:
            total_distance = self.calculate_tour_distance(tour)
            title += f"\nTotal Distance: {total_distance:.2f}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        
        return ax
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """
        Calcular distancia total de un tour
        
        Args:
            tour: Lista con el orden de visita
            
        Returns:
            Distancia total
        """
        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]
        
        distance = 0
        for i in range(len(tour) - 1):
            city1 = self.cities[tour[i]]
            city2 = self.cities[tour[i + 1]]
            distance += np.sqrt((city1[0] - city2[0])**2 + 
                              (city1[1] - city2[1])**2)
        
        return distance
    
    def compare_tours(self, tours: Dict[str, List[int]], 
                     title: str = "Tour Comparison") -> plt.Figure:
        """
        Comparar múltiples tours lado a lado
        
        Args:
            tours: Diccionario {nombre: tour}
            title: Título principal
            
        Returns:
            Figura con la comparación
        """
        n_tours = len(tours)
        cols = min(3, n_tours)
        rows = (n_tours + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_tours == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = sns.color_palette("husl", n_tours)
        
        for idx, (name, tour) in enumerate(tours.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            self.plot_tour(tour, ax=ax, 
                          title=f"{name}\nDistance: {self.calculate_tour_distance(tour):.2f}",
                          color=colors[idx], 
                          show_labels=False)
        
        # Ocultar axes vacíos
        for idx in range(n_tours, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_convergence(self, histories: Dict[str, List[float]], 
                        title: str = "Algorithm Convergence",
                        log_scale: bool = False) -> plt.Figure:
        """
        Plotear historiales de convergencia
        
        Args:
            histories: Diccionario {algoritmo: historial}
            title: Título del gráfico
            log_scale: Si usar escala logarítmica
            
        Returns:
            Figura con el gráfico
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(histories))
        
        for (name, history), color in zip(histories.items(), colors):
            ax.plot(history, label=name, color=color, linewidth=2)
        
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Distance Found", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        return fig
    
    def animate_tour_construction(self, tour: List[int], 
                                 interval: int = 500,
                                 save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Animar la construcción de un tour
        
        Args:
            tour: Tour a animar
            interval: Intervalo entre frames (ms)
            save_path: Ruta para guardar la animación
            
        Returns:
            Objeto de animación
        """
        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotear todas las ciudades
        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        ax.scatter(x, y, c='red', s=200, alpha=0.6, 
                  edgecolors='darkred', linewidth=2, zorder=5)
        
        # Inicializar línea vacía
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6)
        
        # Configurar límites
        margin = 10
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, alpha=0.3)
        
        # Texto para distancia
        distance_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            line.set_data([], [])
            distance_text.set_text('')
            return line, distance_text
        
        def animate(frame):
            if frame == 0:
                line.set_data([], [])
                distance_text.set_text('Starting tour construction...')
            else:
                # Construir el tour hasta el frame actual
                tour_segment = tour[:frame+1]
                x_data = [self.cities[city][0] for city in tour_segment]
                y_data = [self.cities[city][1] for city in tour_segment]
                
                line.set_data(x_data, y_data)
                
                # Calcular distancia parcial
                partial_distance = 0
                for i in range(len(tour_segment) - 1):
                    city1 = self.cities[tour_segment[i]]
                    city2 = self.cities[tour_segment[i + 1]]
                    partial_distance += np.sqrt((city1[0] - city2[0])**2 + 
                                              (city1[1] - city2[1])**2)
                
                distance_text.set_text(f'Cities visited: {frame}/{len(tour)-1}\n'
                                     f'Distance: {partial_distance:.2f}')
                
                # Actualizar título
                if frame == len(tour) - 1:
                    ax.set_title(f"Tour Complete! Total Distance: {partial_distance:.2f}", 
                               fontsize=14, fontweight='bold')
                else:
                    ax.set_title(f"Constructing Tour... ({frame}/{len(tour)-1})", 
                               fontsize=14)
            
            return line, distance_text
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(tour), interval=interval,
                                     blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved to: {save_path}")
        
        return anim
    
    def plot_distance_matrix(self, dist_matrix: Optional[np.ndarray] = None,
                            title: str = "Distance Matrix Heatmap") -> plt.Figure:
        """
        Plotear matriz de distancias como heatmap
        
        Args:
            dist_matrix: Matriz de distancias (si None, la calcula)
            title: Título del gráfico
            
        Returns:
            Figura con el heatmap
        """
        if dist_matrix is None:
            # Calcular matriz de distancias
            n = self.n_cities
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        city1 = self.cities[i]
                        city2 = self.cities[j]
                        dist_matrix[i][j] = np.sqrt((city1[0] - city2[0])**2 + 
                                                   (city1[1] - city2[1])**2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Crear heatmap
        im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        
        # Agregar colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Distance', rotation=270, labelpad=20)
        
        # Configurar ticks
        if self.n_cities <= 30:
            ax.set_xticks(np.arange(self.n_cities))
            ax.set_yticks(np.arange(self.n_cities))
            ax.set_xticklabels(np.arange(self.n_cities))
            ax.set_yticklabels(np.arange(self.n_cities))
            
            # Rotar labels del eje x
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("City Index")
        ax.set_ylabel("City Index")
        
        plt.tight_layout()
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict[str, Dict],
                                 metrics: List[str] = ['cost', 'time']) -> plt.Figure:
        """
        Comparar métricas de diferentes algoritmos
        
        Args:
            results: Diccionario con resultados de algoritmos
            metrics: Métricas a comparar
            
        Returns:
            Figura con la comparación
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        algorithms = list(results.keys())
        colors = sns.color_palette("husl", len(algorithms))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results[alg].get(metric, 0) for alg in algorithms]
            
            bars = ax.bar(algorithms, values, color=colors)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}' if metric == 'cost' else f'{value:.4f}',
                       ha='center', va='bottom')
            
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel('Algorithm')
            
            # Rotar labels si son muchos
            if len(algorithms) > 4:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        fig.suptitle("Algorithm Performance Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig