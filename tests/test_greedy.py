"""
Unit tests for Greedy algorithms
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.greedy import GreedyTSP
from utils.data_generator import TSPDataGenerator


class TestGreedyTSP:
    """Test cases for Greedy TSP algorithms"""
    
    @pytest.fixture
    def small_distance_matrix(self):
        """Create a small test distance matrix"""
        # 4 cities in a square
        return np.array([
            [0, 1, 1.414, 1],
            [1, 0, 1, 1.414],
            [1.414, 1, 0, 1],
            [1, 1.414, 1, 0]
        ])
    
    @pytest.fixture
    def random_distance_matrix(self):
        """Create a random distance matrix"""
        generator = TSPDataGenerator(seed=42)
        cities = generator.generate_random_cities(10)
        return generator.calculate_distance_matrix(cities)
    
    def test_nearest_neighbor_basic(self, small_distance_matrix):
        """Test nearest neighbor on a simple case"""
        solver = GreedyTSP(small_distance_matrix)
        tour, cost = solver.nearest_neighbor(start_city=0)
        
        # Check that tour is valid
        assert len(tour) == 5  # 4 cities + return to start
        assert tour[0] == tour[-1] == 0  # Starts and ends at city 0
        assert set(tour[:-1]) == {0, 1, 2, 3}  # Visits all cities
        
        # Check that cost is positive
        assert cost > 0
    
    def test_nearest_neighbor_all_starts(self, random_distance_matrix):
        """Test that different starting cities give valid tours"""
        solver = GreedyTSP(random_distance_matrix)
        n_cities = len(random_distance_matrix)
        
        for start in range(n_cities):
            tour, cost = solver.nearest_neighbor(start_city=start)
            
            # Validate tour
            assert len(tour) == n_cities + 1
            assert tour[0] == tour[-1] == start
            assert len(set(tour[:-1])) == n_cities
            assert cost > 0
    
    def test_cheapest_insertion(self, random_distance_matrix):
        """Test cheapest insertion algorithm"""
        solver = GreedyTSP(random_distance_matrix)
        tour, cost = solver.cheapest_insertion()
        
        n_cities = len(random_distance_matrix)
        
        # Validate tour
        assert len(tour) == n_cities + 1
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == n_cities
        assert cost > 0
    
    def test_two_opt_improvement(self, random_distance_matrix):
        """Test that 2-opt improves or maintains the solution"""
        solver = GreedyTSP(random_distance_matrix)
        
        # Get initial solution
        initial_tour, initial_cost = solver.nearest_neighbor()
        
        # Apply 2-opt
        improved_tour, improved_cost = solver.two_opt_improvement(initial_tour)
        
        # Check improvement
        assert improved_cost <= initial_cost
        
        # Validate improved tour
        assert len(improved_tour) == len(initial_tour)
        assert set(improved_tour[:-1]) == set(initial_tour[:-1])
    
    def test_three_opt_improvement(self, small_distance_matrix):
        """Test 3-opt improvement"""
        solver = GreedyTSP(small_distance_matrix)
        
        # Get initial solution
        initial_tour, initial_cost = solver.nearest_neighbor()
        
        # Apply 3-opt
        improved_tour, improved_cost = solver.three_opt_improvement(initial_tour, max_iterations=10)
        
        # Check that cost doesn't increase
        assert improved_cost <= initial_cost
        
        # Validate tour
        assert len(improved_tour) == len(initial_tour)
    
    def test_multiple_start_nearest_neighbor(self, random_distance_matrix):
        """Test multiple start nearest neighbor"""
        solver = GreedyTSP(random_distance_matrix)
        
        # Get best solution from multiple starts
        best_tour, best_cost = solver.multiple_start_nearest_neighbor()
        
        # Compare with single start
        single_tour, single_cost = solver.nearest_neighbor(0)
        
        # Best should be at least as good as any single start
        assert best_cost <= single_cost
    
    def test_savings_algorithm(self, random_distance_matrix):
        """Test Clarke-Wright savings algorithm"""
        solver = GreedyTSP(random_distance_matrix)
        tour, cost = solver.savings_algorithm()
        
        n_cities = len(random_distance_matrix)
        
        # Validate tour
        assert len(tour) == n_cities + 1
        assert tour[0] == tour[-1]
        assert len(set(tour[:-1])) == n_cities
        assert cost > 0
    
    def test_tour_cost_calculation(self, small_distance_matrix):
        """Test tour cost calculation"""
        solver = GreedyTSP(small_distance_matrix)
        
        # Known tour with known cost
        tour = [0, 1, 2, 3, 0]
        expected_cost = 1 + 1 + 1 + 1  # Square perimeter
        
        calculated_cost = solver._calculate_tour_cost(tour)
        
        assert abs(calculated_cost - expected_cost) < 0.001
    
    def test_deterministic_results(self):
        """Test that results are deterministic with same seed"""
        generator1 = TSPDataGenerator(seed=123)
        cities1 = generator1.generate_random_cities(15)
        dist1 = generator1.calculate_distance_matrix(cities1)
        
        generator2 = TSPDataGenerator(seed=123)
        cities2 = generator2.generate_random_cities(15)
        dist2 = generator2.calculate_distance_matrix(cities2)
        
        solver1 = GreedyTSP(dist1)
        solver2 = GreedyTSP(dist2)
        
        tour1, cost1 = solver1.nearest_neighbor(0)
        tour2, cost2 = solver2.nearest_neighbor(0)
        
        assert tour1 == tour2
        assert cost1 == cost2
    
    @pytest.mark.parametrize("n_cities", [5, 10, 20, 30])
    def test_scalability(self, n_cities):
        """Test algorithms with different problem sizes"""
        generator = TSPDataGenerator(seed=42)
        cities = generator.generate_random_cities(n_cities)
        dist_matrix = generator.calculate_distance_matrix(cities)
        
        solver = GreedyTSP(dist_matrix)
        
        # Test all algorithms
        algorithms = [
            solver.nearest_neighbor,
            solver.cheapest_insertion,
            solver.multiple_start_nearest_neighbor,
            solver.savings_algorithm
        ]
        
        for algo in algorithms:
            tour, cost = algo()
            
            # Basic validation
            assert len(set(tour[:-1])) == n_cities
            assert cost > 0
            
            # Apply improvement
            improved_tour, improved_cost = solver.two_opt_improvement(tour)
            assert improved_cost <= cost
    
    def test_empty_or_single_city(self):
        """Test edge cases with 0 or 1 city"""
        # Empty matrix
        empty_matrix = np.array([])
        with pytest.raises((ValueError, IndexError)):
            solver = GreedyTSP(empty_matrix)
            solver.nearest_neighbor()
        
        # Single city
        single_matrix = np.array([[0]])
        solver = GreedyTSP(single_matrix)
        tour, cost = solver.nearest_neighbor()
        assert tour == [0, 0]
        assert cost == 0
    
    def test_asymmetric_matrix(self):
        """Test with asymmetric distance matrix"""
        # Create asymmetric matrix
        matrix = np.array([
            [0, 1, 2, 3],
            [2, 0, 1, 2],
            [3, 2, 0, 1],
            [1, 3, 2, 0]
        ])
        
        solver = GreedyTSP(matrix)
        tour, cost = solver.nearest_neighbor()
        
        # Calculate cost manually to verify
        manual_cost = sum(matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        assert abs(cost - manual_cost) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])