from utils.bits import find_number_of_bits
from small_ga.functions import functions
from small_ga.genetic_algorithm import genetic_algorithm
import numpy as np

from utils.plotter import plot_per_generation
from small_ga.init import generate_random_ga_bits


def run_ga(func, n_dim):
    g_pop_size = 30
    g_tournament_size = 5
    g_selection_pool_size = 20
    g_num_generations = 50
    g_dimensions = n_dim  # 100
    num_workers = 4
    mutation_rate = 0.05

    g_bits_per_value = find_number_of_bits(lower_bound=func.lower_bound, upper_bound=func.upper_bound)

    ga_bits = generate_random_ga_bits(g_bits_per_value, g_dimensions, g_pop_size, g_num_generations,
                                      g_tournament_size, g_selection_pool_size, mutation_rate=mutation_rate)

    result = genetic_algorithm(ga_bits, g_bits_per_value, func, g_dimensions,
                               g_pop_size, g_num_generations, g_selection_pool_size,
                               mutation_rate, g_tournament_size, num_workers=num_workers)
    return result


if __name__ == '__main__':
    for f in functions:
        r = run_ga(f, n_dim=100)
        np.save(f'history/{f.name}.npy', r['history'])
        print(f.name, r['best_score'])
    plot_per_generation()
