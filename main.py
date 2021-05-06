import pandas as pd

from utils.bits import find_number_of_bits
from small_ga.functions import functions
from small_ga.genetic_algorithm import genetic_algorithm
import numpy as np

from utils.plotter import plot_per_generation
from small_ga.init import generate_random_ga_bits

import multiprocessing as mp

def run_ga(func, n_dim, tol=1e-4, mutation_rate=0.01):
    g_pop_size = 30
    g_tournament_size = 5
    g_selection_pool_size = 20
    g_num_generations = 3000
    max_plateau_stop = 100
    g_dimensions = n_dim
    num_workers = 4

    g_bits_per_value = find_number_of_bits(lower_bound=func.lower_bound, upper_bound=func.upper_bound, tolerance=tol)

    ga_bits = generate_random_ga_bits(g_bits_per_value, g_dimensions, g_pop_size, g_num_generations,
                                      g_tournament_size, g_selection_pool_size, mutation_rate=mutation_rate)
    pool = mp.Pool(processes=num_workers)
    result = genetic_algorithm(ga_bits, g_bits_per_value, func, g_dimensions,
                               g_pop_size, g_num_generations, g_selection_pool_size,
                               mutation_rate, g_tournament_size,
                               max_plateau_stop=max_plateau_stop, verbose=True, pool=pool)
    pool.close()
    return result

from time import time

if __name__ == '__main__':
    n_dim = 20
    mutation_rate = 0.01
    tol = 1e-6

    results = []
    tests = 3
    start_time = time()

    for f in functions:
        for _ in range(tests):
            _ = run_ga(f, n_dim=n_dim, tol=tol, mutation_rate=mutation_rate)

    total_time = (time() - start_time) / tests
    time_per_generation = total_time * 2 * 30
    print('Expected time per generation: ', time_per_generation)
    days_per_generation = time_per_generation / 3600 / 24
    suggested_generations = 10 / days_per_generation
    print('Suggested number of generations: ', suggested_generations)
    # plot_per_generation()

