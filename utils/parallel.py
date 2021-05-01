import multiprocessing

import numpy as np

from utils.bits import find_number_of_bits
from small_ga.functions import functions
from small_ga.genetic_algorithm import genetic_algorithm
from main import generate_random_ga_bits
from time import time

def search_number_of_workers(func, dims):
    g_pop_size = 30
    g_tournament_size = 5
    g_selection_pool_size = 20
    g_num_generations = 100
    g_dimensions = dims  # 100

    best_num_workers = None
    lowest_time = None

    g_bits_per_value = find_number_of_bits(lower_bound=func.lower_bound, upper_bound=func.upper_bound)

    ga_bits = generate_random_ga_bits(g_bits_per_value, g_dimensions, g_pop_size, g_num_generations,
                                      g_tournament_size, g_selection_pool_size)

    for num_workers in np.arange(1, multiprocessing.cpu_count() * 2 + 1):
        print('Workers: ', num_workers)
        start_time = time()
        ga_result = genetic_algorithm(ga_bits, g_bits_per_value, func, g_dimensions,
                                    g_pop_size, g_num_generations, g_selection_pool_size,
                                    0.05, g_tournament_size,num_workers=num_workers)
        avg_eval_time = time() - start_time
        if lowest_time is None or avg_eval_time < lowest_time:
            lowest_time = avg_eval_time
            best_num_workers = num_workers
    return best_num_workers


if __name__ == '__main__':
    best_num_workers = search_number_of_workers(functions[0], 100)
    print('Optimal number of workers: ', best_num_workers)
