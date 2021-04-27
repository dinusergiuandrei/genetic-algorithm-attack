from bits import number_of_bits_for_int, find_number_of_bits
from functions import functions
from genetic_algorithm import genetic_algorithm
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from plotter import plot_per_generation


def generate_random_ga_bits(bits_per_value, n_dimensions, pop_size, num_generations,
                            tournament_size, selection_pool_size, mutation_rate=0.05):
    init_bits = bits_per_value * n_dimensions * pop_size
    state_bits = init_bits
    mutation_bits = int(state_bits * mutation_rate)
    selection_bits = (pop_size - 1) * tournament_size

    bit_groups = [np.random.choice([0, 1], init_bits)]
    for _ in range(num_generations):
        m = np.random.randint(0, state_bits, mutation_bits)

        cx = []
        for _ in range(selection_pool_size):
            cx.append(np.random.randint(0, pop_size - len(cx)))
        for _ in range(selection_pool_size // 2):
            cx.append(np.random.randint(1, bits_per_value * n_dimensions - 1))
        cx = np.array(cx, dtype=np.object)

        selection = np.random.randint(0, pop_size, selection_bits)

        bit_groups.append(m)
        bit_groups.append(cx)
        bit_groups.append(selection)

    return np.concatenate(bit_groups)


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
