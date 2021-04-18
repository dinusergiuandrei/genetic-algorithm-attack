from bits import number_of_bits_for_int, find_number_of_bits
from functions import functions
from genetic_algorithm import genetic_algorithm
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time


# def ga_master_ind_len():
#     pop_size = 30
#     num_generations = 50
#     ind_bit_len = 10
#     tournament_size = 5
#
#     mutations = num_generations * pop_size * ind_bit_len
#     cx = num_generations * pop_size
#     selection = num_generations * pop_size * tournament_size
#     shuffle = num_generations * pop_size
#
#     print(mutations + cx + selection + shuffle)


def generate_random_ga_bits(bits_per_value, n_dimensions, pop_size, num_generations,
                            tournament_size, selection_pool_size, mutation_rate=0.05):
    init_bits = bits_per_value * n_dimensions * pop_size
    mutation_bits = bits_per_value * n_dimensions * pop_size
    cx_bits = selection_pool_size * 5 \
              + number_of_bits_for_int(n_dimensions * bits_per_value) * selection_pool_size // 2  # and cutting points
    selection_bits = (pop_size - 1) * tournament_size * 5

    generation_bits = mutation_bits + cx_bits + selection_bits
    total = init_bits + num_generations * generation_bits

    bit_groups = [np.random.choice(['0', '1'], init_bits)]
    for _ in range(num_generations):
        m = np.random.choice(['0', '1'], mutation_bits, p=[1 - mutation_rate, mutation_rate])
        cx = np.random.choice(['0', '1'], cx_bits)
        selection = np.random.choice(['0', '1'], selection_bits)

        bit_groups.append(m)
        bit_groups.append(cx)
        bit_groups.append(selection)

    return np.concatenate(bit_groups)


def run_ga(func):
    g_pop_size = 32
    g_tournament_size = 5
    g_selection_pool_size = 20
    g_num_generations = 50
    g_dimensions = 100
    num_workers = 4

    g_bits_per_value = find_number_of_bits(lower_bound=func.lower_bound, upper_bound=func.upper_bound)

    ga_bits = generate_random_ga_bits(g_bits_per_value, g_dimensions, g_pop_size, g_num_generations,
                                      g_tournament_size, g_selection_pool_size)

    h, best_score, solution, eval_times = genetic_algorithm(ga_bits, g_bits_per_value, func, g_dimensions, g_pop_size,
                                                            g_num_generations, g_selection_pool_size, g_tournament_size,
                                                            num_workers=num_workers)
    return eval_times


if __name__ == '__main__':
    times = run_ga(functions[0])