from small_ga.functions import functions
from small_ga.genetic_algorithm import genetic_algorithm
from small_ga.init import generate_random_ga_bits
from utils.bits import find_number_of_bits
from tqdm import tqdm
import numpy as np
import os
import multiprocessing as mp

from utils.parallel import search_number_of_workers
import pickle


def save_pickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def init_meta_population(pop_size, bits_per_value, n_dimensions,
                         small_ga_params):
    meta_population = []
    for _ in range(pop_size):
        meta_individual = generate_random_ga_bits(
            bits_per_value=bits_per_value,
            n_dimensions=n_dimensions,
            pop_size=small_ga_params.pop_size,
            num_generations=small_ga_params.num_generations,
            tournament_size=small_ga_params.tournament_size,
            selection_pool_size=small_ga_params.selection_pool_size,
            mutation_rate=small_ga_params.mutation_rate
        )
        meta_population.append(meta_individual)
    return np.array(meta_population, dtype='int')


def meta_mutate(population, mutation_rate, small_pop_size, bits_per_value, n_dimensions,
                small_tournament_size, small_selection_pool_size, small_mutation_rate):
    num_init_bits = bits_per_value * n_dimensions * small_pop_size
    num_mutation_bits = int(num_init_bits * small_mutation_rate)
    num_selection_bits = (small_pop_size - 1) * small_tournament_size
    num_cx_bits = int(small_selection_pool_size * 3 // 2)

    for individual_index in range(len(population)):
        individual = population[individual_index]
        for bit_index in range(len(individual)):
            if np.random.uniform(0, 1) < mutation_rate:
                p = bit_index
                if p < num_init_bits:
                    v = population[individual_index][bit_index]
                    population[individual_index][bit_index] = 1 - v
                else:
                    num_generation_bits = num_mutation_bits + num_cx_bits + num_selection_bits
                    p = (p - num_init_bits) % num_generation_bits
                    if p < num_mutation_bits:
                        population[individual_index][bit_index] = np.random.randint(0, num_init_bits)
                    elif p < num_mutation_bits + small_selection_pool_size:
                        max_v = p - num_mutation_bits
                        population[individual_index][bit_index] = np.random.randint(0, small_pop_size - max_v)
                    elif p < num_mutation_bits + small_selection_pool_size * 3 // 2:
                        population[individual_index][bit_index] = np.random.randint(1,
                                                                                    bits_per_value * n_dimensions - 1)
                    else:
                        population[individual_index][bit_index] = np.random.randint(0, small_pop_size)
    return np.array(population, dtype='int')


def meta_crossover(population, selection_pool_size):
    selection_pool = []

    population = list(population)
    while len(selection_pool) < selection_pool_size:
        selected_index = np.random.randint(0, len(population))
        selection_pool.append(population[selected_index])
        del population[selected_index]

    new_population = list(population)

    cutting_points = np.random.randint(1, len(population[0]) - 1, selection_pool_size // 2)
    for pair_index in range(selection_pool_size // 2):
        cutting_point = cutting_points[pair_index]
        parent1 = selection_pool[2 * pair_index]
        parent2 = selection_pool[2 * pair_index + 1]

        child1 = np.concatenate([parent1[:cutting_point], parent2[cutting_point:]])
        child2 = np.concatenate([parent2[:cutting_point], parent1[cutting_point:]])
        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population, dtype='int')


def meta_evaluate(population, small_ga_params, function, n_dimensions,
                  bits_per_value, meta_intention, min_possible_mutation_rate,
                  max_possible_mutation_rate, max_plateau_stop, pool):
    scores = []
    all_histories = []
    all_final_populations = []

    num_init_bits = bits_per_value * n_dimensions * small_ga_params.pop_size
    num_mutation_bits = int(num_init_bits * small_ga_params.mutation_rate)
    num_selection_bits = (small_ga_params.pop_size - 1) * small_ga_params.tournament_size
    num_cx_bits = int(small_ga_params.selection_pool_size * 3 // 2)
    num_generation_bits = num_mutation_bits + num_cx_bits + num_selection_bits

    for individual in population:
        generational_bits = np.array(individual[num_init_bits:])
        mutation_rates = []
        for _ in range(small_ga_params.num_generations):
            mutation_bits = generational_bits[:num_mutation_bits]
            this_mutation_rate = len(np.unique(mutation_bits)) / num_init_bits
            mutation_rates.append(this_mutation_rate)
            generational_bits = generational_bits[num_generation_bits:]
        real_mutation_rate = np.mean(mutation_rates)
        invalid_mutation_rate = real_mutation_rate > max_possible_mutation_rate or \
                                real_mutation_rate < min_possible_mutation_rate

        if invalid_mutation_rate:
            scores.append(np.inf)
            all_histories.append(None)
            all_final_populations.append(None)
        else:
            ga_result = genetic_algorithm(
                genetic_algorithm_bits=individual,
                bits_per_value=bits_per_value,
                function=function,
                n_dimensions=n_dimensions,
                pop_size=small_ga_params.pop_size,
                num_generations=small_ga_params.num_generations,
                selection_pool_size=small_ga_params.selection_pool_size,
                mutation_rate=small_ga_params.mutation_rate,
                tournament_size=small_ga_params.tournament_size,
                pool=pool,
                max_plateau_stop=max_plateau_stop
            )
            all_histories.append(ga_result['history'])
            all_final_populations.append(ga_result['final_population'])
            ga_result['best_score'] = ga_result['best_score'] * meta_intention
            scores.append(ga_result['best_score'])
    return np.array(scores), np.array(all_histories, dtype=object), np.array(all_final_populations, dtype=object)


def meta_tournament(population, global_best_individual, scores, tournament_size):
    new_population = [global_best_individual]
    while len(new_population) < len(population):
        positions = np.random.randint(0, len(population), tournament_size)
        tournament_scores = np.array([scores[p] for p in positions])

        winner_position = positions[tournament_scores.argmin()]
        winner = population[winner_position]
        new_population.append(np.array(winner))
    return np.array(new_population, dtype='int')


class GAParams:
    def __init__(self, pop_size, num_generations, selection_pool_size,
                 mutation_rate, tournament_size):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.selection_pool_size = selection_pool_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size


def meta_genetic_algorithm(meta_ga_params, small_ga_params, function, n_dimensions, bits_per_value,
                           meta_intention, min_possible_mutation_rate, max_possible_mutation_rate,
                           small_max_plateau_stop, pool):
    intention_string = {1: 'help', -1: 'attack'}[meta_intention]
    ga_folder = os.path.join('results', f'{function.name}_{intention_string}_{bits_per_value}_bits')
    if not os.path.exists(ga_folder):
        os.mkdir(ga_folder)

    meta_population = init_meta_population(
        meta_ga_params.pop_size,
        bits_per_value=bits_per_value,
        n_dimensions=n_dimensions,
        small_ga_params=small_ga_params)

    global_best_meta_individual = None
    global_best_meta_score = np.inf

    for generation in tqdm(np.arange(meta_ga_params.num_generations)):
        meta_population = meta_mutate(meta_population, meta_ga_params.mutation_rate, small_ga_params.pop_size,
                                      bits_per_value, n_dimensions, small_ga_params.tournament_size,
                                      small_ga_params.selection_pool_size, small_ga_params.mutation_rate)
        meta_population = meta_crossover(meta_population, meta_ga_params.selection_pool_size)
        scores, all_histories, all_final_populations = meta_evaluate(
            meta_population, small_ga_params, function, n_dimensions,
            bits_per_value, meta_intention, min_possible_mutation_rate,
            max_possible_mutation_rate, small_max_plateau_stop, pool=pool)

        # update global best

        best_index = scores.argmin()
        generation_best_score = scores[best_index]
        if generation_best_score < global_best_meta_score:
            global_best_meta_score = generation_best_score
            global_best_meta_individual = np.array(meta_population[best_index])

        # save scores, all_results and meta_population
        generation_path = os.path.join(ga_folder, str(generation))
        if not os.path.exists(generation_path):
            os.mkdir(generation_path)

        np.save(os.path.join(generation_path, 'scores.npy'), scores)
        np.save(os.path.join(generation_path, 'best_meta_individual.npy'), global_best_meta_individual)
        np.save(os.path.join(generation_path, 'small_ga_run_history.npy'), all_histories)
        np.save(os.path.join(generation_path, 'small_ga_final_populations.npy'), all_final_populations)

        meta_population = meta_tournament(meta_population, global_best_meta_individual, scores,
                                          meta_ga_params.tournament_size)


def prepare_experiment():
    if not os.path.exists('results'):
        os.mkdir('results')


def run_experiment():
    prepare_experiment()
    n_dimensions = 20
    min_possible_mutation_rate = 0.001
    max_possible_mutation_rate = 0.07
    max_plateau_stop = 100
    tolerance = 1e-6

    meta_ga_params = GAParams(pop_size=30, num_generations=300, selection_pool_size=20,
                              mutation_rate=0.01, tournament_size=5)

    small_ga_params = GAParams(pop_size=30, num_generations=3000, selection_pool_size=20,
                               mutation_rate=0.01, tournament_size=5)

    num_workers = search_number_of_workers(functions[0], n_dimensions)

    response = input(f'Suggested number of workers: {num_workers}. (Y/<number>)')
    if response not in ['y', 'Y']:
        num_workers = int(response)

    # num_workers = 4
    pool = mp.Pool(processes=num_workers)

    for function in functions:
        lb, ub = function.lower_bound, function.upper_bound
        bits_per_value = find_number_of_bits(lower_bound=lb, upper_bound=ub, tolerance=tolerance)
        for intention in [1, -1]:
            print(function.name, bits_per_value, {1: 'help', -1: 'attack'}[intention])
            if function == functions[0] and intention == 1:
                continue
            meta_genetic_algorithm(
                meta_ga_params, small_ga_params, function, n_dimensions,
                bits_per_value, intention, min_possible_mutation_rate, max_possible_mutation_rate,
                small_max_plateau_stop=max_plateau_stop, pool=pool)

    pool.close()


if __name__ == '__main__':
    run_experiment()
