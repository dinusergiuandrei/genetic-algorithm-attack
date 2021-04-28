from small_ga.functions import functions
from small_ga.genetic_algorithm import genetic_algorithm
from small_ga.init import generate_random_ga_bits
from utils.bits import find_number_of_bits
from tqdm import tqdm
import numpy as np

from utils.parallel import search_number_of_workers


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


def meta_mutate(population, mutation_rate):
    return population


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


def meta_evaluate(population, small_ga_params, function, n_dimensions, num_workers, bits_per_value):
    scores = []
    for individual in population:
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
            num_workers=num_workers,
        )
        scores.append(ga_result['best_score'])
    return scores


def meta_tournament(population, scores, tournament_size):
    return population


class GAParams:
    def __init__(self, pop_size, num_generations, selection_pool_size,
                 mutation_rate, tournament_size):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.selection_pool_size = selection_pool_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size


def meta_genetic_algorithm(meta_ga_params, small_ga_params, function, n_dimensions,
                           num_workers, bits_per_value):
    meta_population = init_meta_population(
        meta_ga_params.pop_size,
        bits_per_value=bits_per_value,
        n_dimensions=n_dimensions,
        small_ga_params=small_ga_params)

    history = []

    global_best_meta_individual = None
    global_best_meta_score = -np.inf

    for _ in tqdm(np.arange(meta_ga_params.num_generations)):
        meta_population = meta_mutate(meta_population, meta_ga_params.mutation_rate)
        meta_population = meta_crossover(meta_population, meta_ga_params.selection_pool_size)
        scores = meta_evaluate(meta_population, small_ga_params, function, n_dimensions, num_workers, bits_per_value)
        history.append(scores)

        # update global best
        # best_index = scores.argmax()
        # generation_best_score = scores[best_index]
        # if generation_best_score > global_best_score:
        #     global_best_score = generation_best_score
        #     global_best_individual = np.array(population[best_index])

        meta_population = meta_tournament(meta_population, scores, meta_ga_params.tournament_size)

    return {
        'meta_history': np.array(history),
        'best_score': global_best_meta_score
    }


def run_meta_ga(function, n_dimensions, num_workers):
    meta_ga_params = GAParams(pop_size=30, num_generations=5, selection_pool_size=20,
                              mutation_rate=0.05, tournament_size=5)

    small_ga_params = GAParams(pop_size=30, num_generations=30, selection_pool_size=20,
                               mutation_rate=0.05, tournament_size=5)

    bits_per_value = find_number_of_bits(lower_bound=function.lower_bound,
                                         upper_bound=function.upper_bound)

    meta_genetic_algorithm(meta_ga_params, small_ga_params, function, n_dimensions, num_workers, bits_per_value)


def run_experiment():
    n_dimensions = 100

    num_workers = search_number_of_workers(functions[0], n_dimensions)
    response = input(f'Suggested number of workers: {num_workers}. (Y/<number>)')
    if response not in ['y', 'Y']:
        num_workers = int(response)

    for function in functions:
        run_meta_ga(function, n_dimensions, num_workers)


if __name__ == '__main__':
    run_meta_ga(functions[0], n_dimensions=100, num_workers=4)
