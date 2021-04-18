import numpy as np
from tqdm import tqdm

from bits import number_of_bits_for_int, bits_to_number
import multiprocessing as mp
from time import time

def initialize_population(pop_size, init_bits):
    return init_bits.reshape(pop_size, -1)


def mutate_population(population, mutation_bits):
    bits_per_individual = len(population[0])
    for individual_index in range(len(population)):
        individual = population[individual_index]
        individual_mutation_bits = mutation_bits[(individual_index * bits_per_individual):(
                (individual_index + 1) * bits_per_individual)]
        individual_ints = np.array(list(map(int, individual)))
        for bit_index in range(len(individual_ints)):
            if individual_mutation_bits[bit_index] == '1':
                individual_ints[bit_index] = 1 - individual_ints[bit_index]
        individual_bitstring = individual_ints.astype(str)  # ''.join(list(map(str, individual_ints)))
        population[individual_index] = individual_bitstring
    return population


def cross_over_population(population, cross_over_bits, selection_pool_size, n_dimensions, bits_per_value):
    pairs_bits = cross_over_bits[:selection_pool_size * 5]
    bit_positions = pairs_bits.reshape(selection_pool_size, 5)  # 5 is the number of bits for 0..31
    int_positions = np.array([int(''.join(bit_position), 2) for bit_position in bit_positions])  # duplicates

    population = list(population)

    selection_pool = []

    for p in int_positions:
        p = min(p, len(population) - 1)
        selection_pool.append(population[p])
        del population[p]

    new_population = list(population)

    num_cutting_points_bits = number_of_bits_for_int(n_dimensions * bits_per_value) * selection_pool_size // 2
    cutting_points_bits = cross_over_bits[selection_pool_size * 5:][:num_cutting_points_bits] \
        .reshape((selection_pool_size // 2, -1))
    cutting_points = np.array([int(''.join(bit_position), 2) for bit_position in cutting_points_bits])

    for pair_index in range(len(cutting_points)):
        parent1 = selection_pool[2 * pair_index]
        parent2 = selection_pool[2 * pair_index + 1]

        cutting_point = cutting_points[pair_index]

        cutting_point = max(1, cutting_point)
        cutting_point = min(cutting_point, len(parent1) - 2)

        child1 = np.concatenate([parent1[:cutting_point], parent2[cutting_point:]])
        child2 = np.concatenate([parent2[:cutting_point], parent1[cutting_point:]])
        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population)


def individual_to_values(individual, function, n_dimensions):
    bit_groups = np.array(individual).reshape(n_dimensions, -1)
    values = [bits_to_number(''.join(bits), function.lower_bound, function.upper_bound) for bits in bit_groups]
    return np.array(values)


def eval_individual(individual, function, n_dimensions):
    return -function.eval(individual_to_values(individual, function, n_dimensions))


def evaluate_population(population, function, n_dimensions, pool):
    scores = []
    for individual in population:
        score = pool.apply_async(eval_individual, args=(individual, function, n_dimensions))
        scores.append(score)
    scores = [score.get() for score in scores]
    return np.array(scores)


def select_next_generation(population, global_best_individual, tournament_size, scores, selection_bits):
    new_population = [np.array(global_best_individual)]  # elitism 1
    while len(new_population) < len(population):
        tournament_bits = selection_bits[:5 * tournament_size]
        bit_positions = tournament_bits.reshape(tournament_size, 5)
        int_positions = np.array([int(''.join(bit_position), 2) for bit_position in bit_positions])
        tournament_scores = np.array([scores[p] for p in int_positions])

        winner_position = int_positions[tournament_scores.argmax()]
        winner = population[winner_position]
        new_population.append(np.array(winner))
    return new_population


# ga_bits must be a list of chars, not a string
def genetic_algorithm(genetic_algorithm_bits, bits_per_value, function, n_dimensions,
                      pop_size, num_generations, selection_pool_size, tournament_size, num_workers=8):
    num_init_bits = bits_per_value * n_dimensions * pop_size
    init_bits = genetic_algorithm_bits[:num_init_bits]
    population = initialize_population(pop_size, init_bits=init_bits)
    genetic_algorithm_bits = genetic_algorithm_bits[num_init_bits:]

    global_best_individual = None
    global_best_score = -np.inf
    history = []
    pool = mp.Pool(processes=num_workers)
    eval_times = []
    for generation in tqdm(np.arange(num_generations), leave=True):
        # mutation

        num_mutation_bits = bits_per_value * n_dimensions * pop_size
        mutation_bits = genetic_algorithm_bits[:num_mutation_bits]
        population = mutate_population(population, mutation_bits=mutation_bits)
        genetic_algorithm_bits = genetic_algorithm_bits[num_mutation_bits:]

        # cross-over

        num_cx_bits = selection_pool_size * 5 + number_of_bits_for_int(n_dimensions * bits_per_value) * selection_pool_size // 2
        cross_over_bits = genetic_algorithm_bits[:num_cx_bits]
        population = cross_over_population(population, cross_over_bits, selection_pool_size, n_dimensions, bits_per_value)
        genetic_algorithm_bits = genetic_algorithm_bits[num_cx_bits:]

        start_time = time()
        scores = evaluate_population(population, function, n_dimensions, pool)
        eval_time = time() - start_time
        eval_times.append(eval_time)

        # update global best
        best_index = scores.argmax()
        generation_best_score = scores[best_index]
        if generation_best_score > global_best_score:
            global_best_score = generation_best_score
            global_best_individual = np.array(population[best_index])

        # tournament selection
        num_selection_bits = (pop_size - 1) * tournament_size * 5  # 5 is the number of bits for the position 0..31
        selection_bits = genetic_algorithm_bits[:num_selection_bits]
        population = select_next_generation(population, global_best_individual, tournament_size, scores, selection_bits)
        genetic_algorithm_bits = genetic_algorithm_bits[num_selection_bits:]
        history.append(scores)
    pool.close()

    return np.array(history), -global_best_score, \
           individual_to_values(global_best_individual, function, n_dimensions), eval_times
