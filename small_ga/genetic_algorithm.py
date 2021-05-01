import numpy as np

from utils.bits import bits_to_number
import multiprocessing as mp


def initialize_population(pop_size, init_bits):
    return init_bits.reshape(pop_size, -1)


def mutate_population(population, mutation_bits):
    original_population_shape = population.shape
    flat_population = population.flatten()
    mutation_positions = list(np.unique(mutation_bits))

    flat_population[mutation_positions] = 1 - flat_population[mutation_positions]

    return population.reshape(original_population_shape)


def cross_over_population(population, cross_over_bits, selection_pool_size):
    cx_pairs = cross_over_bits[:selection_pool_size]
    selection_pool = []

    population = list(population)
    for p in cx_pairs:
        selection_pool.append(population[p])
        del population[p]
    selection_pool = np.array(selection_pool)
    new_population = list(population)

    cutting_points = cross_over_bits[selection_pool_size:][:selection_pool_size // 2]

    for pair_index in range(len(cutting_points)):
        parent1 = selection_pool[2 * pair_index]
        parent2 = selection_pool[2 * pair_index + 1]

        cutting_point = cutting_points[pair_index]

        child1 = np.concatenate([parent1[:cutting_point], parent2[cutting_point:]])
        child2 = np.concatenate([parent2[:cutting_point], parent1[cutting_point:]])
        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population, dtype='int')


def individual_to_values(individual, function, n_dimensions):
    bit_groups = np.array(individual).astype(str).reshape(n_dimensions, -1)
    values = [bits_to_number(''.join(bits), function.lower_bound, function.upper_bound) for bits in bit_groups]
    return np.array(values)


def eval_individual(individual, function, n_dimensions):
    return function.eval(individual_to_values(individual, function, n_dimensions))


def evaluate_population(population, function, n_dimensions, pool):
    scores = []
    for individual in population.astype(int):
        score = pool.apply_async(eval_individual, args=(individual, function, n_dimensions))
        scores.append(score)
    scores = [score.get() for score in scores]
    return np.array(scores)


def select_next_generation(population, global_best_individual, tournament_size, scores, selection_bits):
    new_population = [np.array(global_best_individual)]  # elitism 1
    while len(new_population) < len(population):
        int_positions = selection_bits[:tournament_size]
        tournament_scores = np.array([scores[p] for p in int_positions])

        winner_position = int_positions[tournament_scores.argmin()]
        winner = population[winner_position]
        new_population.append(np.array(winner))
        selection_bits = selection_bits[tournament_size:]

    return np.array(new_population, dtype='int')


def genetic_algorithm(genetic_algorithm_bits, bits_per_value, function, n_dimensions,
                      pop_size, num_generations, selection_pool_size, mutation_rate, tournament_size, num_workers=8):
    num_init_bits = bits_per_value * n_dimensions * pop_size
    init_bits = genetic_algorithm_bits[:num_init_bits]
    population = initialize_population(pop_size, init_bits=init_bits)
    genetic_algorithm_bits = genetic_algorithm_bits[num_init_bits:]

    global_best_individual = None
    global_best_score = np.inf
    history = []
    pool = mp.Pool(processes=num_workers)
    for _ in np.arange(num_generations):
        # mutation

        num_mutation_bits = int(num_init_bits * mutation_rate)
        mutation_bits = genetic_algorithm_bits[:num_mutation_bits]
        population = mutate_population(population, mutation_bits=mutation_bits)
        genetic_algorithm_bits = genetic_algorithm_bits[num_mutation_bits:]

        # cross-over

        num_cx_bits = selection_pool_size * 3 // 2
        cross_over_bits = genetic_algorithm_bits[:num_cx_bits]
        population = cross_over_population(population, cross_over_bits, selection_pool_size)
        genetic_algorithm_bits = genetic_algorithm_bits[num_cx_bits:]

        scores = evaluate_population(population, function, n_dimensions, pool)
        history.append(scores)

        # update global best
        best_index = scores.argmin()
        generation_best_score = scores[best_index]
        if generation_best_score < global_best_score:
            global_best_score = generation_best_score
            global_best_individual = np.array(population[best_index])

        # tournament selection
        num_selection_bits = (pop_size - 1) * tournament_size
        selection_bits = genetic_algorithm_bits[:num_selection_bits]
        population = select_next_generation(population, global_best_individual, tournament_size, scores, selection_bits)

        genetic_algorithm_bits = genetic_algorithm_bits[num_selection_bits:]
    pool.close()
    return {
        'history': np.array(history),
        'best_score': global_best_score,
        'best_individual': individual_to_values(global_best_individual, function, n_dimensions),
    }
