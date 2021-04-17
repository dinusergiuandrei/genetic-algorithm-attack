import numpy as np

from bits import find_number_of_bits


def mutate_population(population, mutation_rate):
    return population


def cross_over_population(population, cross_over_rate):
    return population


def evaluate_population(population, function):
    return None


def select_next_generation(population, scores):
    return population


def genetic_algorithm(pop_size, num_generations, mutation_rate, cross_over_rate, function):
    n_bits = find_number_of_bits(lower_bound=function.lower_bound, upper_bound=function.upper_bound)
    population = generate_random_population(pop_size, bits_per_individual=n_bits)

    for generation in range(num_generations):
        population = mutate_population(population, mutation_rate)
        population = cross_over_population(population, cross_over_rate)


def generate_random_population(pop_size, bits_per_individual):
    population = np.random.choice(['0', '1'], (pop_size, bits_per_individual))
    return np.array([''.join(a) for a in population])


if __name__ == '__main__':
    print(generate_random_population(pop_size=5, bits_per_individual=10))
