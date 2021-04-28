import numpy as np


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