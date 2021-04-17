import numpy as np


def number_to_bits(n, n_bits, lower_bound, upper_bound):
    scaled = (n - lower_bound) / (upper_bound - lower_bound)  # step 1. between 0 and 1
    max_on_ints = int('1' * n_bits, 2)
    as_int = int(scaled * max_on_ints)  # step 2
    as_binary = bin(as_int)[2:].rjust(n_bits, '0')  # step 3
    return as_binary


def bits_to_number(bits, lower_bound, upper_bound):
    as_int = int(bits, 2)  # step 3

    max_on_ints = int('1' * len(bits), 2)
    as_int = as_int / max_on_ints  # step 2. between 0 and 1

    rescaled = as_int * (upper_bound - lower_bound) + lower_bound
    return rescaled


def compute_average_error(n_bits, lower_bound, upper_bound, n_tests=100):
    numbers = np.random.uniform(lower_bound, upper_bound, size=n_tests)
    errors = []
    for n in numbers:
        conv = bits_to_number(number_to_bits(n, n_bits=n_bits, lower_bound=lower_bound, upper_bound=upper_bound),
                              lower_bound=lower_bound, upper_bound=upper_bound)
        errors.append(abs(n - conv))
    return np.mean(errors)


def find_number_of_bits(tolerance=1e-6, lower_bound=-1, upper_bound=1, n_tests=100):
    n_bits = 5
    last_error = 1
    while last_error > tolerance:
        n_bits += 1
        last_error = compute_average_error(n_bits=n_bits, lower_bound=lower_bound,
                                           upper_bound=upper_bound, n_tests=n_tests)
    return n_bits


if __name__ == '__main__':
    print(find_number_of_bits())
