import numpy as np


class Function:
    def __init__(self, name, f, real_minimum, lower_bound, upper_bound):
        self.name = name
        self.f = f
        self.real_minimum = real_minimum
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def eval(self, x):
        return self.f(x)

    def get_real_minimum(self, n=None):
        if callable(self.real_minimum):
            return self.real_minimum(n)
        else:
            return self.real_minimum


def de_jong_1(x):
    return np.sum(np.power(x, 2))


def rastrigin_6(x):
    return 10 * len(x) + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))


def schwefel7(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def griewangk8(x):
    return np.sum(np.power(x, 2)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1


def power_sum(x):
    return np.sum(np.power(np.abs(x), np.arange(2, len(x) + 2)))


def ackley10(x):
    return - 20 * np.exp(- 0.2 * np.sqrt(1 / len(x) * np.sum(np.power(x, 2)))) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)

functions_data = [
    ('De Jong 1', de_jong_1, 0, -5.12, 5.12),
    # ('Rastrigin 6', rastrigin_6, 0, -5.12, 5.12),
    ('Sum of different power 9', power_sum, 0, -1, 1),
    ('Ackley\'s Path 10', ackley10, 0, -1, 1),

    # ('Schwefel 7', schwefel7,  - 20 * 418.9829, -500, 500),
    # ('Griewangk 8', griewangk8, 0, -600, 600),

    # ('Axis parallel hyper-ellipsoid', lambda x: np.sum(np.arange(1, len(x) + 1) * np.power(x, 2)), 0, -5.12, 5.12),
    # ('Rotated hyper-ellipsoid', lambda x: np.sum(np.cumsum(np.power(x, 2))), 0, -65.536, 65.536),
    # ('Moved axis parallel hyper-ellipsoid', lambda x: np.sum(5 * np.arange(1, len(x) + 1) * np.power(x, 2)), 0, -5.12,
    #  5.12),
    # ('De Jong 2, Rosenbrock\'s valley', lambda x: np.sum(100 * (x[1:] - np.power(x[:-1], 2)) + np.power(1 - x[:-1], 2)),
    #  0, -2048, 2048),
]

functions = [Function(name=f[0], f=f[1], real_minimum=f[2], lower_bound=f[3], upper_bound=f[4]) for f in functions_data]
