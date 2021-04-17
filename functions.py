import numpy as np


class Function:
    def __init__(self, name, f, real_minimum, lower_bound, upper_bound):
        self.name = name
        self.f = f
        self.real_minimum = real_minimum
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


functions = [
    ('De Jong 1', lambda x: np.sum(np.power(x, 2)), 0, -5.12, 5.12),
    ('Axis parallel hyper-ellipsoid', lambda x: np.sum(np.arange(1, len(x) + 1) * np.power(x, 2)), 0, -5.12, 5.12),
    ('Rotated hyper-ellipsoid', lambda x: np.sum(np.cumsum(np.power(x, 2))), 0, -65.536, 65.536),
    ('Moved axis parallel hyper-ellipsoid', lambda x: np.sum(5 * np.arange(1, len(x) + 1) * np.power(x, 2)), 0, -5.12,
     5.12),
    ('De Jong 2/Rosenbrock\'s valley', lambda x: np.sum(100 * (x[1:] - np.power(x[:-1], 2)) + np.power(1 - x[:-1], 2)),
     0, -2048, 2048),
    ('Rastrigin 6', lambda x: 10 * len(x) + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x)), 0, -5.12, 5.12),
    ('Schwefel 7', lambda x: np.sum(-x * np.sin(np.sqrt(np.abs(x)))), lambda x: - len(x) * 418.9829, -500, 500),
    (
    'Griewangk 8', lambda x: np.sum(np.power(x, 2)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1,
    lambda x: - len(x) * 418.9829, -600, 600),
    ('Sum of different power 9', lambda x: np.sum(np.power(np.abs(x), np.arange(2, len(x) + 2))), 0, -1, 1),
    ('Ackley\'s Path 10', lambda x: - 20 * np.exp(- 0.2 * np.sqrt(1 / len(x) * np.sum(np.power(x, 2)))) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1), 0, -1, 1),
]