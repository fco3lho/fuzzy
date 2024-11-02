import numpy as np


def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)


def trapezoidal(x, a, b, c, d):
    result = np.zeros_like(x, dtype=float)

    result = np.where((x > a) & (x <= b), (x - a) / (b - a), result)
    result = np.where((x > b) & (x <= c), 1.0, result)
    result = np.where((x > c) & (x < d), (d - x) / (d - c), result)

    return result


def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def sigmoidal(x, a, c, x_start, x_end):
    return np.where(
        (x >= x_start - 0.25) & (x <= x_end + 0.25), 1 / (1 + np.exp(-a * (x - c))), 0
    )


def sino(x, a, b):
    result = np.zeros_like(x, dtype=float)

    in_range = (x >= a) & (x <= b)
    result[in_range] = np.sin((np.pi * (x[in_range] - a)) / (b - a))

    return result


def s_function(x, a, b, c):
    result = np.zeros_like(x, dtype=float)

    low = x <= a
    mid1 = (x > a) & (x <= (a + b) / 2)
    mid2 = (x > (a + b) / 2) & (x <= b)
    high = x > b
    end = x > c

    result[low] = 0.0
    result[mid1] = 2 * ((x[mid1] - a) / (b - a)) ** 2
    result[mid2] = 1 - 2 * ((x[mid2] - b) / (b - a)) ** 2
    result[high] = 1.0
    result[end] = 0.0

    return result


def z_function(x, a, b):
    result = np.zeros_like(x, dtype=float)

    low = x <= a
    mid1 = (x > a) & (x <= (a + b) / 2)
    mid2 = (x > (a + b) / 2) & (x <= b)
    high = x > b

    result[low] = 0.0
    result[mid1] = 1 - 2 * ((x[mid1] - a) / (b - a)) ** 2
    result[mid2] = 2 * ((x[mid2] - b) / (b - a)) ** 2
    result[high] = 0.0

    return result


def cauchy(x, a, gamma):
    return 1 / (1 + ((x - a) / gamma) ** 2)


def gaussian_pair(x, mean1, sigma1, mean2, sigma2):
    return np.maximum(
        np.exp(-0.5 * ((x - mean1) / sigma1) ** 2),
        np.exp(-0.5 * ((x - mean2) / sigma2) ** 2),
    )


def bell_shaped(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))


def decreasing_exponential(x, a, lambd):
    return np.where(x >= a, np.exp(-lambd * (x - a)), 0)
