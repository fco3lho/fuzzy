import matplotlib.pyplot as plt
import numpy as np

import membership_functions_1


def fuzzy_complement(membership_values):
    return 1 - membership_values


def fuzzy_union(membership_values1, membership_values2):
    return np.maximum(membership_values1, membership_values2)


def fuzzy_intersection(membership_values1, membership_values2):
    return np.minimum(membership_values1, membership_values2)


# T-norms
def t_norm_min(membership_values1, membership_values2):
    return np.minimum(membership_values1, membership_values2)


def t_norm_product(membership_values1, membership_values2):
    return membership_values1 * membership_values2


def t_norm_limited_product(membership_values1, membership_values2):
    return np.maximum(0, membership_values1 + membership_values2 - 1)


def t_norm_drastic_product(membership_values1, membership_values2):
    return np.where(
        (membership_values1 == 1) | (membership_values2 == 1),
        np.minimum(membership_values1, membership_values2),
        0,
    )


# S-norms
def s_norm_max(membership_values1, membership_values2):
    return np.maximum(membership_values1, membership_values2)


def s_norm_probabilistic_sum(membership_values1, membership_values2):
    return (
        membership_values1
        + membership_values2
        - membership_values1 * membership_values2
    )


def s_norm_limited_sum(membership_values1, membership_values2):
    return np.minimum(1, membership_values1 + membership_values2)


def s_norm_drastic_sum(membership_values1, membership_values2):
    return np.where(
        membership_values1 + membership_values2 > 1,
        1,
        np.maximum(membership_values1, membership_values2),
    )


### Plot functions


def plot_complement(universe, set1, set2):
    comp1 = fuzzy_complement(set1)
    comp2 = fuzzy_complement(set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot Complemento
    plt.subplot(2, 1, 2)
    plt.plot(
        universe, comp1, label="Complemento Conjunto 1", color="blue", linestyle="--"
    )
    plt.plot(
        universe, comp2, label="Complemento Conjunto 2", color="orange", linestyle="--"
    )
    plt.title("Complemento")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_union(universe, set1, set2):
    union = fuzzy_union(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot União
    plt.subplot(2, 1, 2)
    plt.plot(universe, union, label="União", color="blue", linestyle="--")
    plt.title("União")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_intersection(universe, set1, set2):
    intersection = fuzzy_intersection(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot Intercessão
    plt.subplot(2, 1, 2)
    plt.plot(universe, intersection, label="Intercessão", color="blue", linestyle="--")
    plt.title("Intercessão")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# T-norms
def plot_t_norm_min(universe, set1, set2):
    t_norm = t_norm_min(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot T-norm min
    plt.subplot(2, 1, 2)
    plt.plot(universe, t_norm, label="T-norm min", color="blue", linestyle="--")
    plt.title("T-norm min")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_t_norm_product(universe, set1, set2):
    t_norm = t_norm_product(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot T-norm product
    plt.subplot(2, 1, 2)
    plt.plot(universe, t_norm, label="T-norm product", color="blue", linestyle="--")
    plt.title("T-norm product")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_t_norm_limited_product(universe, set1, set2):
    t_norm = t_norm_limited_product(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot T-norm limited_product
    plt.subplot(2, 1, 2)
    plt.plot(
        universe, t_norm, label="T-norm limited_product", color="blue", linestyle="--"
    )
    plt.title("T-norm limited_product")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_t_norm_drastic_product(universe, set1, set2):
    t_norm = t_norm_drastic_product(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot T-norm drastic_product
    plt.subplot(2, 1, 2)
    plt.plot(
        universe, t_norm, label="T-norm drastic_product", color="blue", linestyle="--"
    )
    plt.title("T-norm drastic_product")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# S-norms
def plot_s_norm_max(universe, set1, set2):
    s_norm = s_norm_max(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot S-norm max
    plt.subplot(2, 1, 2)
    plt.plot(universe, s_norm, label="S-norm max", color="blue", linestyle="--")
    plt.title("S-norm max")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_s_norm_probabilistic_sum(universe, set1, set2):
    s_norm = s_norm_probabilistic_sum(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot S-norm probabilistic_sum
    plt.subplot(2, 1, 2)
    plt.plot(
        universe, s_norm, label="S-norm probabilistic_sum", color="blue", linestyle="--"
    )
    plt.title("S-norm probabilistic_sum")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_s_norm_limited_sum(universe, set1, set2):
    s_norm = s_norm_limited_sum(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot S-norm limited_sum
    plt.subplot(2, 1, 2)
    plt.plot(universe, s_norm, label="S-norm limited_sum", color="blue", linestyle="--")
    plt.title("S-norm limited_sum")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_s_norm_drastic_sum(universe, set1, set2):
    s_norm = s_norm_drastic_sum(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot S-norm drastic_sum
    plt.subplot(2, 1, 2)
    plt.plot(universe, s_norm, label="S-norm drastic_sum", color="blue", linestyle="--")
    plt.title("S-norm drastic_sum")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Run
if __name__ == "__main__":
    universe = np.linspace(0, 50, 1000)
    samples = [14, 35]

    set1 = membership_functions_1.triangular(universe, -2, 5, 10)
    set2 = membership_functions_1.triangular(universe, 7.5, 15, 20)

    plot_complement(universe, set1, set2)
    plot_union(universe, set1, set2)
    plot_intersection(universe, set1, set2)
    plot_t_norm_min(universe, set1, set2)
    plot_t_norm_product(universe, set1, set2)
    plot_t_norm_limited_product(universe, set1, set2)
    plot_t_norm_drastic_product(universe, set1, set2)
    plot_s_norm_max(universe, set1, set2)
    plot_s_norm_probabilistic_sum(universe, set1, set2)
    plot_s_norm_limited_sum(universe, set1, set2)
    plot_t_norm_drastic_product(universe, set1, set2)
