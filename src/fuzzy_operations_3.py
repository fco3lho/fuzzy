import matplotlib.pyplot as plt
import numpy as np


def fuzzy_complement(membership_values):
    return 1 - membership_values


def fuzzy_union(membership_values1, membership_values2):
    return np.maximum(membership_values1, membership_values2)


def fuzzy_intersection(membership_values1, membership_values2):
    return np.minimum(membership_values1, membership_values2)


def fuzzy_t_norm(membership_values1, membership_values2):
    return membership_values1 * membership_values2


def fuzzy_s_norm(membership_values1, membership_values2):
    return np.minimum(1, membership_values1 + membership_values2)


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


def plot_t_norm(universe, set1, set2):
    t_norm = fuzzy_t_norm(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot Norma-dua
    plt.subplot(2, 1, 2)
    plt.plot(universe, t_norm, label="Norma-dua", color="blue", linestyle="--")
    plt.title("Norma-dua")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_s_norm(universe, set1, set2):
    s_norm = fuzzy_s_norm(set1, set2)

    plt.figure(figsize=(12, 10))

    # Plot original sets
    plt.subplot(2, 1, 1)
    plt.plot(universe, set1, label="Conjunto 1", color="blue")
    plt.plot(universe, set2, label="Conjunto 2", color="orange")
    plt.title("Conjuntos Originais")
    plt.legend()
    plt.grid(True)

    # Plot Norma-dua
    plt.subplot(2, 1, 2)
    plt.plot(universe, s_norm, label="Norma-dua", color="blue", linestyle="--")
    plt.title("Norma-dua")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
