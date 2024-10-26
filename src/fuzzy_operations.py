import matplotlib.pyplot as plt
import numpy as np

def fuzzy_complement(membership_values):
    return 1 - membership_values

def fuzzy_union(membership_values1, membership_values2):
    return np.maximum(membership_values1, membership_values2)

def fuzzy_intersection(membership_values1, membership_values2):
    return np.minimum(membership_values1, membership_values2)

def fuzzy_norma_dua(membership_values1, membership_values2):
    return (membership_values1 + membership_values2) / 2

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
    plt.plot(universe, comp1, label="Complemento Conjunto 1", color="blue", linestyle='--')
    plt.plot(universe, comp2, label="Complemento Conjunto 2", color="orange", linestyle='--')
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
    plt.plot(universe, union, label="União", color="blue", linestyle='--')
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

    # Plot Interseção
    plt.subplot(2, 1, 2)
    plt.plot(universe, intersection, label="Interseção", color="blue", linestyle='--')
    plt.title("Interseção")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_norma_dua(universe, set1, set2):
	norma_dua = fuzzy_norma_dua(set1, set2)

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
	plt.plot(universe, norma_dua, label="Norma-dua", color="blue", linestyle='--')
	plt.title("Norma-dua")
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.show()
