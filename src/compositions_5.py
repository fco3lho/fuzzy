import numpy as np
import matplotlib.pyplot as plt
from membership_functions_1 import triangular


def fuzzy_relation(set1, set2, operator):
    if len(set1) != len(set2):
        raise ValueError("Os conjuntos devem ter o mesmo tamanho.")

    relation_matrix = np.zeros((len(set1), len(set2)))

    for i in range(len(set1)):
        for j in range(len(set2)):
            relation_matrix[i, j] = operator(set1[i], set2[j])

    return relation_matrix


def max_min_composition(relation1, relation2):
    return np.array(
        [
            [
                np.max(np.minimum(relation1[i, :], relation2[:, j]))
                for j in range(relation2.shape[1])
            ]
            for i in range(relation1.shape[0])
        ]
    )


def min_max_composition(relation1, relation2):
    return np.array(
        [
            [
                np.min(np.maximum(relation1[i, :], relation2[:, j]))
                for j in range(relation2.shape[1])
            ]
            for i in range(relation1.shape[0])
        ]
    )


def max_prod_composition(relation1, relation2):
    return np.array(
        [
            [
                np.max(relation1[i, :] * relation2[:, j])
                for j in range(relation2.shape[1])
            ]
            for i in range(relation1.shape[0])
        ]
    )


def plot_compositions(relation1, relation2):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    max_min_result = max_min_composition(relation1, relation2)
    min_max_result = min_max_composition(relation1, relation2)
    max_prod_result = max_prod_composition(relation1, relation2)

    for ax, result, title in zip(
        axs,
        [max_min_result, min_max_result, max_prod_result],
        ["Max-Min Composition", "Min-Max Composition", "Max-Prod Composition"],
    ):
        cax = ax.matshow(result, cmap="viridis", extent=[0, 50, 50, 0])
        fig.colorbar(cax, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Set2")
        ax.set_ylabel("Set1")
        ax.set_xlim(0, 50)
        ax.set_ylim(50, 0)

    plt.show()


# Run
if __name__ == "__main__":
    universe = np.linspace(0, 50, 100)

    set1 = triangular(universe, 5, 10, 15)
    set2 = triangular(universe, 9, 15, 21)

    relation1 = fuzzy_relation(set1, set2, np.minimum)
    relation2 = fuzzy_relation(set1, set2, np.maximum)

    plot_compositions(relation1, relation2)
