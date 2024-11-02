import numpy as np
import matplotlib.pyplot as plt

from membership_functions_1 import triangular
from fuzzy_operations_3 import (
    t_norm_min,
    t_norm_product,
    t_norm_limited_product,
    t_norm_drastic_product,
)


def fuzzy_relation(set1, set2, operator):
    if len(set1) != len(set2):
        raise ValueError("Os conjuntos devem ter o mesmo tamanho.")

    relation_matrix = np.zeros((len(set1), len(set2)))

    for i in range(len(set1)):
        for j in range(len(set2)):
            relation_matrix[i, j] = operator(set1[i], set2[j])

    return relation_matrix


def plot_fuzzy_relations(set1, set2, operator):
    relation = fuzzy_relation(set1, set2, operator)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(relation, cmap="viridis", extent=[0, 50, 50, 0])
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"Relação Fuzzy usando {operator.__name__}")

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

    plot_fuzzy_relations(set1, set2, t_norm_min)
    plot_fuzzy_relations(set1, set2, t_norm_product)
    plot_fuzzy_relations(set1, set2, t_norm_drastic_product)
    plot_fuzzy_relations(set1, set2, t_norm_limited_product)
