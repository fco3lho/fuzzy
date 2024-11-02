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
    cax = ax.matshow(relation, cmap="Blues")
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"Relação Fuzzy usando {operator.__name__}")

    plt.show()


# Exemplo de execução
universe = np.linspace(0, 50, 10)

set1 = triangular(universe, -2, 5, 10)
set2 = triangular(universe, 20, 30, 40)

plot_fuzzy_relations(set1, set2, t_norm_min)
plot_fuzzy_relations(set1, set2, t_norm_product)
plot_fuzzy_relations(set1, set2, t_norm_drastic_product)
plot_fuzzy_relations(set1, set2, t_norm_limited_product)
