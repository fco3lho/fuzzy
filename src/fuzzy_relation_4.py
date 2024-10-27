import numpy as np
import matplotlib.pyplot as plt

from membership_functions_1 import triangular
from fuzzy_operations_3 import fuzzy_t_norm, fuzzy_s_norm


def fuzzy_relation(set1, set2, t_norm):
    if len(set1) != len(set2):
        raise ValueError("Os conjuntos devem ter o mesmo tamanho.")

    relation_matrix = np.zeros((len(set1), len(set2)))

    for i in range(len(set1)):
        for j in range(len(set2)):
            relation_matrix[i, j] = t_norm(set1[i], set2[j])

    return relation_matrix


def plot_fuzzy_relations(set1, set2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    relation_t_norm_product = fuzzy_relation(set1, set2, fuzzy_t_norm)
    cax1 = axs[0].matshow(relation_t_norm_product, cmap="Blues")
    fig.colorbar(cax1, ax=axs[0])
    axs[0].set_title("T-norma (Product)")

    relation_s_norm_minimum = fuzzy_relation(set1, set2, fuzzy_s_norm)
    cax2 = axs[1].matshow(relation_s_norm_minimum, cmap="Blues")
    fig.colorbar(cax2, ax=axs[1])
    axs[1].set_title("S-norma (Minimum)")

    plt.show()


####### Exemplos de execução
universe = np.linspace(0, 50, 100)

# Plot dos resultados
plot_fuzzy_relations(triangular(universe, -2, 5, 10), triangular(universe, 7.5, 15, 20))
