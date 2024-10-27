import numpy as np
import matplotlib.pyplot as plt

from membership_functions_1 import triangular


# Função para calcular a composição Max-Min
def max_min_composition(set1, set2):
    result = np.zeros((set1.shape[0], set2.shape[1]))
    for i in range(set1.shape[0]):
        for j in range(set2.shape[1]):
            result[i, j] = np.max(np.minimum(set1[i, :], set2[:, j]))
    return result


# Função para calcular a composição Min-Max
def min_max_composition(set1, set2):
    result = np.zeros((set1.shape[0], set2.shape[1]))
    for i in range(set1.shape[0]):
        for j in range(set2.shape[1]):
            result[i, j] = np.min(np.maximum(set1[i, :], set2[:, j]))
    return result


# Função para calcular a composição Max-Prod
def max_prod_composition(set1, set2):
    result = np.zeros((set1.shape[0], set2.shape[1]))
    for i in range(set1.shape[0]):
        for j in range(set2.shape[1]):
            result[i, j] = np.max(set1[i, :] * set2[:, j])
    return result


# Função para exibir a matriz de relação fuzzy e a composição
def plot_composition_matrix(matrix, title):
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Conjunto 2")
    plt.ylabel("Conjunto 1")
    plt.show()


# Exemplo de uso
universe = np.linspace(0, 50, 100)

set1 = triangular(universe, -2, 5, 10)
set2 = triangular(universe, 7.5, 15, 20)
relation_matrix = np.outer(set1, set2)  # Criando uma matriz de relação exemplo

# Calculando as composições
max_min_result = max_min_composition(relation_matrix, relation_matrix)
min_max_result = min_max_composition(relation_matrix, relation_matrix)
max_prod_result = max_prod_composition(relation_matrix, relation_matrix)

# Plotando os resultados
plot_composition_matrix(max_min_result, "Max-Min Composition")
plot_composition_matrix(min_max_result, "Min-Max Composition")
plot_composition_matrix(max_prod_result, "Max-Prod Composition")
