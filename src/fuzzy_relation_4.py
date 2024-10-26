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

def plot_fuzzy_relations(relation_matrix_min, relation_matrix_product, title_min, title_product):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    cax1 = axs[0].matshow(relation_matrix_min, cmap='Blues')
    fig.colorbar(cax1, ax=axs[0])
    axs[0].set_title(title_min)

    cax2 = axs[1].matshow(relation_matrix_product, cmap='Blues')
    fig.colorbar(cax2, ax=axs[1])
    axs[1].set_title(title_product)

    plt.show()

####### Exemplos de execução
universe = np.linspace(0, 50, 100)

# Plot dos resultados
plot_fuzzy_relations(fuzzy_relation(triangular(universe, -2, 5, 10), triangular(universe, 7.5, 15, 20), fuzzy_t_norm), 
                     fuzzy_relation(triangular(universe, -2, 5, 10), triangular(universe, 7.5, 15, 20), fuzzy_s_norm), 
                     "Relação Fuzzy - T-norma (Product)", "Relação Fuzzy - S-norma (Sum)")

