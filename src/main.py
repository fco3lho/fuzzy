import matplotlib.pyplot as plt
import numpy as np
import plot_functions_2
import fuzzy_operations_3
import membership_functions_1
import fuzzy_relation_4

############### Definir universo de discurso
graph_size = 50
x = np.linspace(0, graph_size, 1000)

############### Amostras para fuzzificação
amostras = [14, 35]

############### Plotar as funções de pertinência
# plot_functions.plot_triangular(0, 50, amostras)
# plot_functions.plot_trapezoidal(0, 50, amostras)
# plot_functions.plot_gaussian(0, 50, amostras)
# plot_functions.plot_sigmoidal(0, 50, amostras)
# plot_functions.plot_sino(0, 50, amostras)
# plot_functions.plot_s_function(0, 50, amostras)
# plot_functions.plot_z_function(0, 50, amostras)
# plot_functions.plot_cauchy(0, 50, amostras)
# plot_functions.plot_gaussian_pair(0, 50, amostras)
# plot_functions.plot_bell_shaped(0, 50, amostras)
# plot_functions.plot_exponencial(0, 50, amostras)
universe = np.linspace(0, 50, 1000)
fuzzy_operations_3.plot_s_norm(universe, membership_functions_1.triangular(universe, -2, 5, 10), membership_functions_1.triangular(universe, 7.5, 15, 20))
