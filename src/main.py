import matplotlib.pyplot as plt
import numpy as np
import membership_functions
import plot_functions

# Definir universo de discurso
graph_size = 50
x = np.linspace(0, graph_size, 1000)

# Aplicar as funções de pertinência
bell_values_1 = membership_functions.bell_shaped(x, 1, 2, 1)
exp_values_1 = membership_functions.decreasing_exponential(x, 0.2)
triangular_values_1 = membership_functions.triangular(x, 1, 2.5, 4)
gauss_values_1 = membership_functions.gaussian(x, 3, 1.5)

bell_values_2 = membership_functions.bell_shaped(x, 1, 2, 4)
triangular_values_2 = membership_functions.triangular(x, 4, 5.5, 7)
gauss_values_2 = membership_functions.gaussian(x, 6, 1.5)

bell_values_3 = membership_functions.bell_shaped(x, 1, 2, 7)
triangular_values_3 = membership_functions.triangular(x, 7, 8.5, 10)
gauss_values_3 = membership_functions.gaussian(x, 9, 1.5)

# Amostras para fuzzificação
amostras = [4, 8]

### Plotar as funções de pertinência
plot_functions.plot_triangular(0, 10, amostras)
plot_functions.plot_bell_shaped(0, 40, amostras)
plot_functions.plot_gaussian(0, 50, amostras)
plot_functions.plot_exponencial(0, 50, amostras)