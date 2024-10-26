import matplotlib.pyplot as plt
import numpy as np
import membership_functions

colors = ["blue", "yellow", "green", "red", "magenta"]

def plot_triangular(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10) - 0.5
		c = (a + 10) + 0.5
		b = a + (c - a)/2
		plt.plot(universe, membership_functions.triangular(universe, a, b, c), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência triangular ({i}): {membership_functions.triangular(samples[0], a, b, c)}")
		print(f"Valor da amostra 2 para a função de pertinência triangular ({i}): {membership_functions.triangular(samples[1], a, b, c)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show()

def plot_trapezoidal(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10) - 0.5
		d = (a + 10) + 0.5
		b = a + 3
		c = d - 3
		plt.plot(universe, membership_functions.trapezoidal(universe, a, b, c, d), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência trapeizodal ({i}): {membership_functions.trapezoidal(samples[0], a, b, d)}")
		print(f"Valor da amostra 2 para a função de pertinência trapeizodal ({i}): {membership_functions.trapezoidal(samples[1], a, b, d)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_gaussian(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		plt.plot(universe, membership_functions.gaussian(universe, universe_init + 5 + 10*i, 2.5), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência triangular ({i}): {membership_functions.gaussian(samples[0], universe_init + 5 + 10*i, 2.5)}")
		print(f"Valor da amostra 2 para a função de pertinência triangular ({i}): {membership_functions.gaussian(samples[1], universe_init + 5 + 10*i, 2.5)}")
		
	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show()

def plot_sigmoidal(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		start = i * 10
		end = start + 10

		plt.plot(universe, membership_functions.sigmoidal(universe, 0.1, 1, start, end), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência sigmoidal ({i}): {membership_functions.sigmoidal(samples[0], 0.1, 1, start, end)}")
		print(f"Valor da amostra 2 para a função de pertinência sigmoidal ({i}): {membership_functions.sigmoidal(samples[1], 0.1, 1, start, end)}")
		
	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show()

def plot_sino(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10) - 0.25
		b = (a + 10) + 0.25

		plt.plot(universe, membership_functions.sino(universe, a, b), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência sino ({i}): {membership_functions.sino(samples[0], a, b)}")
		print(f"Valor da amostra 2 para a função de pertinência sino ({i}): {membership_functions.sino(samples[1], a, b)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_s_function(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10) - 0.25
		b = (a + 10) + 0.25
		c = (a + 10) + 3

		plt.plot(universe, membership_functions.s_function(universe, a, b, c), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência s_function ({i}): {membership_functions.s_function(samples[0], a, b, c)}")
		print(f"Valor da amostra 2 para a função de pertinência s_function ({i}): {membership_functions.s_function(samples[1], a, b, c)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_z_function(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10)
		b = (a + 15)

		plt.plot(universe, membership_functions.z_function(universe, a, b), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência z_function ({i}): {membership_functions.z_function(samples[0], a, b)}")
		print(f"Valor da amostra 2 para a função de pertinência z_function ({i}): {membership_functions.z_function(samples[1], a, b)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_cauchy(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = (i * 10) + 5
		gamma = 2

		plt.plot(universe, membership_functions.cauchy(universe, a, gamma), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência cauchy ({i}): {membership_functions.cauchy(samples[0], a, gamma)}")
		print(f"Valor da amostra 2 para a função de pertinência cauchy ({i}): {membership_functions.cauchy(samples[1], a, gamma)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_gaussian_pair(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		mean1 = (i * 10)
		sigma1 = 2
		mean2 = (i * 10) + 15
		sigma2 = 2

		plt.plot(universe, membership_functions.gaussian_pair(universe, mean1, sigma1, mean2, sigma2), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência gaussian_pair ({i}): {membership_functions.gaussian_pair(samples[0], mean1, sigma1, mean2, sigma2)}")
		print(f"Valor da amostra 2 para a função de pertinência gaussian_pair ({i}): {membership_functions.gaussian_pair(samples[1], mean1, sigma1, mean2, sigma2)}")

	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show() 

def plot_bell_shaped(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = 2.5
		b = 2
		c = 5+(i*10)

		plt.plot(universe, membership_functions.bell_shaped(universe, a, b, c), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência bell_shaped ({i}): {membership_functions.bell_shaped(samples[0], a, b, c)}")
		print(f"Valor da amostra 2 para a função de pertinência bell_shaped ({i}): {membership_functions.bell_shaped(samples[1], a, b, c)}")
	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show()

def plot_exponencial(universe_init, universe_end, samples):
	universe = np.linspace(universe_init, universe_end, 1000)
	plt.figure(figsize=(50, 6))

	for i in range(5):
		a = i*10
		b = 0.4
		plt.plot(universe, membership_functions.decreasing_exponential(universe, a, b), color=colors[i])
		print(f"Valor da amostra 1 para a função de pertinência decreasing_exponencial ({i}): {membership_functions.decreasing_exponential(samples[0], a, b)}")
		print(f"Valor da amostra 2 para a função de pertinência decreasing_exponencial ({i}): {membership_functions.decreasing_exponential(samples[1], a, b)}")
		
	for sample in samples:
		plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

	plt.title('Funções de Pertinência')
	plt.xlabel('Universo de Discurso (x)')
	plt.ylabel('Pertinência (µ)')
	plt.legend()
	plt.grid(True)
	plt.show()