import matplotlib.pyplot as plt
import numpy as np
import membership_functions

def plot_triangular(universe_init, universe_end, samples):
  universe = np.linspace(universe_init, universe_end, 1000)
  plt.figure(figsize=(50, 6))

  for i in range(4):
      a = (i*3)-0.25
      c = (a+0.25)+3
      b = (a+0.25) + ((c - (a+0.25))/2)
      plt.plot(universe, membership_functions.triangular(universe, a, b, c), color="orange", label=f"Triangular {i}")

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

  for i in range(4):
      plt.plot(universe, membership_functions.bell_shaped(universe, 2.5, 2, 5+(i*10)), color="blue")
    
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

  for i in range(4):
      plt.plot(universe, membership_functions.gaussian(universe, universe_init + 10 + 10*i, 2.5), color="red")
    
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

  plt.plot(universe, membership_functions.decreasing_exponential(universe, 0.05), color="green")
    
  for sample in samples:
      plt.axvline(x=sample, color='gray', linestyle='--', label=f'Amostra {sample}')

  plt.title('Funções de Pertinência')
  plt.xlabel('Universo de Discurso (x)')
  plt.ylabel('Pertinência (µ)')
  plt.legend()
  plt.grid(True)
  plt.show()