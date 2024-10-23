import numpy as np

def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)

def gaussian(x, mean, sigma):   
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def sigmoidal(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))

def sino(x, a, b):
    if a <= x <= b:
        return np.sin((np.pi * (x - a)) / (b - a))
    else:
        return 0.0

def s_function(x, a, b):
    if x <= a:
        return 0.0
    elif a < x <= (a + b) / 2:
        return 2 * ((x - a) / (b - a)) ** 2
    elif (a + b) / 2 < x <= b:
        return 1 - 2 * ((x - b) / (b - a)) ** 2
    else:
        return 1.0

def z_function(x, a, b):
    if x <= a:
        return 1.0
    elif a < x <= (a + b) / 2:
        return 1 - 2 * ((x - a) / (b - a)) ** 2
    elif (a + b) / 2 < x <= b:
        return 2 * ((x - b) / (b - a)) ** 2
    else:
        return 0.0

def cauchy(x, x0, gamma):
    return 1 / (1 + ((x - x0) / gamma) ** 2)

def gaussian_dupla(x, mean1, sigma1, mean2, sigma2):
    return max(np.exp(-0.5 * ((x - mean1) / sigma1) ** 2), np.exp(-0.5 * ((x - mean2) / sigma2) ** 2))

def bell_shaped(x, a, b, c):
    """
    Função de pertinência Bell-Shaped (em forma de sino).
    
    Parâmetros:
    x : valor de entrada
    a : controla a largura da curva
    b : controla a inclinação
    c : define o ponto central da curva
    
    Retorna:
    Grau de pertinência para o valor de x.
    """
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

def decreasing_exponential(x, lambd):
    """
    Função de pertinência Exponencial Decrescente.
    
    Parâmetros:
    x : valor de entrada
    lambd : taxa de decaimento exponencial
    
    Retorna:
    Grau de pertinência para o valor de x.
    """
    return np.exp(-lambd * x)
