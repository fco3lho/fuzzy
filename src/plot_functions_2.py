import matplotlib.pyplot as plt
import numpy as np
import membership_functions_1

colors = ["blue", "yellow", "green", "red", "magenta"]


def plot_triangular(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = (i * 10) - 0.5
        c = (a + 10) + 0.5
        b = a + (c - a) / 2
        plt.plot(
            universe,
            membership_functions_1.triangular(universe, a, b, c),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
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
        plt.plot(
            universe,
            membership_functions_1.trapezoidal(universe, a, b, c, d),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gaussian(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        plt.plot(
            universe,
            membership_functions_1.gaussian(universe, universe_init + 5 + 10 * i, 2.5),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sigmoidal(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        start = i * 10
        end = start + 10

        plt.plot(
            universe,
            membership_functions_1.sigmoidal(universe, 0.1, 1, start, end),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sino(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = (i * 10) - 0.25
        b = (a + 10) + 0.25

        plt.plot(universe, membership_functions_1.sino(universe, a, b), color=colors[i])

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
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

        plt.plot(
            universe,
            membership_functions_1.s_function(universe, a, b, c),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_z_function(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = i * 10
        b = a + 15

        plt.plot(
            universe, membership_functions_1.z_function(universe, a, b), color=colors[i]
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cauchy(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = (i * 10) + 5
        gamma = 2

        plt.plot(
            universe, membership_functions_1.cauchy(universe, a, gamma), color=colors[i]
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gaussian_pair(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        mean1 = i * 10
        sigma1 = 2
        mean2 = (i * 10) + 15
        sigma2 = 2

        plt.plot(
            universe,
            membership_functions_1.gaussian_pair(
                universe, mean1, sigma1, mean2, sigma2
            ),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_bell_shaped(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = 2.5
        b = 2
        c = 5 + (i * 10)

        plt.plot(
            universe,
            membership_functions_1.bell_shaped(universe, a, b, c),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_exponencial(universe_init, universe_end, samples):
    universe = np.linspace(universe_init, universe_end, 1000)
    plt.figure(figsize=(50, 6))

    for i in range(5):
        a = i * 10
        b = 0.4
        plt.plot(
            universe,
            membership_functions_1.decreasing_exponential(universe, a, b),
            color=colors[i],
        )

    for sample in samples:
        plt.axvline(x=sample, color="gray", linestyle="--", label=f"Amostra {sample}")

    plt.title("Funções de Pertinência")
    plt.xlabel("Universo de Discurso (x)")
    plt.ylabel("Pertinência (µ)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run
universe = np.linspace(0, 50, 1000)
samples = [14, 35]

plot_triangular(0, 50, samples)
plot_trapezoidal(0, 50, samples)
plot_gaussian(0, 50, samples)
plot_sigmoidal(0, 50, samples)
plot_sino(0, 50, samples)
plot_s_function(0, 50, samples)
plot_z_function(0, 50, samples)
plot_cauchy(0, 50, samples)
plot_gaussian_pair(0, 50, samples)
plot_bell_shaped(0, 50, samples)
plot_exponencial(0, 50, samples)
