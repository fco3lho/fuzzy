import numpy as np
import matplotlib.pyplot as plt

from membership_functions_1 import gaussian
from fuzzy_operations_3 import t_norm_min


# Definir a função não-linear f(x)
def f(x):
    return np.exp(-x / 5) * np.sin(3 * x) + 0.5 * np.sin(x)


###### 4
def fuzzy_approximation(
    x,
    mu_low_gaussian,
    sigma_low,
    mu_medium_gaussian,
    sigma_medium,
    mu_high_gaussian,
    sigma_high,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
):
    mu_low = gaussian(x, mu_low_gaussian, sigma_low)
    mu_medium = gaussian(x, mu_medium_gaussian, sigma_medium)
    mu_high = gaussian(x, mu_high_gaussian, sigma_high)

    mu1 = t_norm_min(mu_low, mu_medium)
    mu2 = t_norm_min(mu_low, mu_high)
    mu3 = t_norm_min(mu_medium, mu_high)

    y1 = a1 * x + b1
    y2 = a2 * x + b2
    y3 = a3 * x + b3

    numerator = mu1 * y1 + mu2 * y2 + mu3 * y3
    denominator = mu1 + mu2 + mu3

    return numerator / denominator if denominator != 0 else 0


if __name__ == "__main__":
    ###### 1. Criando universo e função não-linear passada
    x_values = np.linspace(0, 10, 100)
    y_values = f(x_values)

    plt.plot(x_values, y_values, label="f(x)", color="blue")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Função Não-Linear f(x)")
    plt.legend()
    plt.show()

    ###### 2. Plotando funções de pertinência gaussianas
    mu_low_gaussian, sigma_low = 2.5, 1.5
    mu_medium_gaussian, sigma_medium = 5.0, 1.5
    mu_high_gaussian, sigma_high = 7.5, 1.5

    plt.plot(
        x_values,
        [gaussian(x, mu_low_gaussian, sigma_low) for x in x_values],
        label="Baixo",
    )
    plt.plot(
        x_values,
        [gaussian(x, mu_medium_gaussian, sigma_medium) for x in x_values],
        label="Médio",
    )
    plt.plot(
        x_values,
        [gaussian(x, mu_high_gaussian, sigma_high) for x in x_values],
        label="Alto",
    )
    plt.xlabel("x")
    plt.ylabel("Pertinência")
    plt.title("Funções de Pertinência para x")
    plt.legend()
    plt.show()

    ###### 3. Consequentes das regras (valores iniciais que podemos ajustar)
    a1, b1 = 0.1, 0.5
    a2, b2 = -0.1, -0.5
    a3, b3 = 0.05, 0.3

    ###### 4. Função fuzzy_approximation()

    ###### 5. Calcular a saída aproximada para cada ponto de x
    y_approx = np.array(
        [
            fuzzy_approximation(
                x,
                mu_low_gaussian,
                sigma_low,
                mu_medium_gaussian,
                sigma_medium,
                mu_high_gaussian,
                sigma_high,
                a1,
                b1,
                a2,
                b2,
                a3,
                b3,
            )
            for x in x_values
        ]
    )

    # Calcula o erro quadrático médio (MSE)
    mse = np.mean((y_values - y_approx) ** 2)
    print(f"Erro Quadrático Médio (MSE): {mse}")

    plt.plot(x_values, y_values, label="f(x)", color="blue")
    plt.plot(
        x_values, y_approx, label="Aproximação Fuzzy", color="red", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Comparação entre f(x) e a Aproximação Fuzzy")
    plt.legend()
    plt.show()

    ###### 6
    # Parâmetros para o gradiente descendente
    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):
        y_pred = np.array(
            [
                fuzzy_approximation(
                    x,
                    mu_low_gaussian,
                    sigma_low,
                    mu_medium_gaussian,
                    sigma_medium,
                    mu_high_gaussian,
                    sigma_high,
                    a1,
                    b1,
                    a2,
                    b2,
                    a3,
                    b3,
                )
                for x in x_values
            ]
        )
        error = y_values - y_pred
        mse = np.mean(error**2)

        # Cálculo dos gradientes para cada parâmetro (a1, b1, a2, b2, a3, b3)
        a1_grad = -2 * np.mean(
            error
            * np.array(
                [
                    mu * x
                    for x, mu in zip(
                        x_values,
                        [gaussian(x, mu_low_gaussian, sigma_low) for x in x_values],
                    )
                ]
            )
        )
        b1_grad = -2 * np.mean(
            error
            * np.array([gaussian(x, mu_low_gaussian, sigma_low) for x in x_values])
        )

        a2_grad = -2 * np.mean(
            error
            * np.array(
                [
                    mu * x
                    for x, mu in zip(
                        x_values,
                        [
                            gaussian(x, mu_medium_gaussian, sigma_medium)
                            for x in x_values
                        ],
                    )
                ]
            )
        )
        b2_grad = -2 * np.mean(
            error
            * np.array(
                [gaussian(x, mu_medium_gaussian, sigma_medium) for x in x_values]
            )
        )

        a3_grad = -2 * np.mean(
            error
            * np.array(
                [
                    mu * x
                    for x, mu in zip(
                        x_values,
                        [gaussian(x, mu_high_gaussian, sigma_high) for x in x_values],
                    )
                ]
            )
        )
        b3_grad = -2 * np.mean(
            error
            * np.array([gaussian(x, mu_high_gaussian, sigma_high) for x in x_values])
        )

        # Atualização dos parâmetros
        a1 -= learning_rate * a1_grad
        b1 -= learning_rate * b1_grad
        a2 -= learning_rate * a2_grad
        b2 -= learning_rate * b2_grad
        a3 -= learning_rate * a3_grad
        b3 -= learning_rate * b3_grad

        # Exibir o erro a cada 100 épocas para monitoramento
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse:.4f}")

    # Parâmetros finais
    print("Parâmetros ajustados dos consequentes:")
    print(f"a1: {a1}, b1: {b1}")
    print(f"a2: {a2}, b2: {b2}")
    print(f"a3: {a3}, b3: {b3}")

    # Plotar o gráfico
    y_approx = np.array(
        [
            fuzzy_approximation(
                x,
                mu_low_gaussian,
                sigma_low,
                mu_medium_gaussian,
                sigma_medium,
                mu_high_gaussian,
                sigma_high,
                a1,
                b1,
                a2,
                b2,
                a3,
                b3,
            )
            for x in x_values
        ]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Função original f(x)", color="blue")
    plt.plot(x_values, y_approx, label="Aproximação fuzzy", color="red", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(
        "Comparação da Função Original com a Aproximação Fuzzy de Takagi-Sugeno (Primeira Ordem)"
    )
    plt.legend()
    plt.grid()
    plt.show()
