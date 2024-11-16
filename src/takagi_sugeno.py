import numpy as np
import matplotlib.pyplot as plt

from membership_functions_1 import gaussian
from fuzzy_operations_3 import t_norm_min, t_norm_product, t_norm_limited_product


# Definir a função não-linear f(x)
def f(x):
    return np.exp(-x / 5) * np.sin(3 * x) + 0.5 * np.sin(x)


###### 4
def fuzzy_approximation(x, mu_params, consequents):
    mu_values = [gaussian(x, mu, sigma) for mu, sigma in mu_params]

    t_norm_values = []
    for i in range(len(mu_values)):
        for j in range(i + 1, len(mu_values)):
            t_norm_values.append(t_norm_product(mu_values[i], mu_values[j]))

    y_values = [a * x + b for a, b in consequents]

    numerator = sum(mu * y for mu, y in zip(t_norm_values, y_values))
    denominator = sum(t_norm_values)

    return numerator / denominator if denominator != 0 else 0


###### 6
def calculate_gradient(error, x_values, mu_values):
    a_grad = -2 * np.mean(
        error * np.array([mu * x for x, mu in zip(x_values, mu_values)])
    )
    b_grad = -2 * np.mean(error * np.array(mu_values))
    return a_grad, b_grad


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
    mu_params = [(2.5, 1.5), (5.0, 1.5), (7.5, 1.5)]

    for i in range(len(mu_params)):
        plt.plot(
            x_values,
            [gaussian(x, mu_params[i][0], mu_params[i][1]) for x in x_values],
            label=f"Função {i}",
        )

    plt.xlabel("x")
    plt.ylabel("Pertinência")
    plt.title("Funções de Pertinência para x")
    plt.legend()
    plt.show()

    ###### 3. Consequentes das regras
    consequents = [(0.1, 0.5), (-0.1, -0.5), (0.05, 0.3)]

    ###### 4. Função fuzzy_approximation()

    ###### 5. Calcular a saída aproximada para cada ponto de x
    y_approx = np.array(
        [fuzzy_approximation(x, mu_params, consequents) for x in x_values]
    )

    # Calcula o erro quadrático médio (MSE)
    mse = np.mean((y_values - y_approx) ** 2)
    rmse = mse ** (1 / 2)
    print(f"Erro quadrático médio (MSE): {mse:.4f}")
    print(f"Raiz quadrada do erro quadrático médio (RMSE): {rmse:.4f}")

    plt.plot(x_values, y_values, label="f(x)", color="blue")
    plt.plot(
        x_values, y_approx, label="Aproximação Fuzzy", color="red", linestyle="dashed"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Comparação entre f(x) e a Aproximação Fuzzy")
    plt.legend()
    plt.show()

    ###### 6. Gradiente descendente
    # Parâmetros para o gradiente descendente
    learning_rate = 0.05
    epochs = 1000

    mse_history = []
    rmse_history = []

    for epoch in range(epochs):
        y_pred = np.array(
            [fuzzy_approximation(x, mu_params, consequents) for x in x_values]
        )

        error = y_values - y_pred

        mse = np.mean(error**2)
        rmse = (np.mean(error**2)) ** (1 / 2)

        mse_history.append(mse)
        rmse_history.append(rmse)

        for i in range(len(consequents)):
            mu_values = [
                gaussian(x, mu_params[i][0], mu_params[i][1]) for x in x_values
            ]
            a_grad, b_grad = calculate_gradient(error, x_values, mu_values)

            consequents[i] = (
                consequents[i][0] - learning_rate * a_grad,
                consequents[i][1] - learning_rate * b_grad,
            )

        # Exibir o erro a cada 100 épocas para monitoramento
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    print("\nParâmetros ajustados dos consequentes:")
    for i, (a, b) in enumerate(consequents):
        print(f"Regra {i + 1}: a{i + 1} = {a:.4f}, b{i + 1} = {b:.4f}")

    # Erro
    print(f"\nMSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # Plotar o gráfico
    y_approx = np.array(
        [fuzzy_approximation(x, mu_params, consequents) for x in x_values]
    )

    plt.plot(x_values, y_values, label="Função original f(x)", color="blue")
    plt.plot(x_values, y_approx, label="Aproximação fuzzy", color="red", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Aproximação Fuzzy com gradiente descendente")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotar o gráfico do erro MSE
    plt.plot(range(epochs), mse_history, label="Erro (MSE)", color="orange")
    plt.xlabel("Épocas")
    plt.ylabel("Erro (MSE)")
    plt.title("Evolução do erro ao longo do treinamento")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotar o gráfico do erro RMSE
    plt.plot(range(epochs), rmse_history, label="Erro (RMSE)", color="orange")
    plt.xlabel("Épocas")
    plt.ylabel("Erro (RMSE)")
    plt.title("Evolução do erro ao longo do treinamento")
    plt.legend()
    plt.grid()
    plt.show()
