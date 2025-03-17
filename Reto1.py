import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 1. Cargar los datos (asegúrate de que el delimitador sea correcto)
try:
    siniestros = pd.read_csv("siniestros.csv", delimiter=';')
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo siniestros.csv")
    exit()

# 2. Definir la función para el negativo del log-likelihood
def neg_log_likelihood(params, data):
    """Calcula el negativo del log-likelihood para la distribución Gamma."""
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf  # Evitar valores inválidos

    log_likelihood = gamma.logpdf(data, a=alpha, scale=beta)
    return -np.sum(log_likelihood)

# 3. Valores iniciales para alpha y beta
initial_guess = [1, 1]

# 4. Optimizar el log-likelihood usando la columna 'Pago' (sin ajustar)
from scipy.optimize import minimize
result = minimize(neg_log_likelihood, initial_guess,
                    args=(siniestros['Pago'],),  # Usa la columna 'Pago' original
                    method='L-BFGS-B',
                    bounds=[(0.001, None), (0.001, None)])

# 5. Imprimir los resultados
if result.success:
    alpha_opt, beta_opt = result.x
    print(f"Parámetros Gamma optimizados (columna 'Pago' sin ajustar):\n  alpha = {alpha_opt:.4f}\n  beta = {beta_opt:.4f}")
else:
    print("La optimización no convergió.")
    print(result.message)
    exit()

# 6. Crear el gráfico
plt.figure(figsize=(10, 6))

# a. Histograma de los datos reales
plt.hist(siniestros['Pago'], bins=30, density=True, alpha=0.6, color='skyblue', label='Datos Reales (Pago)')

# b. Generar valores x para la distribución Gamma
x = np.linspace(0, siniestros['Pago'].max(), 100)

# c. Calcular la función de densidad de probabilidad (PDF) de la Gamma
pdf = gamma.pdf(x, a=alpha_opt, scale=beta_opt)

# d. Graficar la PDF de la Gamma
plt.plot(x, pdf, 'r-', label=f'Ajuste Gamma (α={alpha_opt:.2f}, β={beta_opt:.2f})')

# e. Añadir etiquetas y título
plt.xlabel('Costo del Siniestro (Pago)')
plt.ylabel('Densidad')
plt.title('Ajuste de Distribución Gamma a Costos de Siniestro')
plt.legend()

# f. Mostrar el gráfico
plt.show()