import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
from scipy.optimize import minimize

# 1. Cargar los datos
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

# 4. Ajustar la distribución Gamma a los datos de 2023
siniestros_2023 = siniestros[siniestros['Anio'] == 2023]['Pago']

result_2023 = minimize(neg_log_likelihood, initial_guess,
                    args=(siniestros_2023,),
                    method='L-BFGS-B',
                    bounds=[(0.001, None), (0.001, None)])

if result_2023.success:
    alpha_opt_2023, beta_opt_2023 = result_2023.x
    print(f"Parámetros Gamma optimizados (2023):\n  alpha = {alpha_opt_2023:.4f}\n  beta = {beta_opt_2023:.4f}")
else:
    print("La optimización para 2023 no convergió.")
    print(result_2023.message)
    exit()

# 5. Ajustar beta por la inflación esperada para 2024
inflacion_2024 = 1.10  # 10% de inflación para 2024
beta_opt_2024 = beta_opt_2023 * inflacion_2024

print(f"\nBeta ajustado por inflación para 2024: {beta_opt_2024:.4f}")

# 6. Parámetros de la simulación
num_asegurados = 6700  # Número de asegurados
num_simulaciones = 1000  # Número de simulaciones de Monte Carlo
lambda_poisson = 0.7  # Tasa promedio de siniestros por persona

# 7. Función para simular la pérdida total para un asegurado (con deducible)
def simular_perdida_asegurado(alpha, beta, lambda_poisson, deducible):
    """Simula la pérdida total para un asegurado en un año, teniendo en cuenta un deducible."""
    num_siniestros = poisson.rvs(mu=lambda_poisson)  # Simula el número de siniestros
    perdida_total = 0
    for _ in range(num_siniestros):
        costo_siniestro = gamma.rvs(a=alpha, scale=beta)  # Simula el costo del siniestro
        if costo_siniestro > deducible:
            perdida_total += costo_siniestro - deducible  # Aplica el deducible
    return perdida_total

# 8. Función para encontrar el deducible que reduce la frecuencia en un 12%
def encontrar_deducible(alpha, beta, lambda_poisson, objetivo_reduccion=0.12):
    """Encuentra el deducible que reduce la frecuencia de siniestros en el porcentaje objetivo."""
    def frecuencia_reducida(deducible):
        num_simulaciones_deducible = 1000  # Número de simulaciones para estimar la frecuencia
        siniestros_con_deducible = 0
        for _ in range(num_simulaciones_deducible):
            #simulamos si el asegurado tiene algun siniestro
            perdida = simular_perdida_asegurado(alpha, beta, lambda_poisson, deducible)
            if perdida> 0:
                siniestros_con_deducible += 1
        frecuencia_con_deducible = siniestros_con_deducible / num_simulaciones_deducible
        reduccion = 1 - frecuencia_con_deducible
        return abs(reduccion - objetivo_reduccion)

    # Usar el promedio de 'Pago' como un valor inicial razonable para el deducible
    initial_deducible_guess = siniestros['Pago'].mean() / 2
    result = minimize(frecuencia_reducida, x0=initial_deducible_guess, bounds=[(0, siniestros['Pago'].max())])

    if result.success:
        deducible_optimo = result.x[0]
        print(f"\nDeducible optimo: {deducible_optimo:.4f}")
        return deducible_optimo
    else:
        print("\nNo se pudo encontrar el deducible optimo.")
        return None

# 9. Encontrar el deducible optimo
deducible_optimo = encontrar_deducible(alpha_opt_2023, beta_opt_2024, lambda_poisson)

# 10. Simulación de Monte Carlo (con deducible)
perdidas_agregadas = []
for i in range(num_simulaciones):
    perdida_simulacion = sum(simular_perdida_asegurado(alpha_opt_2023, beta_opt_2024, lambda_poisson, deducible_optimo) for _ in range(num_asegurados))
    perdidas_agregadas.append(perdida_simulacion)
    #Imprimir progreso
    if (i + 1) % 100 == 0:
      print(f"Simulación {i + 1}/{num_simulaciones} completada")

# 11. Análisis de los resultados
perdidas_agregadas = np.array(perdidas_agregadas)
percentil_50 = np.percentile(perdidas_agregadas, 50)
percentil_99_5 = np.percentile(perdidas_agregadas, 99.5)

print(f"\nPercentil 50 de la pérdida agregada (con deducible): {percentil_50:.2f}")
print(f"\nPercentil 99.5 de la pérdida agregada (con deducible): {percentil_99_5:.2f}")

# 12. Graficar la distribución de pérdidas agregadas (opcional)
plt.figure(figsize=(10, 6))
plt.hist(perdidas_agregadas, bins=30, color='skyblue')
plt.xlabel('Pérdida Agregada')
plt.ylabel('Frecuencia')
plt.title('Distribución de Pérdidas Agregadas Simuladas (con Deducible)')
plt.axvline(percentil_50, color='red', linestyle='dashed', linewidth=1, label=f'Percentil 50: {percentil_50:.2f}')
plt.axvline(percentil_99_5, color='green', linestyle='dashed', linewidth=1, label=f'Percentil 99.5: {percentil_99_5:.2f}')
plt.legend()
plt.show()