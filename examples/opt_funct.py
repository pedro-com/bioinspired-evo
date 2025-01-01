import numpy as np
from pymoo.problems import get_problem

def zdt1(x):
    problem = get_problem("zdt1")
    return problem.evaluate(x).tolist()

def zdt3(x):
    problem = get_problem("zdt3")
    return problem.evaluate(x).tolist()

# Custom al tener restricciones
def mw7(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - (f1 / g)**0.5)
    c1 = np.sum(x) - 1.5
    c2 = 0.5 - np.sum(x[:-1])
    return [f1, f2], [c1, c2]

def mw14(x):
    problem = get_problem("mw14", n_var=30, n_obj=3)
    return problem.evaluate(x).tolist()

# Custom, TSP no está en los problemas por defecto de pymoo
def tsp_mo(x, distance_matrix, time_matrix):
    # Aseguraar permutación válida
    path = np.argsort(x)
    total_distance = sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    total_time = sum(time_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    # Añadir la vuelta a la ciudad de salida
    total_distance += distance_matrix[path[-1], path[0]]
    total_time += time_matrix[path[-1], path[0]]
    return [total_distance, total_time]

def get_fit_function(function, non_zero_divisor=False):
    def fit(x):
        divisor_adjustment = 1e-10 if non_zero_divisor else 0  # Avoid division by zero
        return 1 / (1 + np.sum(function(x)) + divisor_adjustment)
    return fit

# Ejemplo
# zdt1_fit = get_fit_function(zdt1)
# mw7_fit = get_fit_function(lambda x: mw7(x)[0])  # Only objectives, exclude constraints