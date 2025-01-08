from typing import Callable, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def apply_restriction(points: np.ndarray, restriction: Tuple[str, float]):
    operation, value = restriction
    if operation == 'greq':
        return points < value
    if operation == 'gr':
        return points <= value
    if operation == 'lteq':
        return points > value
    if operation == 'lt':
        return points >= value
    raise NameError(f"The operation {operation} is not recognised ['greq', 'gr', 'lteq', 'lt]")

def penalized_fit(fit: Callable[[Any], np.ndarray], maximize: Tuple[bool], restrictions: List[Tuple[str, float]], epsilon: float=1.):
    maximize_penalization = np.array([-1 if m else 1 for m in maximize])
    def fit_pen(points: np.ndarray):
        scores, f_restrictions = fit(points)
        if len(f_restrictions.shape) > 1:
            restriction_matrix = np.column_stack([apply_restriction(f_restrictions[:, k], restrictions[k])
                                              for k in range(len(restrictions))])
            penalizations = epsilon * np.where(restriction_matrix, np.abs(f_restrictions), 0).sum(axis=1)
            penalizations = np.outer(penalizations, maximize_penalization)
            return scores + penalizations
        restriction_matrix = np.array([apply_restriction(f_restrictions[k], restrictions[k])
                                       for k in range(len(restrictions))], dtype=bool)
        penalizations = np.sum(np.abs(f_restrictions[restriction_matrix]))
        penalizations = np.outer(penalizations, maximize_penalization).reshape(-1)
        return scores + penalizations
    return fit_pen

def dominance_matrix_diff(a, b):
    diff = np.array(a) - np.array(b)
    
    if np.all(diff >= 0) and not np.array_equal(a, b):
        return 1
    else:
        return 0
    
def get_dominance_matrix(coordinates):
    dominance_matrix = np.zeros((len(coordinates), len(coordinates)))
    for i, elem1 in enumerate(coordinates):
        for j, elem2 in enumerate(coordinates):
            dominance_matrix[i,j] = dominance_matrix_diff(elem1, elem2)
    return dominance_matrix

def plot_pareto_fronts(frentes):
    """
    Grafica los frentes de Pareto a partir de una lista de arrays de numpy.
    
    Args:
        frentes (list): Lista de arrays de numpy, donde cada array contiene puntos pertenecientes a un frente.
    """
    colores = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    plt.figure(figsize=(10, 6))

    for idx, (pts, color) in enumerate(zip(frentes, colores), start=1):
        plt.scatter(pts[:, 0], pts[:, 1], color=color, label=f"Frente {idx}", s=100)

    # Configuración de la gráfica
    plt.title('Frentes de Pareto')
    plt.xlabel('Objetivo f1(x)')
    plt.ylabel('Objetivo f2(x)')
    plt.legend()
    plt.show()


def plot_pareto_fronts_3d(frentes):
    """
    Grafica los frentes de Pareto en 3D a partir de una lista de arrays de numpy utilizando Plotly.

    Args:
        frentes (list): Lista de arrays de numpy, donde cada array contiene puntos pertenecientes a un frente.
                        Cada punto debe tener tres coordenadas para los objetivos.
    """
    colores = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    fig = go.Figure()

    for idx, (pts, color) in enumerate(zip(frentes, colores), start=1):
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f"Frente {idx}"
        ))

    # Configuración de la gráfica
    fig.update_layout(
        title="Frentes de Pareto en 3D",
        scene=dict(
            xaxis_title="Objetivo f1(x)",
            yaxis_title="Objetivo f2(x)",
            zaxis_title="Objetivo f3(x)"
        ),
        legend=dict(title="Frentes")
    )

    fig.show()

def get_pareto_fronts(coordinates):
    coordinates_copy = coordinates.copy()
    fronts = []

    while len(coordinates_copy) > 1:
        dominance_matrix = get_dominance_matrix(coordinates_copy)
        dominated_count = dominance_matrix.sum(axis=0)
        current_front = np.where(dominated_count == 0)[0]
        pareto_points = coordinates_copy[current_front]
        fronts.append(pareto_points)
        coordinates_copy = np.delete(coordinates_copy, (current_front), axis=0)

    return fronts

def get_point_ranges(coordinates):
    p_ranges = get_dominance_matrix(coordinates).sum(axis=0)
    ranges = []
    for k in np.unique(p_ranges):
        ranges.append(coordinates[p_ranges == k])
    return ranges

def calcular_distancia_crowding(frente):
    """
    Calcula las distancias de crowding para un frente de Pareto.

    Args:
        frente (np.ndarray): Array de puntos (n x m), donde n es el número de puntos y m el número de objetivos.

    Returns:
        list: Lista de distancias de crowding para cada punto del frente.
    """
    n_puntos = len(frente)
    if n_puntos < 2:
        # Si hay menos de dos puntos, no se puede calcular crowding
        return [float('inf')] * n_puntos

    # Inicializar las distancias de crowding
    distancia_frente = [0] * n_puntos

    for k in range(frente.shape[1]):  # Para cada objetivo
        # Ordenar los puntos según el objetivo k
        args = np.argsort(frente[:, k])

        # Asignar infinito a los extremos
        distancia_frente[args[0]] = float('inf')
        distancia_frente[args[-1]] = float('inf')

        # Calcular contribuciones para los puntos internos
        for i in range(1, n_puntos - 1):
            distancia_frente[args[i]] += (
                (frente[args[i + 1], k] - frente[args[i - 1], k]) /
                (frente[args[-1], k] - frente[args[0], k])
                if frente[args[-1], k] != frente[args[0], k] else 0
            )

    return distancia_frente

def get_crowding_distances(fronts):
    distancias = []
    for frente in fronts:
        distancias.append(calcular_distancia_crowding(frente))
    return distancias

def plot_pareto_front(pareto_front, obtained_results, algorithm_name:str, problem_name:str, ylim=None, xlim=None):
    fig, axs = plt.subplots(1, 1)
    if ylim:
        axs.set_ylim(ylim[0], ylim[1])
    if xlim:
        axs.set_xlim(xlim[0], xlim[1])
    axs.scatter(pareto_front[:, 0], pareto_front[:, 1], label= problem_name + ' Pareto')
    axs.scatter(obtained_results[:, 0], obtained_results[:, 1], label= algorithm_name + ' Obtained Pareto')
    plt.legend()
    plt.show()

def plot_pareto_front_3d(pareto_front, obtained_results, algorithm_name=str, problem_name=str):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pareto_front[:, 0], y=pareto_front[:, 1], z=[0]*pareto_front.shape[0], mode='markers', name=problem_name + ' Pareto'))
    fig.add_trace(go.Scatter3d(x=obtained_results[:, 0], y=obtained_results[:, 1], z=[0]*obtained_results.shape[0], mode='markers', name=algorithm_name + ' Obtained Pareto'))
    fig.show()

def plot_multiple_pareto_fronts(fronts_dict, title="Pareto Fronts Comparison", cmap='Dark2', axs:plt.Axes=None, ylim=None, xlim=None):
    cmap = plt.get_cmap(cmap)
    if axs is None:
        fig, axs= plt.subplots(1, 1, figsize=(10, 6))
    if ylim:
        axs.set_ylim(ylim[0], ylim[1])
    if xlim:
        axs.set_xlim(xlim[0], xlim[1])
    n_fronts = len(fronts_dict)
    colors = [cmap(i / (n_fronts - 1) if n_fronts > 1 else 0) for i in range(n_fronts)]
    
    # Define a list of interesting markers
    markers = ['*',    # star
              'o',     # circle
              '^',     # triangle up
              's',     # square
              'D',     # diamond
              'v',     # triangle down
              'p',     # pentagon
              'h',     # hexagon
              '8',     # octagon
              'P']     # plus (filled)
    
    # Plot each front with a different color and marker
    for i, (front_name, points) in enumerate(fronts_dict.items()):
        marker_idx = i % len(markers)  # Cycle through markers if more fronts than markers
        axs.scatter(points[:, 0], 
                   points[:, 1],
                   c=[colors[i]], 
                   marker=markers[marker_idx],
                   s=50,  # Increased marker size for better visibility
                   label=front_name,
                   alpha=0.7)
    
    axs.set_title(title)
    axs.set_xlabel("Objective 1")
    axs.set_ylabel("Objective 2")
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent legend cutoff
    if axs is None:
        plt.tight_layout()
