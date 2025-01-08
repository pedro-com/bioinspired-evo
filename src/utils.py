import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Tuple, Union
from random import choice, choices
from itertools import combinations

def alphabet_gene_type(alphabet_size:int):
    size = 8
    while alphabet_size > 1 << size and size != 64:
        size *= 2
    return f'uint{size}'

def plot_evolution_metric(ax:plt.Axes, metric_name:str, metric:Dict):
    styles = {"min": "r", "mean": "b", "max": "g", "best": "darkviolet"}
    fontdict = {"fontweight": "bold", "fontfamily": "DejaVu Sans"}
    n_generations = 0
    for k_name, m_values in metric.items():
        if k_name not in styles:
            print(f"Error metric {k_name} has no assigned color. Setting black")
            color = "black"
        else:
            color = styles[k_name]
        n_generations = max(len(m_values), n_generations)
        ax.plot(m_values, color=color, label=k_name.capitalize())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=True)
    ax.set_title(f"{metric_name.capitalize()} across Generations", fontsize=14, pad=20, fontdict=fontdict)
    ax.set_xlabel("Generations", fontsize=12, fontdict=fontdict)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12, fontdict=fontdict)
    ax.set_xlim(0, n_generations - 1)
    ax.grid(True)

def apply_restriction(points: np.ndarray, restriction: Tuple[str, float]):
    operation, value = restriction
    if operation == 'greq':
        return points >= value
    if operation == 'gr':
        return points > value
    if operation == 'lteq':
        return points <= value
    if operation == 'lt':
        return points < value
    raise NameError(f"The operation {operation} is not recognised ['greq', 'gr', 'lteq', 'lt]")

def domination_matrix(points: np.ndarray, restrict: np.ndarray=None, restriction: List[Tuple[str, float]]=None):
    d_matrix = np.zeros((points.shape[0], points.shape[0]))
    if restriction is None:
        for idx, p in enumerate(points):
            d_matrix[idx, :] = np.all(p >= points, axis=1) & np.all(p != points, axis=1)
        d_matrix[np.diag_indices(points.shape[0])] = True
        return d_matrix
    restriction_matrix = np.column_stack([apply_restriction(restrict[:, k], restriction[k]) for k in range(len(restriction))])
    valid_solutions = np.all(restriction_matrix, axis=1)
    for idx, p in enumerate(points):
        domination_mask = np.all(p >= points, axis=1) & np.all(p != points, axis=1)
        if valid_solutions[idx]:
            d_matrix[idx, :] = (valid_solutions[idx] != valid_solutions) | domination_mask
        else:
            d_matrix[idx, :] = (valid_solutions[idx] == valid_solutions) & domination_mask
    d_matrix[np.diag_indices(points.shape[0])] = True
    return d_matrix

def point_fronts(domination_mx: np.ndarray):
    p_ranges = domination_mx.sum(axis=0)
    p_fronts = np.zeros(shape=domination_mx.shape[0], dtype='uint16')
    front_id = 1
    point_count = 0
    while point_count < p_fronts.shape[0]:
        p_front = p_ranges == 1
        point_count += p_front.sum()
        p_fronts[p_front] = front_id
        front_id += 1
        p_ranges -= domination_mx[p_front, :].sum(axis=0)
    return p_fronts

def distance_matrix(points: np.ndarray):
    d_matrix = np.zeros((points.shape[0], points.shape[0]))
    for idx, point in enumerate(points[:-1]):
        d_matrix[idx, idx + 1:] = np.sqrt(np.sum((points[idx + 1:] - point)**2, axis=1))
        d_matrix[idx + 1:, idx] = d_matrix[idx, idx + 1:]
    return d_matrix

def sharing_distances(points: np.ndarray, p_front1: np.ndarray, niche_size: float):
    norm_points = norm_points = points / (points[p_front1].max(axis=0) - points[p_front1].min(axis=0))            
    def share_matrix(p_front: np.ndarray):
        sh_matrix = distance_matrix(norm_points[p_front])
        delta_mask = (sh_matrix >= 0) & (sh_matrix < niche_size)
        sh_matrix[delta_mask] = 1 - (sh_matrix[delta_mask] / niche_size)
        sh_matrix[np.logical_not(delta_mask)] = 0
        return sh_matrix
    return lambda p_front: 1 / share_matrix(p_front).sum(axis=0)

def crowding_distances(points: np.ndarray):
    def crowding_distance(p_front: np.ndarray):
        front = points[p_front]
        if front.shape[0] <= 3:
            return np.ones(front.shape[0]) * front.shape[1]
        normalization = front.max(axis=0) - front.min(axis=0)
        normalization[normalization == 0] = 1
        norm_front = front / normalization
        distances = np.zeros(front.shape[0])
        for k in range(front.shape[1]):
            sorted_args = np.argsort(front[:, k])
            distances[sorted_args[[0, -1]]] = front.shape[1]
            for i in range(1, front.shape[0]):
                if distances[sorted_args[i]] != front.shape[1]:
                    distances[sorted_args[i]] += norm_front[sorted_args[i + 1]][k] - norm_front[sorted_args[i - 1]][k]
        return distances
    return crowding_distance

def normalized_crowding_distances(points: np.ndarray):
    crowding_distances_func = crowding_distances(points)
    return lambda p_front: crowding_distances_func(p_front) / points.shape[1]

def merge_sorted_lists(l1: List[Any], l2: List[Any], key: Callable[[Any], Any]=lambda v: v):
    i, j = 0, 0
    while i < len(l1) and j < len(l2):
        if key(l1[i]) > key(l2[j]):
            l2.insert(i, l2[j])
            j += 1
        i += 1
    l1.extend(l2[j:])
    return l1

def roulette_selection(population: List, k: int, weight_keys: Callable[[Any], Any]=lambda v: v):
    if k >= len(population):
        return population
    population_left = len(population)
    cumulative_weights = np.cumsum(np.array([weight_keys(p) for p in population]))
    selected = []
    for _ in range(k):
        spin = np.random.uniform(0, cumulative_weights[population_left - 1])
        idx = np.searchsorted(cumulative_weights, spin, side='left')
        selected.append(population[idx])
        if idx != population_left - 1:
            cumulative_weights[idx:population_left - 1] = cumulative_weights[idx + 1:population_left] - cumulative_weights[idx]
        population_left -= 1
    return selected

def section_selection(population: List, k: int, point_keys: Callable[[Any], Any]=lambda v: v):
    if k >= len(population):
        return population
    points = np.array([point_keys(p) for p in population])
    idx_sort = np.random.randint(0, points.shape[1])
    sort_indexes = np.argsort(points[:, idx_sort])
    k_pos = np.random.randint(0, points.shape[0] - k)
    return [population[idx] for idx in sort_indexes[k_pos: k_pos + k]]

def probability_selection(values: Union[Any, List[Any], List[Tuple[Any, float]]]):
    if not isinstance(values, (list, tuple)) or not values:
        return values
    if not isinstance(values[0], (list, tuple)):
        return choice(values)
    return choices([v[0] for v in values], weights=[v[1] for v in values], k=1)

def cluster_points(points: np.ndarray, k_clusters: int, distance_mx: np.ndarray=None):
    if points.shape[0] <= k_clusters:
        return points
    if distance_mx == None:
        distance_mx = distance_matrix(points)
    sort_dist = np.argsort(distance_mx.sum(axis=0))
    return points[sort_dist[-k_clusters:]]

def list_combinations(values: List, exclude_from: tuple=()):
    combs = []
    for k in range(1, len(values) + 1):
        if k in exclude_from:
            continue
        if k == 1:
            combs.extend(values[:])
        elif k == len(values):
            combs.append(values)
        else:
            combs.extend(combinations(values, k))
    if not combs:
        return values
    return combs
    
def filter_restrictions(points: np.ndarray, restrictions: List[Tuple[float, float]]=None):
    mask = np.ones(points.shape[0], dtype=bool)
    if restrictions is None:
        return mask
    for idx, rest in enumerate(restrictions):
        if rest is None:
            continue
        if rest[0] is not None:
            mask = mask & (points[:, idx] >= rest[0])
        if rest[1] is not None:
            mask = mask & (points[:, idx] <= rest[1])
    return mask
