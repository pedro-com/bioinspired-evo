import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Tuple
from pymoo.vendor.vendor_coco import COCOProblem
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV

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

def domination_matrix(points: np.ndarray):
    d_matrix = np.zeros((points.shape[0], points.shape[0]), dtype='bool')
    for idx, p in enumerate(points):
        d_matrix[idx, :] = np.all(p >= points, axis=1) & np.all(p != points, axis=1)
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
            return np.ones(front.shape[0])
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


# def crowding_distances(fit_scores: np.ndarray):
#     a = 0
'''
def share_matrix(normalized_fit: np.ndarray, niche_size: float):
    sh_matrix = distance_matrix(normalized_fit)
    delta_mask = np.logical_and(0 <= sh_matrix, sh_matrix < niche_size)
    sh_matrix[delta_mask] = 1 - (sh_matrix[delta_mask] / niche_size)
    sh_matrix[np.logical_not(delta_mask)] = 0
    return sh_matrix

def point_fronts(points: np.ndarray, fit_scores: np.ndarray, domination_mx: np.ndarray=None):
    domination_mx = domination_matrix(fit_scores) if domination_mx is None else np.copy(domination_mx)
    fronts = []
    total_point_count = 0
    p_fronts = np.zeros(shape=points.shape[0], dtype='uint16')
    while total_point_count < points.shape[0]:
        actual_front = domination_mx.sum(axis=0) == 1
        p_count = actual_front.sum()
        p_fronts[actual_front] = len(fronts)
        fronts.append((points[actual_front], fit_scores[actual_front],
                       points.shape[0] - (p_count - 1) / 2 - total_point_count))
        total_point_count += p_count
        domination_mx = domination_mx[:, np.logical_not(actual_front)]
    return fronts, p_fronts

def point_ranges(points: np.ndarray, fit_scores: np.ndarray, domination_mx: np.ndarray=None):
    if domination_mx is None:
        domination_mx = domination_matrix(points)
    p_ranges = domination_mx.sum(axis=0)
    ranges = []
    total_point_count = 0
    for k in np.unique(p_ranges):
        actual_front = p_ranges == k
        p_count = actual_front.sum()
        ranges.append((points[actual_front], fit_scores[actual_front],
                       points.shape[0] - (p_count - 1) / 2 - total_point_count))
        total_point_count += p_count
    return ranges, p_ranges - 1
'''