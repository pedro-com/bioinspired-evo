from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Dict
from pymoo.vendor.vendor_coco import COCOProblem
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV

from ..utils import distance_matrix, cluster_points, filter_restrictions
import numpy as np

class Metric(ABC):
    @abstractmethod
    def evaluation(self, scores:np.ndarray, is_sorted: bool=True) -> Dict[str, Any]:
        pass

    @abstractmethod
    def __call__(self, results: List):
        pass

    @abstractmethod
    def evaluation_metric_names(self):
        pass

    def evaluation_str(self, scores: np.ndarray, is_sorted: bool=True):
        evaluation_dict = self.evaluation(scores, is_sorted)
        return " | ".join([f'{k.capitalize()}: {v:6.4f}' for k, v in evaluation_dict.items()])

@dataclass
class EvaluationMetric(Metric):
    evaluate: Callable[[Any], Any]
    maximize: bool

    def evaluation(self, scores: np.ndarray, is_sorted: bool=True):
        return {
            "max": np.max(scores),
            "min": np.min(scores),
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores)
        }

    def evaluation_metric_names(self):
        return ("max", "min", "mean", "median", "std")

    def __call__(self, results: List):
        if len(results) == 0:
            return {}
        eval_results = [(res, self.evaluate(res)) for res in results]
        eval_results.sort(key=lambda v: v[1], reverse=not self.maximize)
        evaluation_metrics = self.evaluation(np.array([res[1] for res in eval_results]))
        evaluation_metrics["eval_results"] = eval_results
        return evaluation_metrics

def gd_plus_metric(pareto_front: np.ndarray):
    restrictions = np.column_stack([pareto_front.min(axis=0), pareto_front.max(axis=0)])
    gd_plus_eval = GDPlus(pareto_front)
    def gd_plus(points: np.ndarray, distance_mx: np.ndarray=None):
        points = points[filter_restrictions(points, restrictions)]
        # for idx in range(pareto_front.shape[1]):
        #     points = points[(points[:, idx] >= min_lim[idx]) & (points[:, idx] <= max_lim[idx])]
        return gd_plus_eval(points)
    return gd_plus

def igd_plus_metric(pareto_front: np.ndarray):
    restrictions = np.column_stack([pareto_front.min(axis=0), pareto_front.max(axis=0)])
    igd_plus_eval = IGDPlus(pareto_front)
    def igd_plus(points: np.ndarray, distance_mx: np.ndarray=None):
        points = points[filter_restrictions(points, restrictions)]
        return igd_plus_eval(points)
    return igd_plus

def hv_metric(reference_point: np.ndarray, maximize: Tuple[bool]):
    hv_eval = HV(ref_point=np.array(reference_point))
    def hv(points: np.ndarray, distance_mx: np.ndarray=None):
        for idx, v in enumerate(reference_point):
            if maximize[idx]:
                points = points[points[:, idx] <= v]
            else:
                points = points[points[:, idx] >= v]
        return hv_eval(points)
    return hv

def save_min(line: np.ndarray):
    return line.min() if line.size > 0 else 0.

def zitler_dispersion(points: np.ndarray=None, distance_mx:np.ndarray=None):
    '''
    Uniformity of a set of solutions (e.g., Pareto fronts). It ensures that the solutions are evenly spread out
    in the objective space.
    '''
    if points is None and distance_mx is None:
        return 0
    if distance_mx is None:
        distance_mx = distance_matrix(points)
    if distance_mx.shape[0] == 0:
        return 0
    min_distances = np.array([save_min(lin[lin > 0]) for lin in distance_mx])
    sum_min = min_distances.sum()
    if sum_min == 0:
        return 0
    avg_min = sum_min / points.shape[0]
    return np.abs(min_distances - avg_min).sum() / sum_min

def spa(points: np.ndarray=None, distance_mx:np.ndarray=None):
    '''
    Schott Spacing (SPA) evaluates the uniformity of a set of solutions (e.g., along a Pareto front). It measures the
    average spacing of solutions and quantifies the deviation from the mean spacing.
    '''
    if points is None and distance_mx is None:
        return 0
    if distance_mx is None:
        distance_mx = distance_matrix(points)
    if distance_mx.shape[0] == 0:
        return 0
    min_distances = np.array([save_min(lin[lin > 0]) for lin in distance_mx])
    sum_min = min_distances.sum()
    if sum_min == 0:
        return 0
    avg_min = sum_min / points.shape[0]
    return np.sqrt(np.sum((min_distances - avg_min)**2) / (points.shape[0] - 1))

def hrs(points: np.ndarray=None, distance_mx:np.ndarray=None, is_sorted: bool=True):
    '''
    Hole Relative Size (HRS) metric evaluates the gaps or "holes" in a Pareto front in multiobjective optimization.
    The metric quantifies the relative size of the largest gap (hole) in comparison to the range of the Pareto front.
    '''
    if points is None and distance_mx is None:
        return 0
    if distance_mx is None:
        if not is_sorted:
            points.sort(axis=0)
        distance_mx = distance_matrix(points)
    if distance_mx.shape[0] == 0:
        return 0
    d_max = distance_mx.diagonal(offset=-1).max()
    R = np.sqrt(np.sum(np.ptp(points, axis=0)**2))
    return d_max / R if R > 0 else 0

@dataclass
class MultiObjectiveEvaluation(Metric):
    '''
    Obtains several metrics to evaluate the functionality of a multiobjective algorithm.
    - Generational Distance+: Distance from solution to Pareto Front (problem from pymoo is needed | \
        <https://pymoo.org/misc/indicators.html>).
    - Inverted Generational Distance+: Inverts the generational distance and measures the distance from \
        any point in Z to the closest in A (problem from pymoo is needed | <https://pymoo.org/misc/indicators.html>).
    - Hypervolume: Hypervolume requires a reference point, and obtains the area / volume of points dominated \
        by the solution (reference_point is required).
    - Zitler Dispersion: Obtains the dispersion of the surface of the front.
    - Schott Spacing (SPA) evaluates the uniformity of a set of solutions.

    The variable lims limits the area of the points that are considered for the evaluation.
    It must be a list of tuples for each dimension. If a dimension is not meant to be limited, set None for the dimension.
    '''
    evaluate: Callable[[Any], np.ndarray]
    problem: COCOProblem=None
    reference_point: List=None
    maximize: Tuple[bool]=None
    limits: Tuple=None
    restrictions: bool = False

    def __post_init__(self):
        self.metric_funcs = {
            "ZitlerD": zitler_dispersion,
            "SPA": spa,
            "HRS": hrs
        }
        if self.problem is not None:
            self.metric_funcs["GDPlus"] = gd_plus_metric(self.problem.pareto_front())
            self.metric_funcs["IGDPlus"] = igd_plus_metric(self.problem.pareto_front())
        if self.reference_point:
            self.reference_point = np.array(self.reference_point)
            if self.maximize is None:
                self.maximize = np.ones(self.reference_point.shape[0], dtype='bool')
            self.metric_funcs["HV"] = hv_metric(self.reference_point, self.maximize)

    def evaluation_metric_names(self):
        return tuple(self.metric_funcs.keys())

    def filter_limits(self, scores: np.ndarray):
        if self.limits is None:
            return scores
        for idx, lims in enumerate(self.limits):
            if lims != None:
                scores = scores[(scores[:, idx] >= lims[0]) & (scores[:, idx] <= lims[1])]
        return scores
    def evaluation(self, scores: np.ndarray, is_sorted: bool=True):
        if self.limits is not None:
            for idx, lims in enumerate(self.limits):
                if lims != None:
                    scores = scores[(scores[:, idx] >= lims[0]) & (scores[:, idx] <= lims[1])]
        if not is_sorted:
            scores.sort(axis=0)
        distance_mx = distance_matrix(scores)
        return {k: metric(points=scores, distance_mx=distance_mx) for k, metric in self.metric_funcs.items()}

    def __call__(self, results: List):
        if len(results) == 0:
            return {}
        if self.restrictions:
            eval_results = [sorted([tuple(self.evaluate(val)[0]) for val in res]) for res in results]
        else:
            eval_results = [sorted([tuple(self.evaluate(val)) for val in res]) for res in results]
        filt_eval = [self.filter_limits(np.array(res)) for res in eval_results]
        res_lens = [len(res) for res in filt_eval]
        k_cluster_points = min(res_lens)
        mean_points = sum(res_lens) / len(eval_results)
        mean_point_front = np.zeros((k_cluster_points, len(eval_results[0][0])))
        evaluation_results = []
        mean_scores = {k: 0. for k in self.metric_funcs}
        for idx, e_results in enumerate(eval_results):
            points = np.array(e_results)
            res_evaluation = self.evaluation(points)
            for k in mean_scores:
                mean_scores[k] += res_evaluation[k]
            mean_point_front += cluster_points(filt_eval[idx], k_cluster_points)
            res_evaluation["results"] = results[idx]
            res_evaluation["scores"] = e_results
            evaluation_results.append(res_evaluation)
        for k in mean_scores:
            mean_scores[k] /= len(eval_results)
        mean_scores["evaluation_results"] = evaluation_results
        mean_scores["mean_front"] = mean_point_front / len(eval_results)
        mean_scores["mean_points"] = mean_points
        return mean_scores

'''
def multiobjective_metrics(problem: COCOProblem=None, reference_point: List=None, limits: Tuple=None):
    Obtains several metrics to evaluate the functionality of a multiobjective algorithm.
    - Generational Distance+: Distance from solution to Pareto Front (problem from pymoo is needed | <https://pymoo.org/misc/indicators.html>).
    - Inverted Generational Distance+: Inverts the generational distance and measures the distance from any point in Z to the closest in A (problem from pymoo is needed | <https://pymoo.org/misc/indicators.html>).
    - Hypervolume: Hypervolume requires a reference point, and obtains the area / volume of points dominated by the solution (reference_point is required).
    - lims: Limits the area of the points that are considered for the evaluation. It must be a list of tuples for each dimension. If a dimension is not meant to be limited, set None for the dimension.
    - Zitler Dispersion: Obtains the dispersion of the surface of the front.
    metric_funcs = {
        "ZitlerD": zitler_dispersion,
        "SPA": spa,
        "HRS": hrs
    }
    if problem is not None:
        metric_funcs["GDPlus"] = lambda points, distance_mx: GDPlus(problem)(points)
        metric_funcs["IGDPlus"] = lambda points, distance_mx: IGDPlus(problem)(points)
    if reference_point != None:
        metric_funcs["HV"] = lambda points, distance_mx: HV(ref_point=np.array(reference_point))(points)
    def metrics(points: np.ndarray, is_sorted: bool=False):
        if limits is not None:
            for idx, lims in enumerate(limits):
                if lims != None:
                    points = points[points[:, idx] >= lims[0] & points[:, idx] <= lims[1]]
        if not is_sorted:
            points.sort(axis=0)
        distance_mx = distance_matrix(points)
        return {k: metric(points=points, distance_mx=distance_mx) for k, metric in metric_funcs.items()}
    return metrics
'''