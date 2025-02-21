from dataclasses import dataclass
from typing import List, Callable, Tuple, Literal, Union
import numpy as np
from random import sample, choice

from ..utils import (
    domination_matrix,
    point_fronts,
    sharing_distances,
    crowding_distances,
    normalized_crowding_distances,
    merge_sorted_lists,
    roulette_selection,
    section_selection,
    probability_selection,
    filter_restrictions
)
from ..evaluation import MultiObjectiveEvaluation
from ..evolutive import Evolutive

'''WARNING
This class is meant to maximize both functions, if you want to minimize a function pass -f(x) as the valid function
'''
SelectionPool = Literal['best', 'roulette', 'section', 'random']
Front = Literal['range', 'front']
Penalization = Literal['sharing', 'crowding', 'crowding_norm']
@dataclass
class MultiEvolutive(Evolutive):
    maximize: Tuple[bool]
    selection_pool: Union[SelectionPool, List[SelectionPool], List[Tuple[SelectionPool, float]]] = 'best'
    selection_pool_size: float = 0.8
    front: Union[Front, List[Front], List[Tuple[Front, float]]] = 'range'
    penalization: Union[Penalization, List[Penalization], List[Tuple[Penalization, float]]] = 'sharing'
    niche_sharing_size: float = 0.8
    elitism: bool = False
    steps_to_reduce_p_elite: int = 100
    fit_penalization: List[Tuple[float, float]] = None
    penalize_fit_restriction: float = 2.
    restrictions: List[Tuple[str, float]] = None
    evaluation_metrics: MultiObjectiveEvaluation = None

    def __post_init__(self):
        self.n_selection_individuals = int(np.ceil(self.selection_pool_size*self.n_individuals))
        self.maximize_correction = np.array([1 if m else -1 for m in self.maximize])

    def reduce_elite(self, p_elite: List[np.ndarray]):
        # Obtain domination ranges
        p_range = domination_matrix(np.array([indv[1] for indv in p_elite])).sum(axis=0)
        return [p_elite[k] for k in np.flatnonzero(p_range == 1)]

    def _front(self, fit_scores: np.ndarray, restrict: np.ndarray):
        front = probability_selection(self.front)
        domination_mx = domination_matrix(fit_scores, restrict=restrict, restriction=self.restrictions)
        if front == 'front':
            return domination_mx, point_fronts(domination_mx)
        return domination_mx, domination_mx.sum(axis=0)

    def _penalization(self, fit_scores: np.ndarray, p_front: np.ndarray):
        penalization = probability_selection(self.penalization)
        if penalization == 'crowding':
            return crowding_distances(fit_scores)
        elif penalization == 'crowding_norm':
            return normalized_crowding_distances(fit_scores)
        return sharing_distances(fit_scores, p_front == 1, self.niche_sharing_size)

    def _selection_pool(self, p_selection: List, n_ind: int):
        selection_pool = probability_selection(self.selection_pool)
        if selection_pool == 'best':
            return sorted(p_selection, key=lambda v: v[-1])[-n_ind:]
        elif selection_pool == 'roulette':
            return roulette_selection(p_selection, n_ind, weight_keys=lambda v: v[-1])
        elif selection_pool == 'section':
            return section_selection(p_selection, n_ind, point_keys=lambda v: v[1])
        return sample(p_selection, n_ind)

    def calculate_fit(self, fit: Tuple[Callable], population: List[np.ndarray], p_elite: List):
        # Obtain fit
        if self.restrictions is not None:
            scores = [fit(self.apply_phenotype(p)) for p in population]
            fit_scores = np.array([sc[0] for sc in scores]) * self.maximize_correction
            restrict = np.array([sc[1] for sc in scores])
        else:
            fit_scores = np.array([fit(self.apply_phenotype(p)) for p in population]) * self.maximize_correction
            restrict = None
        if self.elitism and p_elite:
            population.extend(p[0] for p in p_elite)
            fit_scores = np.concatenate([fit_scores, np.array([p[1] for p in p_elite])], axis=0)
            if self.restrictions is not None:
                restrict = np.concatenate([restrict, np.array([p[2] for p in p_elite])], axis=0)
        # Obtain domination matrix and fronts
        domination_mx, p_fronts = self._front(fit_scores, restrict)
        strength_ranges = domination_mx.sum(axis=1)
        # Obtain penalization function
        penalization_func = self._penalization(fit_scores, p_fronts)
        # Obtain the points multiobjective scores by front
        acc_points = 0
        total_points = len(population)
        for front in np.unique(p_fronts):
            p_front = p_fronts == front
            n_points = p_front.sum()
            fit_front = (total_points - (n_points - 1) / 2 - acc_points) / (total_points / 2)
            penalization = penalization_func(p_front)
            strength_fr = strength_ranges[p_front] / (total_points + 1)
            multi_fit = (fit_front + strength_fr) * penalization
            if self.fit_penalization is not None:
                rest_mask = filter_restrictions(fit_scores[p_front], self.fit_penalization)
                multi_fit[rest_mask] /= self.penalize_fit_restriction
            if restrict is not None:
                yield [(population[p_idx], fit_scores[p_idx], restrict[p_idx], m_fit)
                       for (p_idx, m_fit) in zip(np.flatnonzero(p_front), multi_fit) if m_fit > 0.]
            yield [(population[p_idx], fit_scores[p_idx], m_fit)
                   for (p_idx, m_fit) in zip(np.flatnonzero(p_front), multi_fit) if m_fit > 0.]
            # The last check is specific for functions that have near points with the same fit score
            acc_points += n_points
    
    def fit_population(self, fit: Callable, population: List[np.ndarray], p_elite: List):
        fit_pop = self.calculate_fit(fit, population, p_elite)
        p_selection = next(fit_pop, [])
        p_elite = p_selection[:]
        p_elite.sort(key=lambda v: tuple(v[1]))
        if len(p_selection) > self.n_selection_individuals:
            return p_elite, self._selection_pool(p_selection, self.n_selection_individuals)
        while len(p_selection) < self.n_selection_individuals:
            p_add = next(fit_pop, [])
            if not p_add:
                break
            if len(p_add) + len(p_selection) < self.n_selection_individuals:
                p_selection.extend(p_add)
                continue
            p_selection.extend(self._selection_pool(p_add, self.n_selection_individuals - len(p_selection)))
        return p_elite, p_selection
    
    def select(self, fit_population: List):
        T_sample = sample(fit_population, k=self.T_selection)
        best_individual = T_sample[0]
        for individual in T_sample[1:]:
            best_individual = max(best_individual, individual, key=lambda v: v[-1])
        return best_individual[0]

    def selection(self, fit_population: List):
        """
        Selects T random individuals from the population and obtains the one with the best values.
        """
        select = lambda: self.select(fit_population)
        return ((select(), select()) for _ in range(self.n_individuals // 2))

    def evolve(self, fit: Callable, n_generations:int, trace:int=0, obtain_metrics: bool=False, seed_population: List[np.ndarray]=None):
        population = self.generate_population(seed_population)
        if obtain_metrics:
            evolution_metrics = {
                metric: {name: np.zeros(n_generations, dtype=np.float64) for name in ("min", "mean", "max")}
                for metric in ("fit", "diversity")
            }
        last_update = (0, 0)
        p_elite = []
        for generation in range(1, n_generations + 1):
            # Obtain the elite population and the selected population for reproduction
            p_elite_new, fit_population = self.fit_population(fit, population, p_elite)
            if len(p_elite) != last_update[1]:
                last_update = (generation, len(p_elite))
            # Metrics
            if obtain_metrics:
                for metric, values in self.calculate_metrics(fit_population).items():
                    for m_name, value in values.items():
                        evolution_metrics[metric][m_name][generation - 1] = value
            if not self.elitism:
                p_elite = merge_sorted_lists(p_elite, p_elite_new, key=lambda v: tuple(v[1]))
                # Add to elite population if elitism is not selected
                if generation % self.steps_to_reduce_p_elite == 0 and generation != 0:
                    p_elite = self.reduce_elite(p_elite)
            else:
                p_elite = p_elite_new
            # Trace
            if trace != 0 and generation % trace == 0:
                print(self.trace_metrics(generation, p_elite))
            # Selection + Crossover + Mutation
            population = []
            for par1, par2 in self.selection(fit_population):
                child1, child2 = self.crossover.crossover(par1, par2)
                population.append(self.mutation.mutate(child1))
                population.append(self.mutation.mutate(child2))
            # For uneven populations
            if self.n_individuals % 2 == 1:
                population.append(self.mutation.mutate(self.select(fit_population)))
        if not self.elitism:
            p_elite = self.reduce_elite(p_elite)
        # Trace
        if trace != 0 and generation % trace != 0:
            print(self.trace_metrics(generation, p_elite))
        results = {
            "best": [self.apply_phenotype(indv[0]) for indv in p_elite],
            "best_cromosome": [indv[0] for indv in p_elite],
            "population": population,
            "last_update": last_update[0]
        }
        if obtain_metrics:
            results["evolution_metrics"] = evolution_metrics
        return results

    def trace_metrics(self, generation: int, p_elite: List):
        def_out = f"Generation {generation:8d} | Population Elite Points: {len(p_elite):8d}"
        if self.evaluation_metrics is None:
            elite_scores = np.array([indv[2] for indv in p_elite])
            return f"{def_out} | Best Fit: {elite_scores.max():4.4f} | Mean Fit: {elite_scores.mean():4.4f}"
        return f"{def_out} | {self.evaluation_metrics.evaluation_str(np.array([indv[1] for indv in p_elite]) * self.maximize_correction)}"

    def calculate_metrics(self, fit_population: List):
        diversity, _ = self.calculate_diversity([indv[0] for indv in fit_population])
        scores = np.array([indv[-1] for indv in fit_population])
        if len(scores.shape) == 0:
            return {
                "fit": {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                },
                "diversity": {
                    "min": diversity.min(),
                    "max": diversity.max(),
                    "mean": np.mean(diversity),
                }
            }
        return {
            "fit": {
                "min": scores.min(),
                "max": scores.max(),
                "mean": scores.mean(),
            },
            "diversity": {
                "min": diversity.min(),
                "max": diversity.max(),
                "mean": np.mean(diversity),
            }
        }


'''
    def multi_fit(self, fit: Tuple[Callable], population: List[np.ndarray]):
        # Obtain fit
        fit_scores = np.array([[f(self.apply_phenotype(p)) for f in fit] for p in population])
        # Obtain domination ranges
        multi_fit = np.zeros((fit_scores.shape[0], 1))
        # Obtain domination ranges
        dom_matrix = domination_matrix(fit_scores)
        point_ranges = dom_matrix.sum(axis=0)
        strength_ranges = dom_matrix.sum(axis=1)
        # Obtain normalization ranges
        p_1 = point_ranges == 1
        norm_fit_scores = fit_scores / (fit_scores[p_1].max(axis=0) - fit_scores[p_1].min(axis=0))
        accumulated_agg_fit = 0
        for r in point_ranges.unique().sort():
            p_r = point_ranges == r
            r_indv = p_r.sum()
            fit_r = self.n_individuals - (r_indv - 1) / 2 - accumulated_agg_fit
            count_sh = share_matrix(norm_fit_scores[p_r], self.alpha_share, self.delta_share).sum(axis=0)
            strength_r = strength_ranges[p_r] / (self.n_individuals + 1)
            multi_fit[p_r] = fit_r / count_sh + strength_r
            accumulated_agg_fit += r_indv
        return [(p, r_p, m_f) for p, r_p, m_f in zip(population, point_ranges, multi_fit)]
'''