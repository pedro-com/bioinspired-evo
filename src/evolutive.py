from abc import abstractmethod, ABC
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Any, Union, List, Dict, Type
from random import sample
import numpy as np
from functools import reduce

from .utils import plot_evolution_metric
from .crossover import Crossover, MULTI_CROSSOVER, MultiCrossover
from .mutation import Mutation, MULTIMUTATION_DICT, MultiMutation

@dataclass
class Evolutive(ABC):
    n_individuals: int
    mutation: Union[Mutation, MultiMutation]
    crossover: Union[Crossover, MultiCrossover]
    phenotype: Callable
    gene_type: str
    T_selection: int

    @classmethod
    def get_mutation(cls, mutation_name: Union[str, Tuple[str]], multi_mutation: str,
                     mutation_dict: Dict[str, Type[Mutation]], mutation_kwargs: Dict[str, Any]
                     ):
        """Given either a mutation name or a list of mutations, return the associated Mutation class"""
        if isinstance(mutation_name, str):
            assert mutation_name in mutation_dict, f"Invalid mutation name as its not in the passed mutation dictionary: {list(mutation_dict.keys())}"
            return mutation_dict[mutation_name](**mutation_kwargs)
        assert all(mut_name in mutation_dict for mut_name in mutation_name), f"Invalid mutation name as its not in the passed mutation dictionary: {list(mutation_dict.keys())}"
        mutations = (mutation_dict[mut_name](**mutation_kwargs) for mut_name in mutation_name)
        return MULTIMUTATION_DICT[multi_mutation](tuple(mutations))

    @classmethod
    def get_crossover(self, crossover_name: Union[str, Tuple[str]], multi_crossover: str, crossover_dict: Dict[str, Type[Crossover]],
                      crossover_kwargs: Dict[str, Any]):
        """Given either a crossover name or a list of crossovers, return the associated Crossover class"""
        if isinstance(crossover_name, str):
            assert crossover_name in crossover_dict, f"Invalid crossover name as its not in the passed crossover dictionary: {list(crossover_dict.keys())}"
            return crossover_dict[crossover_name](**crossover_kwargs)
        assert all(cross_name in crossover_dict for cross_name in crossover_name), f"Invalid crossover name as its not in the passed crossover dictionary: {list(crossover_dict.keys())}"
        crossovers = (crossover_dict[cross_name](**crossover_kwargs) for cross_name in crossover_name)
        return MULTI_CROSSOVER[multi_crossover](tuple(crossovers))

    def calculate_diversity(self, population: List[np.ndarray]):   
        population = np.array(population)
        centroid = np.mean(population, axis=0)
        return np.sqrt(np.sum((population - centroid)**2, axis=0)), centroid

    def plot_evolution_metrics(self, evolution_metrics: Dict, axs: Tuple[plt.Axes]=None,figsize=(12, 5)):
        assert axs is None or len(axs) != len(evolution_metrics), "Invalid number of axes"
        if axs is None:
            fig, axs = plt.subplots(len(evolution_metrics), 1, figsize=figsize)
            plt.subplots_adjust(hspace=1.)
        for idx, (metric_name, metric) in enumerate(evolution_metrics.items()):
            plot_evolution_metric(axs[idx], metric_name, metric)

    def generate_population(self, seed_population: List[np.ndarray]=None):
        if seed_population is None:
            return self.creation()
        if len(seed_population) < self.n_individuals:
            return seed_population + self.creation()[:self.n_individuals - len(seed_population)]
        return seed_population
    
    @abstractmethod
    def selection(self, fit_population: List):
        """
        Selects T random individuals from the population and obtains the one with the best values.
        """
        pass

    @abstractmethod
    def creation(self):
        """Creates an random population from the gene variation"""
        pass

    @abstractmethod
    def apply_phenotype(self, cromosome: Any):
        """Applies the phenotype to the passed cromosome"""
        pass

    @abstractmethod
    def evolve(self, fit: Callable, n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False, seed_population: List[np.ndarray]=None):
        pass

    @abstractmethod
    def calculate_metrics(self, *args):
        pass

@dataclass
class MultiGeneticEvolutive:
    evolutions: Union[Evolutive, Tuple[Evolutive, ...]]
    pop_to_reintroduce: float = 0.2
    n_generations_per_evo: Union[int, Tuple[int]] = -1

    def __post_init__(self):
        if isinstance(self.evolutions, Evolutive):
            self.evolutions = (self.evolutions,)

    def n_generations_per_evolution(self, n_generations: int):
        if isinstance(self.n_generations_per_evo, int):
            n_gens = (n_generations // len(self.evolutions) if self.n_generations_per_evo <= 0 else
                      min(self.n_generations_per_evo, n_generations))
            n_generations_evo = [n_gens for _ in range(0, n_generations, n_gens)]
            total_actual_gens = sum(n_generations_evo)
            if total_actual_gens < n_generations:
                n_generations_evo.append(n_generations - total_actual_gens)
            return n_generations_evo
        n_generations_evo = [self.n_generations_per_evo for _ in range(0, n_generations, sum(self.n_generations_per_evo))]
        n_generations_evo = reduce(lambda a, b: a + b, n_generations_evo, [])
        total_actual_gens = sum(n_generations_evo)
        n_evolutions = len(self.n_generations_per_evo)
        idx = 0
        while total_actual_gens < n_generations:
            n_evo = self.n_generations_per_evo[idx % n_evolutions]
            if total_actual_gens + n_evo > n_generations:
                n_evo = n_generations - total_actual_gens
            n_generations_evo.append(n_evo)
            total_actual_gens += n_evo
        return n_generations_evo
        
    def evolution_it(self, n_generations: int):
        def evolutions():
            n_evolutions = len(self.evolutions)
            for idx, n_generation in enumerate(self.n_generations_per_evolution(n_generations)):
                yield self.evolutions[idx % n_evolutions], n_generation
        return evolutions()

    def evolve(self, fit: Union[Callable, List[Callable]], n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False):
        total_gens = 0
        last_update = 0
        results = None
        evolution_metrics = []
        for idx, (evolution, n_generations_evo) in enumerate(self.evolution_it(n_generations)):
            if trace != 0:
                print(f"Evolutive {idx}:")
            if results is None:
                results = evolution.evolve(fit, n_generations_evo, target, trace, obtain_metrics)
                last_update = results["last_update"]
                total_gens += n_generations_evo
                if obtain_metrics:
                    evolution_metrics.append(results["evolution_metrics"])
                continue
            seed_population = results["best_cromosome"]
            n_reintroduce = int(self.pop_to_reintroduce * evolution.n_individuals - len(seed_population))
            if n_reintroduce > 0:
                seed_population.extend(sample(results["population"], n_reintroduce))
            results = evolution.evolve(fit, n_generations_evo, target, trace, obtain_metrics, seed_population)
            if results["last_update"] != 0:
                last_update = total_gens + results["last_update"]
            total_gens += n_generations_evo
            if obtain_metrics:
                evolution_metrics.append(results["evolution_metrics"])
        results["last_update"] = last_update
        if obtain_metrics:
            results["evolution_metrics"] = evolution_metrics
        return results
    
    def plot_evolution_metrics(self, evolution_metrics: Dict, axs: Tuple[plt.Axes]=None, figsize=(12, 5)):
        fig, axs = plt.subplots(len(self.evolutions), 2, figsize=figsize)
        plt.subplots_adjust(hspace=1.25)
        for idx, ax_line in enumerate(axs):
            metric = evolution_metrics[idx]
            for idx, (metric_name, metric) in enumerate(metric.items()):
                plot_evolution_metric(ax_line[idx], metric_name, metric)
