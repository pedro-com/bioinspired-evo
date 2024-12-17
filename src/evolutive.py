from abc import abstractmethod, ABC
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Any, Union, List, Dict, Type
from random import sample
import numpy as np

from .crossover import Crossover, MULTI_CROSSOVER, MultiCrossover
from .mutation import Mutation, MULTIMUTATION_DICT, MultiMutation

def plot_metric(ax:plt.Axes, metric_name:str, metric:Dict):
    styles = {"min": "r", "mean": "b", "max": "g", "best": "darkviolet"}
    fontdict={"fontweight": "bold", "fontfamily": "DejaVu Sans"}
    n_generations = 0
    for k_name, color in styles.items():
        n_generations = max(len(metric[k_name]), n_generations)
        ax.plot(metric[k_name], color=color, label=k_name.capitalize())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=True)
    ax.set_title(f"{metric_name.capitalize()} across Generations", fontsize=14, pad=20, fontdict=fontdict)
    ax.set_xlabel("Generations", fontsize=12, fontdict=fontdict)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12, fontdict=fontdict)
    ax.set_xlim(0, n_generations - 1)
    ax.grid(True)

@dataclass
class Evolutive(ABC):
    n_individuals: int
    mutation: Union[Mutation, MultiMutation]
    crossover: Union[Crossover, MultiCrossover]
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

    def select(self, fit_population: List):
        T_sample = sample(fit_population, k=self.T_selection)
        best_individual = T_sample[0]
        for individual in T_sample[1:]:
            best_individual = self.best_individual(best_individual, individual)
        return best_individual[0]

    def selection(self, fit_population: List):
        """
        Selects T random individuals from the population and obtains the one with the best values.
        """
        select = lambda : self.select(fit_population)
        return ((select(), select()) for _ in range(self.n_individuals // 2))
    
    @abstractmethod
    def evolve(self, fit: Union[Callable, Tuple[Callable]], n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False, seed_population: List[np.ndarray]=None):
        pass

    @abstractmethod
    def calculate_metrics(self, *args):
        pass

    @abstractmethod
    def calculate_diversity(self, *args):
        pass

    def plot_metrics(self, evolution_metrics: Dict, axs: Tuple[plt.Axes]=None,figsize=(12, 5)):
        assert axs is None or len(axs) != len(evolution_metrics), "Invalid number of axes"
        if axs is None:
            fig, axs = plt.subplots(len(evolution_metrics), 1, figsize=figsize)
            plt.subplots_adjust(hspace=1.)
        for idx, (metric_name, metric) in enumerate(evolution_metrics.items()):
            plot_metric(axs[idx], metric_name, metric)
