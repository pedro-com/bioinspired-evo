from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Callable, Any, Union, List, Dict, Type
from dataclasses import dataclass
from multiprocessing import Pool
from random import sample
import matplotlib.pyplot as plt

from ..crossover import Crossover, MULTI_CROSSOVER, MultiCrossover
from ..mutation import Mutation, MULTIMUTATION_DICT, MultiMutation
# from scipy.spatial.distance import pdist

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
    # Cromosome
    gene_type: str
    # Phenotype + Selection Variables
    phenotype: Callable = lambda cromosome: cromosome
    elitism: bool = False
    # p_normalize: float = 0.
    T_selection: int = 2
    maximize: bool = True
    # Multithread
    use_multithread: bool = False

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

    def fit_sort(self, fit: Callable, population: Tuple):
        """
        Applies the fit function to the phenotypes of the cromosomes and sorts by leaving the values to optimize
        at the end of the list.
        """
        def apply_fit(cromosome):
            return (cromosome, fit(self.apply_phenotype(cromosome)))
        if not self.use_multithread:
            fit_values = map(apply_fit, population)
        else:
            with Pool(self.n_individuals) as p:
                fit_values = p.map(apply_fit, population)
        return sorted(fit_values, key=lambda v: v[1], reverse=not self.maximize)

    def best_individual(self, indv1:Tuple, indv2:Tuple):
        """
        Obtains the best individual from a tuple of (indv, fit(indv)).
        """
        if indv1 is None or indv2 is None:
            return indv1 or indv2
        if self.maximize:
            return max(indv1, indv2, key=lambda v: v[1])
        return min(indv1, indv2, key=lambda v: v[1])

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

    def evolve(self, fit: Callable, n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False):
        """
        Evolves the individuals over n_generations, and if n_populations is greater than 1, applies the process over
        n_generations.
        """
        best_individual = None
        population = self.creation()
        if obtain_metrics:
            evolution_metrics = {
                metric: {name: np.zeros(n_generations, dtype=np.float64) for name in ("min", "mean", "max", "best")}
                for metric in ("fit", "diversity")
            }
        for generation in range(n_generations):
            # Obtain fit for the population
            fit_population = self.fit_sort(fit, population)
            # Obtain best individual
            best_individual = self.best_individual(best_individual, fit_population[-1])
            # Metrics
            if obtain_metrics:
                for metric, values in self.calculate_metrics(best_individual, fit_population).items():
                    for m_name, value in values.items():
                        evolution_metrics[metric][m_name][generation] = value
            # Trace
            if trace != 0 and generation % trace == 0:
                print(f"Generation {generation:8d} | Best Fit: {best_individual[1]:4.4f} | Mean Fit: {evolution_metrics['fit']['mean'][generation]:4.4f}")
            # Selection + Crossover + Mutation
            new_population = []
            for par1, par2 in self.selection(fit_population):
                child1, child2 = self.crossover.crossover(par1, par2)
                new_population.append(self.mutation.mutate(child1))
                new_population.append(self.mutation.mutate(child2))
            # For uneven populations
            if self.n_individuals % 2 == 1:
                new_population.append(self.mutation.mutate(self.select(fit_population)))
            # Add elitist individuals
            if self.elitism:
                new_population[-1] = np.copy(best_individual[0])
            population = new_population
            if target is not None and ((self.maximize and best_individual[1] >= target) or (not self.maximize and best_individual[1] <= target)):
                break
        if trace != 0:
            print(f"Generation {generation + 1:8d} | Best Fit: {best_individual[1]:4.4f} | Mean Fit: {evolution_metrics['fit']['mean'][generation]:4.4f}")
        results = {
            "best": self.apply_phenotype(best_individual[0]),
            "population": population,
            "last_generation": [self.apply_phenotype(crom) for crom in population],
        }
        if obtain_metrics:
            results["evolution_metrics"] = evolution_metrics
        return results

    def calculate_metrics(self, best_individual: Tuple, fit_population: List):
        diversity, best_diversity = self.calculate_diversity(best_individual[0], [pop[0] for pop in fit_population])
        return {
            "fit": {
                "min": fit_population[0][1],
                "max": fit_population[-1][1],
                "mean": np.mean([pop[1] for pop in fit_population]),
                "best": best_individual[1]
            },
            "diversity": {
                "min": diversity.min(),
                "max": diversity.max(),
                "mean": np.mean(diversity),
                "best": best_diversity
            }
        }

    def plot_metrics(self, evolution_metrics: Dict, axs: Tuple[plt.Axes]=None,figsize=(12, 5)):
        assert axs is None or len(axs) != len(evolution_metrics), "Invalid number of axes"
        if axs is None:
            fig, axs = plt.subplots(len(evolution_metrics), 1, figsize=figsize)
            plt.subplots_adjust(hspace=1.)
        for idx, (metric_name, metric) in enumerate(evolution_metrics.items()):
            plot_metric(axs[idx], metric_name, metric)

    def calculate_diversity(self, best_individual:np.ndarray, population: List):
        max_size = np.max((best_individual.shape[0], *(p.shape[0] for p in population)))
        if all(p.shape[0] == max_size for p in population) and best_individual.shape[0] == max_size:
            population = np.array(population)
            centroid = np.mean(population, axis=0)
            return np.sqrt(np.sum((population - centroid)**2, axis=0)), np.sqrt(np.sum((best_individual - centroid)**2, axis=0))
        population_v = np.zeros((max_size, len(population)), dtype=self.gene_type)
        for idx, p in enumerate(population):
            population_v[idx][:p.shape[0]] = p
        centroid = np.mean(population_v, axis=0)
        best_individual_v = np.zeros(max_size, dtype=self.gene_type)
        best_individual_v[:best_individual.shape[0]] = best_individual
        return np.sqrt(np.sum((population_v - centroid)**2, axis=0)), np.sqrt(np.sum((best_individual_v - centroid)**2, axis=0))

    @abstractmethod
    def apply_phenotype(self, cromosome: Any):
        """Applies the phenotype to the passed cromosome"""
        pass

    @abstractmethod
    def creation(self):
        """Creates an random population from the gene variation"""
        pass
