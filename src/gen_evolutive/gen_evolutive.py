from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Callable, Any, Union, List, Dict, Type
from random import sample
from multiprocessing import Pool
from functools import reduce
import matplotlib.pyplot as plt

from ..evolutive import Evolutive, plot_metric

@dataclass
class GeneticEvolutive(Evolutive):
    phenotype: Callable = lambda cromosome: cromosome
    elitism: bool = False
    maximize: bool = True
    use_multithread: bool = False

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

    def evolve(self, fit: Callable, n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False, seed_population: List[np.ndarray]=None):
        """
        Evolves the individuals over n_generations, and if n_populations is greater than 1, applies the process over
        n_generations.
        """
        if seed_population is None:
            population = self.creation()
        elif len(seed_population) < self.n_individuals:
            population = seed_population + self.creation()[:self.n_individuals - len(seed_population)]
        else:
            population = sample(seed_population, self.n_individuals)
        if obtain_metrics:
            evolution_metrics = {
                metric: {name: np.zeros(n_generations, dtype=np.float64) for name in ("min", "mean", "max", "best")}
                for metric in ("fit", "diversity")
            }
        # Obtain fit for the population
        fit_population = self.fit_sort(fit, population)
        # Obtain best individual
        best_individual = fit_population[-1]
        last_change = (0, best_individual[1])
        for generation in range(n_generations):
            # Metrics
            if obtain_metrics:
                for metric, values in self.calculate_metrics(best_individual, fit_population).items():
                    for m_name, value in values.items():
                        evolution_metrics[metric][m_name][generation] = value
            # Trace
            if trace != 0 and generation % trace == 0:
                print(f"Generation {generation:8d} | Best Fit: {best_individual[1]:4.4f} | Mean Fit: {evolution_metrics['fit']['mean'][generation]:4.4f}")
            # Selection + Crossover + Mutation
            population = []
            for par1, par2 in self.selection(fit_population):
                child1, child2 = self.crossover.crossover(par1, par2)
                population.append(self.mutation.mutate(child1))
                population.append(self.mutation.mutate(child2))
            # For uneven populations
            if self.n_individuals % 2 == 1:
                population.append(self.mutation.mutate(self.select(fit_population)))
            # Add elitist individuals
            if self.elitism:
                population[-1] = np.copy(best_individual[0])
            # Obtain fit for the population
            fit_population = self.fit_sort(fit, population)
            # Obtain best individual
            best_individual = self.best_individual(best_individual, fit_population[-1])
            if last_change[1] != best_individual[1]:
                last_change = (generation, best_individual[1])
            if target is not None and ((self.maximize and best_individual[1] >= target) or (not self.maximize and best_individual[1] <= target)):
                break
        if trace != 0:
            print(f"Generation {generation + 1:8d} | Best Fit: {best_individual[1]:4.4f} | Mean Fit: {evolution_metrics['fit']['mean'][generation]:4.4f}")
        results = {
            "best": self.apply_phenotype(best_individual[0]),
            "best_cromosome": best_individual[0],
            "population": population,
            "last_generation": [self.apply_phenotype(crom) for crom in population],
            "last_evolution_change": last_change[0]
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

@dataclass
class MultiGeneticEvolutive:
    evolutions: Union[GeneticEvolutive, Tuple[GeneticEvolutive, ...]]
    pop_to_reintroduce: float = 0.2
    n_generations_per_evo: Union[int, Tuple[int]] = -1

    def __post_init__(self):
        if isinstance(self.evolutions, GeneticEvolutive):
            self.evolutions = (self.evolutions,)
        assert self.n_generations_per_evo > 0, "Invalid number of generations, must be greater than 1"

    def evolution_it(self, n_generations: int):
        if self.n_generations_per_evo == -1:
            n_generations_evo = (n_generations // len(self.evolutions),)*len(self.evolutions)
        elif isinstance(self.n_generations_per_evo, int):
            n_gens = min(self.n_generations_per_evo, n_generations)
            n_generations_evo = [self.n_generations_per_evo for _ in range(0, n_generations, n_gens)]
        else:
            n_generations_evo = [self.n_generations_per_evo for _ in range(0, n_generations, sum(self.n_generations_per_evo))]
            n_generations_evo = reduce(lambda a, b: a + b, n_generations_evo, [])
            s_n_generations = sum(n_generations)
            while s_n_generations != n_generations:
                s_n_generations -= n_generations[-1]
                n_generations = n_generations[:-1]
                if s_n_generations < n_generations:
                    n_generations.append(n_generations - s_n_generations)
        def evolutions():
            yield
            return
        return

    def evolve(self, fit: Callable, n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False):
        gen_per_evolution = n_generations // len(self.evolutions)
        if trace != 0:
            print("Evolutive 1:")
        results = self.evolutions[0].evolve(fit, gen_per_evolution, target, trace, obtain_metrics)
        evolution_metrics = []
        last_change = results["last_evolution_change"]
        for idx, evo in enumerate(self.evolutions[1:], 2):
            seed_population = ([] if self.pop_to_reintroduce == 0. else 
                               sample(results["population"], int(self.pop_to_reintroduce*len(results["population"]))))
            seed_population.append(results["best_cromosome"])
            if obtain_metrics:
                evolution_metrics.append(results["evolution_metrics"])
            if trace != 0:
                print(f"Evolutive {idx}:")
            results = evo.evolve(fit, gen_per_evolution, target, trace, obtain_metrics, seed_population)
        if obtain_metrics:
            evolution_metrics.append(results["evolution_metrics"])
            results["evolution_metrics"] = evolution_metrics
        return results
    
    def plot_metrics(self, evolution_metrics: Dict, axs: Tuple[plt.Axes]=None, figsize=(12, 5)):
        fig, axs = plt.subplots(len(self.evolutions), 2, figsize=figsize)
        plt.subplots_adjust(hspace=1.)
        for idx, ax_line in enumerate(axs):
            metric = evolution_metrics[idx]
            for idx, (metric_name, metric) in enumerate(metric.items()):
                plot_metric(ax_line[idx], metric_name, metric)
