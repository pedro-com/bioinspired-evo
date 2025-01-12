from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Callable, Any, List, Type
from random import sample
from multiprocessing import Pool

from ..evolutive import Evolutive

@dataclass
class GeneticEvolutive(Evolutive):
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

    def evolve(self, fit: Callable, n_generations:int, target:float=None, trace:int=0, obtain_metrics: bool=False, seed_population: List[np.ndarray]=None):
        """
        Evolves the individuals over n_generations, and if n_populations is greater than 1, applies the process over
        n_generations.
        """
        population = self.generate_population(seed_population)
        if obtain_metrics:
            evolution_metrics = {
                metric: {name: np.zeros(n_generations, dtype=np.float64) for name in ("min", "mean", "max", "best")}
                for metric in ("fit", "diversity")
            }
        # Obtain fit for the population
        fit_population = self.fit_sort(fit, population)
        # Obtain best individual
        best_individual = fit_population[-1]
        last_update = (0, best_individual[1])
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
            if last_update[1] != best_individual[1]:
                last_update = (generation, best_individual[1])
            if target is not None and ((self.maximize and best_individual[1] >= target) or (not self.maximize and best_individual[1] <= target)):
                break
        if trace != 0:
            print(f"Generation {generation + 1:8d} | Best Fit: {best_individual[1]:4.4f} | Mean Fit: {evolution_metrics['fit']['mean'][generation]:4.4f}")
        results = {
            "best": self.apply_phenotype(best_individual[0]),
            "best_cromosome": [best_individual[0]],
            "population": population,
            "last_update": last_update[0]
        }
        if obtain_metrics:
            results["evolution_metrics"] = evolution_metrics
        return results

    def calculate_metrics(self, best_individual: Tuple, fit_population: List):
        diversity, centroid = self.calculate_diversity([pop[0] for pop in fit_population])
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
                "best": np.sqrt(np.sum((best_individual[0] - centroid)**2, axis=0))
            }
        }
