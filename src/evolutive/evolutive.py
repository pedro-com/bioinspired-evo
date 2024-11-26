from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Callable, Any, Union, List, Dict, Type
from dataclasses import dataclass
from multiprocessing import Pool
from random import sample

from ..crossover import Crossover, MultiCrossover
from ..mutation import Mutation, MultiMutation

def gene_type(alphabet_size:int):
    size = 8
    while alphabet_size > 1 << size and size != 64:
        size *= 2
    return f'uint{size}'

@dataclass
class Evolutive(ABC):
    n_individuals: int
    mutation: Union[Mutation, MultiMutation]
    crossover: Union[Crossover, MultiCrossover]
    phenotype: Callable = lambda cromosome: cromosome
    elitist_individuals: int = 0
    maximize: bool = True
    # Selection
    # p_normalize: float = 0.
    T_selection: int = 2
    # Multithread
    use_multithread: bool = False

    @classmethod
    def get_mutation(cls, mutation_name: Union[str, Tuple[str]], mutation_dict: Dict[str, Type[Mutation]],
                     mutation_kwargs: Dict[str, Any]):
        """Given either a mutation name or a list of mutations, return the associated Mutation class"""
        if isinstance(mutation_name, str):
            assert (mutation_name in mutation_dict,
                    f"Invalid mutation name as its not in the passed mutation dictionary: {list(mutation_dict.keys())}")
            return mutation_dict[mutation_kwargs](**mutation_kwargs)
        assert (all(mut_name in mutation_dict for mut_name in mutation_name),
                f"Invalid mutation name as its not in the passed mutation dictionary: {list(mutation_dict.keys())}")
        mutations = (mutation_dict[mut_name](**mutation_kwargs) for mut_name in mutation_name)
        return MultiMutation(*mutations)

    @classmethod
    def get_crossover(self, crossover_name: Union[str, Tuple[str]], crossover_dict: Dict[str, Type[Crossover]],
                      crossover_kwargs: Dict[str, Any]):
        """Given either a crossover name or a list of crossovers, return the associated Crossover class"""
        if isinstance(crossover_name, str):
            assert (crossover_name in crossover_dict,
                    f"Invalid crossover name as its not in the passed crossover dictionary: {list(crossover_dict.keys())}")
            return crossover_dict[crossover_name](**crossover_kwargs)
        assert (all(cross_name in crossover_dict for cross_name in crossover_name),
                f"Invalid crossover name as its not in the passed crossover dictionary: {list(crossover_dict.keys())}")
        crossovers = (crossover_dict[cross_name](**crossover_kwargs) for cross_name in crossover_name)
        return MultiCrossover(*crossovers)

    def fit_sort(self, fit: Callable, population: Tuple):
        """
        Applies the fit function to the phenotypes of the cromosomes and sorts by leaving the values to optimize
        at the end of the list.
        """
        fit = lambda cromosome : self.fit(self.apply_phenotype(cromosome))
        if not self.use_multithread:
            fit_values = map(lambda cromosome: (cromosome, fit(cromosome)), population)
        else:
            with Pool(self.n_individuals) as p:
                fit_values = p.map(lambda cromosome: (cromosome, fit(cromosome)), population)
        return sorted(fit_values, key=lambda v: v[0], reverse=not self.maximize)

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
        """
        Selects T random individuals from the population and obtains the one with the best values.
        """
        T_sample = sample(fit_population, k=self.T_selection)
        best_cromosome = T_sample[0]
        for cromosome in T_sample[1:]:
            best_cromosome = self.best_individual(best_cromosome, cromosome)
        return best_cromosome[0]

    def evolve(self, fit: Callable, n_generations:int, n_populations:int=1):
        """
        Evolves the individuals over n_generations, and if n_populations is greater than 1, applis the process over
        n_generations.
        """
        if n_populations > 1:
            with Pool(n_populations) as p:
                phenotypes = p.map(lambda : self.evolve(n_generations), [n_generations]*n_populations)
            return sorted(phenotypes, lambda v: fit(v), reverse=not self.maximize)[-1]
        population = self.creation()
        best_individual = None
        for generation in range(n_generations): # TODO Add logging
            # Obtain fit for the population
            fit_population = self.fit_sort(fit, population)
            # Obtain best individual
            best_individual = self.best_individual(best_individual, fit_population[-1])
            # Selection + Crossover + Mutation
            population = []
            for _ in range(0, self.n_individuals, 2):
                indv1, indv2 = self.crossover.crossover(self.select(fit_population), self.select(fit_population))
                population.append(self.mutation.mutate(indv1))
                population.append(self.mutation.mutate(indv2))
            if self.n_individuals % 2 != 1: # For uneven populations
                population.append(self.mutation.mutate(self.select(fit_population)))
            # Add elitist individuals
            for idx in range(self.elitist_individuals):
                population[-idx] = fit_population[-idx][0]
        return self.apply_phenotype(best_individual[0])

    @abstractmethod
    def apply_phenotype(self, cromosome: Any):
        """Applies the phenotype to the passed cromosome"""
        pass

    @abstractmethod
    def creation(self):
        """Creates an random population from the gene variation"""
        pass

