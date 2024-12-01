from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from typing import Tuple
from random import choice

@dataclass
class Mutation(ABC):
    """
    Genetic Mutation to be applied over the Cromosome of an Individual
    * `average_mutation_rate`: Average number of mutations that are applied on the cromosome.
        `p_mutation = av_mutation_rate * (1 / L_cromosome)`.
    * `mutation_eps`: If the mutation causes changes in the locality of a gene, this epsilon determines
        the size of this locality.
    * `gene_type`: Uses numpy arrays to optimize operations (this optimizes memory size).
    """
    average_mutation_rate: float
    mutation_eps: float
    gene_type: str

    @abstractmethod
    def mutate(self, indv: ndarray) -> ndarray:
        pass

@dataclass
class MultiMutation(ABC):
    mutations: Tuple[Mutation]

    @abstractmethod
    def mutate(self, indv: ndarray) -> ndarray:
        pass

class RandomMutation(MultiMutation):
    def mutate(self, indv: ndarray) -> ndarray:
        return choice(self.mutations).mutate(indv)
