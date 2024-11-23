from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from typing import Tuple
from random import choice

@dataclass
class Mutation(ABC):
    average_mutation_rate: float
    gene_type: str

    @abstractmethod
    def mutate(self, indv: ndarray) -> ndarray:
        pass

@dataclass
class MultiMutation:
    mutations: Tuple[Mutation]

    def mutate(self, indv: ndarray) -> ndarray:
        return choice(self.mutations).mutate(indv)
