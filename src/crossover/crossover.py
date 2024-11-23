from abc import abstractmethod, ABC
from typing import Tuple
from dataclasses import dataclass
from random import choice
from numpy import ndarray

@dataclass
class Crossover(ABC):
    p_crossover: float
    gene_type: str

    @abstractmethod
    def crossover(self, ind1: ndarray, ind2: ndarray) -> Tuple[ndarray, ndarray]:
        pass

@dataclass
class MultiCrossover:
    crossovers: Tuple[Crossover]

    def crossover(self, ind1: ndarray, ind2: ndarray) -> Tuple[ndarray, ndarray]:
        return choice(self.crossovers).crossover(ind1, ind2)
