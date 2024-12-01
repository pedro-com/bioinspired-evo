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
    def crossover(self, indv1: ndarray, indv2: ndarray) -> Tuple[ndarray, ndarray]:
        pass

@dataclass
class MultiCrossover(ABC):
    crossovers: Tuple[Crossover]

    def crossover(self, indv1: ndarray, indv2: ndarray) -> Tuple[ndarray, ndarray]:
        pass

class RandomCrossover(MultiCrossover):
    def crossover(self, indv1: ndarray, indv2: ndarray) -> Tuple[ndarray, ndarray]:
        return choice(self.crossovers).crossover(indv1, indv2)
