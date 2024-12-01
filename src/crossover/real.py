from typing import Tuple
from .crossover import Crossover
import numpy as np
import numpy.random as rng
from dataclasses import dataclass

@dataclass
class RealCrossover(Crossover):
    value_range: Tuple[int]
    alpha: float

    def __post_init__(self):
        assert 0 <= self.alpha < 1, "Invalid alpha value"

class BLXCrossover(RealCrossover):
    def crossover(self, indv1: np.ndarray, indv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(indv1) != len(indv2):
            raise ValueError("Parents must be same length")  
        if rng.random() > self.p_crossover:
            return np.copy(indv1), np.copy(indv2)
        beta = rng.random(size=indv1.shape[0])
        return (
            np.clip(beta*indv1 + (1 - beta)*indv2, self.value_range[0], self.value_range[1]),
            np.clip((1 - beta)*indv1 + beta*indv2, self.value_range[0], self.value_range[1])
        )

class BLXAlphaCrossover(RealCrossover):
    def crossover(self, indv1: np.ndarray, indv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(indv1) != len(indv2):
            raise ValueError("Parents must be same length")
        if rng.random() > self.p_crossover:
            return np.copy(indv1), np.copy(indv2)
        beta = (1 + self.alpha)*rng.random(size=indv1.shape[0])
        return (
            np.clip(beta*indv1 + (1 + self.alpha - beta)*indv2, self.value_range[0], self.value_range[1]),
            np.clip((1 + self.alpha - beta)*indv1 + beta*indv2, self.value_range[0], self.value_range[1]),
        )
