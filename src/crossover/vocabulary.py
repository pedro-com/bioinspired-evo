from typing import Tuple
from .crossover import Crossover
import numpy as np

class SplitMergeCrossover(Crossover):
    def crossover(self, ind1: np.ndarray, ind2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return ind1, ind2
