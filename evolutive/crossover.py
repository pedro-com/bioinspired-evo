from abc import abstractmethod
from typing import Tuple
import numpy as np
from dataclasses import dataclass

class Crossover:
    p_crossover: float

    @abstractmethod
    def crossover(ind1: Tuple, ind2: Tuple):
        pass
