from abc import abstractmethod
from typing import Tuple
import numpy as np
from dataclasses import dataclass

class Mutation:
    vocabulary: Tuple
    p_mutation: float

    @abstractmethod
    def mutate(ind: Tuple):
        pass
