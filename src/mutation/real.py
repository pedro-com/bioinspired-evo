from dataclasses import dataclass
import numpy as np
import numpy.random as rng
from .mutation import Mutation
from typing import Tuple

@dataclass
class RealNumberMutation(Mutation):
    value_range: Tuple[int]

class RandomRangeMutation(RealNumberMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(size=indv.shape[0]) < self.p_mutation(indv)
        indv[mutation_mask] = rng.uniform(self.value_range[0], self.value_range[1], size=mutation_mask.sum())
        return indv

class RandomLocalMutation(RealNumberMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(size=indv.shape[0]) < self.p_mutation(indv)
        indv[mutation_mask] += self.mutation_eps*(2*rng.random() - 1.)
        return np.clip(indv, self.value_range[0], self.value_range[1])
