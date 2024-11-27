from dataclasses import dataclass
import numpy as np
import numpy.random as rng
from .mutation import Mutation

@dataclass
class AlphabetMutation(Mutation):
    vocabulary_length: int

class RandomGeneMutation(AlphabetMutation):
    def __post_init__(self):
        self.p_mutation = self.average_mutation_rate / self.vocabulary_length

    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation
        indv[mutation_mask] = rng.randint(0, self.vocabulary_length, size=mutation_mask.sum())
        return indv

class SwapMutation(AlphabetMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        swaps = rng.randint(low=self.average_mutation_rate - 1, high=self.average_mutation_rate + 2)
        random_swaps = rng.randint(low=0, high=self.vocabulary_length, size=(swaps, 2))
        for swap in random_swaps:
            indv[swap] = indv[np.flip(swap)]
        return indv