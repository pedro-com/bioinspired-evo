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

class InsertMutation(AlphabetMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        inserts = rng.randint(low=self.average_mutation_rate - 1, high=self.average_mutation_rate + 2)
        for _ in range(inserts):
            idx1 = rng.randint(indv.shape[0])
            gene = indv[idx1]
            indv = np.delete(indv, idx1)
            idx2 = rng.randint(indv.shape[0])
            indv = np.insert(indv, idx2, gene)
        return indv

class ToOptMutation(AlphabetMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        reverses = rng.randint(low=self.average_mutation_rate - 1, high=self.average_mutation_rate + 2)
        for _ in range(reverses):
            idx1, idx2 = rng.choice(range(indv.shape[0]), size=2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            indv[idx1+1:idx2] = indv[idx1+1:idx2][::-1]
        return indv
