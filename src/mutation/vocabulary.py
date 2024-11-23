from dataclasses import dataclass
import numpy as np
from .mutation import Mutation

@dataclass
class VocabularyMutations(Mutation):
    vocabulary_length: int

class RandomGeneMutation(VocabularyMutations):
    def __post_init__(self):
        self.p_mutation = self.average_mutation_rate / self.vocabulary_length

    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = np.random.random(indv.shape[0]) < self.p_mutation
        indv[mutation_mask] = np.random.randint(0, self.vocabulary_length, size=mutation_mask.sum())
        return indv
