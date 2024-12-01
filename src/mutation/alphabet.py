from dataclasses import dataclass
import numpy as np
import numpy.random as rng
from .mutation import Mutation

@dataclass
class AlphabetMutation(Mutation):
    """
    Alphabet mutations are to be applied only on Alphabet type Evolutions. These mutations work on integer representations
    of the vocabulary.
    * `vocabulary_length`: Number of elements in the vocabulary.
    """
    vocabulary_length: int

    def __post_init__(self):
        assert 0. <= self.mutation_eps < 1, "Invalid mutation eps, it must be between 0 and 1"
        self.p_mutation = self.average_mutation_rate / self.vocabulary_length
        self.eps = np.ceil(self.mutation_eps*self.vocabulary_length)

class RandomGeneMutation(AlphabetMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation
        indv[mutation_mask] = rng.randint(0, self.vocabulary_length, size=mutation_mask.sum())
        return indv

class RandomLocalGeneMutation(AlphabetMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation
        indv[mutation_mask] = rng.randint(-self.eps, self.eps + 1, size=mutation_mask.sum())
        indv[indv >= self.vocabulary_length] %= self.vocabulary_length
        indv[indv < 0] += self.vocabulary_length
        return indv
    