
from typing import Callable, Union, Tuple, Any

import numpy as np

from .evolutive import Evolutive
from ..mutation import VOCABULARY_MUTATION, MultiMutation
from ..crossover import VOCABULARY_CROSSOVER, MultiCrossover

class VocabularyEvolutive(Evolutive):
    def __init__(self,
                 vocabulary: Tuple,
                 n_individuals: int,
                 fit: Callable,
                 mutation: Union[str, Tuple[str]] = "random-gene",
                 crossover: Union[str, Tuple[str]] = "split-merge",
                 p_crossover: float = 0.7,
                 average_mutation_rate: float = 1.,
                 fenotype: Callable[[Tuple], Any] = lambda cromosome: cromosome,
                 elitist_individuals: int = 0,
                 maximize: bool = True,
                 use_multithread: bool = False,
                 T_selection: int = 2
                 ):
        self.gene_type = self._gene_type()
        self.vocabulary = np.array(vocabulary)
        # Init Crossover
        crossover_dict = {
            "p_crossover": p_crossover,
            "gene_type": self.gene_type
        }
        crossover = self.get_crossover(crossover, VOCABULARY_CROSSOVER, crossover_dict)
        # Init mutation
        mutation_dict = {
            "average_mutation_rate": average_mutation_rate,
            "gene_type": self.gene_type,
            "vocabulary_length": self.vocabulary.shape[0]
        }
        mutation = self.get_mutation(mutation, VOCABULARY_MUTATION, mutation_dict)
        super().__init__(
            n_individuals, fit, mutation, crossover, fenotype, elitist_individuals, maximize, use_multithread, T_selection
        )
    
    def _gene_type(self):
        vocab_len = len(self.vocabulary)
        size = 8
        while vocab_len > 1 << size and size != 64:
            size *= 2
        return f'uint{size}'
    
    def apply_fenotype(self, cromosome: np.ndarray):
        return self.fenotype(self.vocabulary[cromosome])

    def creation(self):
        pass
