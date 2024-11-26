
from typing import Callable, Union, Tuple, Any

import numpy as np

from .evolutive import Evolutive
from ..mutation import ALPHABET_MUTATION, MultiMutation
from ..crossover import ALPHABET_CROSSOVER, MultiCrossover

def gene_type(alphabet_size:int):
    size = 8
    while alphabet_size > 1 << size and size != 64:
        size *= 2
    return f'uint{size}'

class VocabularyEvolutive(Evolutive):
    def __init__(self,
                 alphabet: Tuple,
                 cromolength: Union[int, Tuple[int]],
                 n_individuals: int,
                 mutation: Union[str, Tuple[str]] = "random-gene",
                 crossover: Union[str, Tuple[str]] = "split-merge",
                 p_crossover: float = 0.7,
                 average_mutation_rate: float = 1.,
                 phenotype: Callable[[Tuple], Any] = lambda cromosome: cromosome,
                 elitist_individuals: int = 0,
                 maximize: bool = True,
                 use_multithread: bool = False,
                 T_selection: int = 2
                 ):
        self.gene_type = gene_type(len(alphabet))
        self.alphabet = np.array(alphabet)
        self.cromolength = cromolength
        # Init Crossover
        crossover_dict = {
            "p_crossover": p_crossover,
            "gene_type": self.gene_type
        }
        crossover = self.get_crossover(crossover, ALPHABET_CROSSOVER, crossover_dict)
        # Init mutation
        mutation_dict = {
            "average_mutation_rate": average_mutation_rate,
            "gene_type": self.gene_type,
            "vocabulary_length": self.alphabet.shape[0]
        }
        mutation = self.get_mutation(mutation, ALPHABET_MUTATION, mutation_dict)
        super().__init__(
            n_individuals, mutation, crossover, phenotype, elitist_individuals, maximize, use_multithread, T_selection
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        return self.phenotype(self.alphabet[cromosome])

    def creation(self):
        return [np.randint(size=self.cromolength, dtype=self.gene_type) for _ in range(self.n_individuals)]
