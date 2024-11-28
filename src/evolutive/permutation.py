
from typing import Callable, Union, Tuple, Any

import numpy as np

from .alphabet import Evolutive, gene_type
from ..mutation import PERMUTATION_MUTATION
from ..crossover import PERMUTATION_CROSSOVER

class PermutationEvolutive(Evolutive):
    def __init__(self,
                 alphabet: Tuple,
                 n_individuals: int,
                 mutation: Union[str, Tuple[str]] = "swap-gene",
                 crossover: Union[str, Tuple[str]] = "half-fixed",
                 p_crossover: float = 0.7,
                 average_mutation_rate: float = 1.,
                 phenotype: Callable[[Tuple], Any] = lambda cromosome: cromosome,
                 elitism: bool = False,
                 maximize: bool = True,
                 use_multithread: bool = False,
                 T_selection: int = 2
                 ):
        self.gene_type = gene_type(len(alphabet))
        self.alphabet = np.array(alphabet)
        # Init Crossover
        crossover_dict = {
            "p_crossover": p_crossover,
            "gene_type": self.gene_type
        }
        crossover = self.get_crossover(crossover, PERMUTATION_CROSSOVER, crossover_dict)
        # Init mutation
        mutation_dict = {
            "average_mutation_rate": average_mutation_rate,
            "gene_type": self.gene_type,
            "vocabulary_length": self.alphabet.shape[0]
        }
        mutation = self.get_mutation(mutation, PERMUTATION_MUTATION, mutation_dict)
        super().__init__(
            n_individuals=n_individuals,
            mutation=mutation,
            crossover=crossover,
            phenotype=phenotype,
            elitism=elitism,
            maximize=maximize,
            use_multithread=use_multithread,
            T_selection=T_selection
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        return self.phenotype(self.alphabet[cromosome])

    def creation(self):
        def random_shuffle():
            permutation = np.arange(self.alphabet.shape[0], dtype=self.gene_type)
            np.random.shuffle(permutation)
            return permutation
        return [random_shuffle() for _ in range(self.n_individuals)]

