from typing import Callable, Union, Tuple, Any

import numpy as np

from ..utils import alphabet_gene_type
from .evolutive import GeneticEvolutive
from ..mutation import ALPHABET_MUTATION, Mutation, MultiMutation
from ..crossover import ALPHABET_CROSSOVER, Crossover, MultiCrossover


class AlphabetEvolutive(GeneticEvolutive):
    def __init__(self,
                 alphabet: Tuple,
                 n_individuals: int,
                 mutation: Union[str, Tuple[str], Mutation, MultiMutation] = "random-gene",
                 crossover: Union[str, Tuple[str], Crossover, MultiCrossover] = "split-merge",
                 multi_mutation: str="random",
                 multi_crossover: str="random",
                 cromolength: int = None,
                 p_crossover: float = 0.7,
                 average_mutation_rate: float = 1.,
                 mutation_eps: float=0.1,
                 phenotype: Callable[[Tuple], Any] = lambda cromosome: cromosome,
                 elitism: bool = False,
                 maximize: bool = True,
                 n_threads: int = 8,
                 use_multithread: bool = False,
                 T_selection: int = 2
                 ):
        self.alphabet = np.array(alphabet)
        len_alpha = self.alphabet.shape[0]
        self.cromolength = self.alphabet.shape[0] if cromolength is None else cromolength
        gene_type_v = alphabet_gene_type(len_alpha)
        # Crossover
        if not isinstance(crossover, (Crossover, MultiCrossover)):
            crossover = self.get_crossover(
                crossover_name=crossover,
                multi_crossover=multi_crossover,
                crossover_dict=ALPHABET_CROSSOVER,
                crossover_kwargs={
                    "p_crossover": p_crossover,
                    "gene_type": gene_type_v
                }
            )
        # Mutation
        if not isinstance(mutation, (Mutation, MultiMutation)):
            mutation = self.get_mutation(
                mutation_name=mutation,
                multi_mutation=multi_mutation,
                mutation_dict=ALPHABET_MUTATION,
                mutation_kwargs={
                    "average_mutation_rate": average_mutation_rate,
                    "gene_type": gene_type_v,
                    "mutation_eps": mutation_eps,
                    "vocabulary_length": len_alpha
                }
            )
        super().__init__(
            n_individuals=n_individuals,
            mutation=mutation,
            crossover=crossover,
            gene_type=gene_type_v,
            phenotype=phenotype,
            elitism=elitism,
            maximize=maximize,
            n_threads=n_threads,
            use_multithread=use_multithread,
            T_selection=T_selection,
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        return self.phenotype(self.alphabet[cromosome])

    def creation(self):
        return [np.random.randint(self.alphabet.shape[0], size=self.cromolength, dtype=self.gene_type) for _ in range(self.n_individuals)]

class PermutationEvolutive(AlphabetEvolutive):
    def creation(self):
        def random_shuffle():
            permutation = np.arange(self.alphabet.shape[0], dtype=self.gene_type)
            np.random.shuffle(permutation)
            return permutation
        return [random_shuffle() for _ in range(self.n_individuals)]
