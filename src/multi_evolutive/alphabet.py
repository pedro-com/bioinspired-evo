from typing import Callable, Union, Tuple, Any, Literal, List

import numpy as np

from ..utils import alphabet_gene_type
from .evolutive import MultiEvolutive, SelectionPool, Front, Penalization
from ..evaluation import MultiObjectiveEvaluation
from ..mutation import ALPHABET_MUTATION, Mutation, MultiMutation
from ..crossover import ALPHABET_CROSSOVER, Crossover, MultiCrossover


class AlphabetMultiEvolutive(MultiEvolutive):
    def __init__(self,
                 alphabet: Tuple,
                 n_individuals: int,
                 maximize: Tuple[bool, ...],
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
                 front: Union[Front, List[Front], List[Tuple[Front, float]]] = 'range',
                 penalization: Union[Penalization, List[Penalization], List[Tuple[Penalization, float]]] = 'sharing',
                 niche_sharing_size: float = 0.8,
                 selection_pool: Union[SelectionPool, List[SelectionPool], List[Tuple[SelectionPool, float]]] = 'best',
                 selection_pool_size: float = 0.8,
                 steps_to_reduce_p_elite: int = 100,
                 T_selection: int = 2,
                 evaluation_metrics: MultiObjectiveEvaluation = None
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
            front=front,
            penalization=penalization,
            niche_sharing_size=niche_sharing_size,
            selection_pool_size=selection_pool_size,
            selection_pool=selection_pool,
            steps_to_reduce_p_elite=steps_to_reduce_p_elite,
            T_selection=T_selection,
            evaluation_metrics=evaluation_metrics
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        return self.phenotype(self.alphabet[cromosome])

    def creation(self):
        return [np.random.randint(self.alphabet.shape[0], size=self.cromolength, dtype=self.gene_type) for _ in range(self.n_individuals)]

class PermutationMultiEvolutive(AlphabetMultiEvolutive):
    def creation(self):
        def random_shuffle():
            permutation = np.arange(self.alphabet.shape[0], dtype=self.gene_type)
            np.random.shuffle(permutation)
            return permutation
        return [random_shuffle() for _ in range(self.n_individuals)]
