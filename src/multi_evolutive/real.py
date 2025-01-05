from typing import Callable, Union, Tuple, Any, Literal, List

import numpy as np

from .evolutive import MultiEvolutive, SelectionPool, Front, Penalization
from ..evaluation import MultiObjectiveEvaluation
from ..mutation import REAL_MUTATION, Mutation, MultiMutation
from ..crossover import REAL_CROSSOVER, Crossover, MultiCrossover

class RealMultiEvolutive(MultiEvolutive):
    def __init__(self,
                 n_individuals: int,
                 value_range: Tuple[float, float],
                 maximize: Tuple[bool, ...],
                 gene_type: str="float64",
                 cromolength: int = None,
                 mutation: Union[str, Tuple[str], Mutation, MultiMutation] = "random-gene",
                 crossover: Union[str, Tuple[str], Crossover, MultiCrossover] = "split-merge",
                 multi_mutation: str="random",
                 multi_crossover: str="random",
                 p_crossover: float = 0.7,
                 average_mutation_rate: float = 1.,
                 alpha: float=0.,
                 mutation_eps: float=0.1,
                 phenotype: Callable[[Tuple], Any] = lambda cromosome: cromosome,
                 elitism: bool = False,
                 normalize: bool = False,
                 front: Union[Front, List[Front], List[Tuple[Front, float]]] = 'range',
                 penalization: Union[Penalization, List[Penalization], List[Tuple[Penalization, float]]] = 'sharing',
                 niche_sharing_size: float = 0.8,
                 selection_pool: Union[SelectionPool, List[SelectionPool], List[Tuple[SelectionPool, float]]] = 'best',
                 selection_pool_size: float = 0.8,
                 steps_to_reduce_p_elite: int = 100,
                 T_selection: int = 2,
                 evaluation_metrics: MultiObjectiveEvaluation = None
                 ):
        self.value_range = value_range if value_range[0] < value_range[1] else (value_range[1], value_range[0])
        self.cromolength = cromolength
        self.normalize = normalize
        # Crossover
        if not isinstance(crossover, (Crossover, MultiCrossover)):
            crossover = self.get_crossover(
                crossover_name=crossover,
                multi_crossover=multi_crossover,
                crossover_dict=REAL_CROSSOVER,
                crossover_kwargs={
                    "p_crossover": p_crossover,
                    "gene_type": gene_type,
                    "alpha": alpha,
                    "value_range": self.value_range if not normalize else (0, 1)
                }
            )
        # Mutation
        if not isinstance(mutation, (Mutation, MultiMutation)):
            mutation = self.get_mutation(
                mutation_name=mutation,
                multi_mutation=multi_mutation,
                mutation_dict=REAL_MUTATION,
                mutation_kwargs={
                    "average_mutation_rate": average_mutation_rate,
                    "gene_type": gene_type,
                    "mutation_eps": mutation_eps,
                    "value_range": self.value_range if not normalize else (0, 1)
                }
            )
        super().__init__(
            n_individuals=n_individuals,
            mutation=mutation,
            crossover=crossover,
            gene_type=gene_type,
            phenotype=phenotype,
            elitism=elitism,
            maximize=maximize,
            front=front,
            penalization=penalization,
            niche_sharing_size=niche_sharing_size,
            selection_pool=selection_pool,
            selection_pool_size=selection_pool_size,
            steps_to_reduce_p_elite=steps_to_reduce_p_elite,
            T_selection=T_selection,
            evaluation_metrics=evaluation_metrics
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        if self.normalize:
            return self.phenotype((self.value_range[1] - self.value_range[0])*cromosome + self.value_range[0])
        return self.phenotype(cromosome)

    def creation(self):
        if self.normalize:
            return [np.random.random(size=self.cromolength).astype(self.gene_type) for _ in range(self.n_individuals)]
        return [np.random.uniform(self.value_range[0], self.value_range[1], size=self.cromolength).astype(self.gene_type) for _ in range(self.n_individuals)]
