from typing import Callable, Union, Tuple, Any

import numpy as np

from .evolutive import Evolutive
from ..mutation import REAL_MUTATION, Mutation, MultiMutation
from ..crossover import REAL_CROSSOVER, Crossover, MultiCrossover

class RealEvolutive(Evolutive):
    def __init__(self,
                 n_individuals: int,
                 value_range: Tuple[float, float],
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
                 maximize: bool = True,
                 normalize: bool = False,
                 use_multithread: bool = False,
                 T_selection: int = 2
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
            use_multithread=use_multithread,
            T_selection=T_selection,
        )
    
    def apply_phenotype(self, cromosome: np.ndarray):
        if self.normalize:
            return self.phenotype((self.value_range[1] - self.value_range[0])*cromosome + self.value_range[0])
        return self.phenotype(cromosome)

    def creation(self):
        if self.normalize:
            return [np.random.random(size=self.cromolength).astype(self.gene_type) for _ in range(self.n_individuals)]
        return [np.random.uniform(self.value_range[0], self.value_range[1], size=self.cromolength, dtype=self.gene_type) for _ in range(self.n_individuals)]
