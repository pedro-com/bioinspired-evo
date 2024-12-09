from dataclasses import dataclass
import numpy as np
import numpy.random as rng
from .mutation import Mutation

class SequenceMutation(Mutation):
    def __init__(self, average_mutation_rate: float, mutation_eps: float, gene_type: str, **kwargs):
        super().__init__(average_mutation_rate, mutation_eps, gene_type)

class SwapMutation(SequenceMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation(indv)
        for _ in range(mutation_mask.sum()):
            swap = rng.choice(indv.shape[0], size=2, replace=False)
            indv[swap] = indv[np.flip(swap)]
        return indv

class InsertMutation(SequenceMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation(indv)
        for _ in range(mutation_mask.sum()):
            idx1, idx2 = rng.choice(indv.shape[0], size=2, replace=False)
            gene = indv[idx1]
            if idx1 < idx2:
                indv[idx1:idx2] = indv[idx1 + 1:idx2 + 1]
            else:
                indv[idx2 + 1:idx1 + 1] = indv[idx2:idx1]
            indv[idx2] = gene
        return indv

class ToOptMutation(SequenceMutation):
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        mutation_mask = rng.random(indv.shape[0]) < self.p_mutation(indv)
        for _ in range(mutation_mask.sum()):
            idx1, idx2 = rng.choice(indv.shape[0], size=2, replace=False)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            indv[idx1 + 1:idx2] = np.flip(indv[idx1 + 1:idx2])
        return indv
