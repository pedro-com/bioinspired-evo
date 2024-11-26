import random
import numpy as np
import mutation

class SwapMutation(mutation.Mutation):
    average_mutation_rate = float
    gene_type = str

    def mutate(self, indv: np.ndarray) -> np.ndarray:
        if random.random() < self.average_mutation_rate:
            idx1 = np.random.choice(range(len(indv)))
            gene = indv[idx1]
            indv = np.delete(indv, idx1)
            idx2 = np.random.choice(range(len(indv)+1))
            indv = np.insert(indv, idx2, gene)
            return indv
        