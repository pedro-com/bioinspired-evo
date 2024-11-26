import random
import numpy as np
import mutation

class SwapMutation(mutation.Mutation):
    average_mutation_rate = float
    gene_type = str

    def mutate(self, indv: np.ndarray) -> np.ndarray:
        if random.random() < self.average_mutation_rate:
            idx1, idx2 = np.random.choice(range(len(indv)), size=2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            indv[idx1+1:idx2] = indv[idx1+1:idx2][::-1]
        