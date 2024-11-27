import random
import numpy as np
import mutation

class SwapMutation(mutation.Mutation):
    average_mutation_rate = float
    gene_type = str
    
    def mutate(self, indv: np.ndarray) -> np.ndarray:
        if random.random() < self.average_mutation_rate:
            indx1, indx2 = np.random.choice(range(len(indv)), size= 2)
            indv[indx1], indv[indx2] = indv[indx2], indv[indx1]
            
        return indv