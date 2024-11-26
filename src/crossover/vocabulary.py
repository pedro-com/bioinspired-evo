from typing import Tuple
from .crossover import Crossover
import numpy as np

class SplitMergeCrossover(Crossover):
    def crossover(self, ind1: np.ndarray, ind2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform split-merge crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by splitting and merging parents
            """
        # Make sure arrays are of same length
        if len(ind1) != len(ind2):
            raise ValueError("Parents must be same length")
            
        # Select random split point
        split_point = np.random.randint(1, len(ind1), 2)

        
        # Create offspring by swapping segments
        child1 = np.concatenate((ind1[:split_point[0]], ind2[split_point[0]:]))
        child2 = np.concatenate((ind2[:split_point[1]], ind1[split_point[1]:]))
        
        return child1, child2



