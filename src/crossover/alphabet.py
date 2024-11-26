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
        
        if np.random.random() > self.p_crossover:
            return ind1, ind2
        
        # Select random split point
        split_point = np.random.randint(1, len(ind1), 2)
    
        # Create offspring by swapping segments
        child1 = np.concatenate((ind1[:split_point[0]], ind2[split_point[0]:]))
        child2 = np.concatenate((ind2[:split_point[1]], ind1[split_point[1]:]))
        
        return child1, child2


class UniformCrossover(Crossover):
    def crossover(self, ind1: np.ndarray, ind2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by uniform crossover
        """
        # Make sure arrays are of same length
        if len(ind1) != len(ind2):
            raise ValueError("Parents must be same length")
        
        if np.random.random() > self.p_crossover:
            return ind1, ind2
        
        # Create copies of parents
        child1 = np.copy(ind1)
        child2 = np.copy(ind2)
        
        # Generate random boolean mask
        mask = np.random.random(len(ind1), size=2) < 0.5
        
        # Swap elements according to mask
        child1[mask[0]] = ind2[mask[0]]
        child2[mask[1]] = ind1[mask[1]]
        
        return child1, child2

class HalfFixedCrossover(Crossover):
    def crossover(self, ind1: np.ndarray, ind2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform half-fixed crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by half-fixed crossover
        """
        # Make sure arrays are of same length
        if len(ind1) != len(ind2):
            raise ValueError("Parents must be same length")  
        
        if np.random.random() > self.p_crossover:
            return ind1, ind2
        
        # Create copies of parents
        child1 = np.copy(ind1)
        child2 = np.copy(ind2)
        
        # Generate random boolean mask
        mask = np.random.random(len(ind1)) < 0.5
        
        child1[np.logical_not(mask)] = ind2[np.isin(ind2, child1[mask], invert=True, assume_unique=True)]
        child2[np.logical_not(mask)] = ind1[np.isin(ind1, child2[mask], invert=True, assume_unique=True)]   

        return child1, child2
