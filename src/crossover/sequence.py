from typing import Tuple
from .crossover import Crossover
import numpy as np

class SequenceCrossover(Crossover):
    def __init__(self, p_crossover: float, gene_type: str, **kwargs):
        super().__init__(p_crossover, gene_type)

class SplitMergeCrossover(SequenceCrossover):
    def crossover(self, indv1: np.ndarray, indv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform split-merge crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by splitting and merging parents
        """
        # Make sure arrays are of same length
        if len(indv1) != len(indv2):
            raise ValueError("Parents must be same length")  

        if np.random.random() > self.p_crossover:
            return np.copy(indv1), np.copy(indv2)
        # Select random split point
        split_point = np.random.randint(1, len(indv1), 2)
        # Create offspring by swapping segments
        child1 = np.concatenate((indv1[:split_point[0]], indv2[split_point[0]:]))
        child2 = np.concatenate((indv2[:split_point[1]], indv1[split_point[1]:]))
        return child1, child2


class UniformCrossover(SequenceCrossover):
    def crossover(self, indv1: np.ndarray, indv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by uniform crossover
        """
        # Make sure arrays are of same length
        if len(indv1) != len(indv2):
            raise ValueError("Parents must be same length")
        child1 = np.copy(indv1)
        child2 = np.copy(indv2)
    
        if np.random.random() > self.p_crossover:
            return child1, child2
        # Generate random boolean mask
        mask = np.random.random(len(indv1)) < 0.5
        # Swap elements according to mask
        child1[mask] = indv2[mask]
        child2[mask] = indv1[mask]
        return child1, child2

class HalfFixedCrossover(SequenceCrossover):
    def crossover(self, indv1: np.ndarray, indv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform half-fixed crossover between two parent individuals.
        
        Args:
            ind1: First parent individual (numpy array)
            ind2: Second parent individual (numpy array)
            
        Returns:
            Tuple of two offspring created by half-fixed crossover
        """
        # Make sure arrays are of same length
        if len(indv1) != len(indv2):
            raise ValueError("Parents must be same length")

        child1 = np.copy(indv1)
        child2 = np.copy(indv2)
        if np.random.random() > self.p_crossover:
            return child1, child2
        # Generate random boolean mask
        mask = np.random.random(len(indv1)) < 0.5
        child1[np.logical_not(mask)] = indv2[np.isin(indv2, child1[mask], invert=True, assume_unique=True)]
        child2[np.logical_not(mask)] = indv1[np.isin(indv1, child2[mask], invert=True, assume_unique=True)]   
        return child1, child2
