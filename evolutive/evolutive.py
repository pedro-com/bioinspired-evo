import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass

from .crossover import Crossover
from .mutation import Mutation

@dataclass
class SimpleEvolutive:
    init_pop: int
    vocabulary: Tuple
    fit_function: Callable
    mutation: Mutation
    crossover: Crossover
    elitist_individuals: int = 0
    T_match: int = 2
    average_mutation_rate: float = 1.
    p_crossover: float = 0.7
    maximize: bool = True

    def __post_init__(self):
        self.mutation.p_mutation = self.average_mutation_rate / self.init_pop
        self.crossover.p_crossover = self.p_crossover
        self.mutation.vocabulary = (*self.vocabulary,) # TODO change this to use a representation vocab

    def fit_sort(self, population: Tuple):
        # Obtain fit and population sorted
        a = 0

    def create(self):
        # Random per gene
        a = 0
    
    def select(self):
        # Select por torneo
        a = 0
    
    def evolve(self, ngen:int=100):
        a = 0
