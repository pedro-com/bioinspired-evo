from typing import Dict, Type
from .crossover import Crossover, MultiCrossover
from .alphabet import SplitMergeCrossover, UniformCrossover, HalfFixedCrossover

ALPHABET_CROSSOVER: Dict[str, Type[Crossover]]= {
    "split-merge": SplitMergeCrossover,
    "uniform": UniformCrossover,
    "half-fixed": HalfFixedCrossover
}

PERMUTATION_CROSSOVER: Dict[str, Type[Crossover]]= {
    "half-fixed": HalfFixedCrossover
}

__all__ = [
    "Crossover",
    "MultiCrossover",
    ALPHABET_CROSSOVER
]
