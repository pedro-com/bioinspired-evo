from typing import Dict, Type
from .crossover import Crossover, MultiCrossover
from .alphabet import SplitMergeCrossover

ALPHABET_CROSSOVER: Dict[str, Type[Crossover]]= {
    "split-merge": SplitMergeCrossover
}


__all__ = [
    "Crossover",
    "MultiCrossover",
    ALPHABET_CROSSOVER
]
