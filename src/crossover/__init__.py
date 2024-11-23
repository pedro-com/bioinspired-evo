from typing import Dict, Type
from .crossover import Crossover, MultiCrossover
from .vocabulary import SplitMergeCrossover

VOCABULARY_CROSSOVER: Dict[str, Type[Crossover]]= {
    "split-merge": SplitMergeCrossover
}

__all__ = [
    "Crossover",
    "MultiCrossover",
    VOCABULARY_CROSSOVER
]
