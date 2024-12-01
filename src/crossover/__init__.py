from typing import Dict, Type, Union
from .real import RealCrossover, BLXCrossover, BLXAlphaCrossover
from .crossover import Crossover, MultiCrossover, RandomCrossover
from .sequence import SequenceCrossover, SplitMergeCrossover, UniformCrossover, HalfFixedCrossover

MULTI_CROSSOVER: Dict[str, Type[SequenceCrossover]]= {
    "random": RandomCrossover
}

ALPHABET_CROSSOVER: Dict[str, Type[SequenceCrossover]]= {
    "split-merge": SplitMergeCrossover,
    "uniform": UniformCrossover,
    "half-fixed": HalfFixedCrossover
}

REAL_CROSSOVER: Dict[str, Type[Union[SequenceCrossover, RealCrossover]]]= {
    "split-merge": SplitMergeCrossover,
    "uniform": UniformCrossover,
    "half-fixed": HalfFixedCrossover,
    "blx": BLXCrossover,
    "blx-alpha": BLXAlphaCrossover
}


__all__ = [
    RealCrossover,
    BLXCrossover,
    BLXAlphaCrossover,
    Crossover,
    MultiCrossover,
    RandomCrossover,
    SequenceCrossover,
    SplitMergeCrossover,
    UniformCrossover,
    HalfFixedCrossover
]
