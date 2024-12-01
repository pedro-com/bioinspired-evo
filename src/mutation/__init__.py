from .mutation import Mutation, MultiMutation, RandomMutation
from .sequence import SequenceMutation, SwapMutation, InsertMutation, ToOptMutation
from .alphabet import AlphabetMutation, RandomGeneMutation, RandomLocalGeneMutation
from .real import RealNumberMutation, RandomRangeMutation, RandomLocalMutation
from typing import Dict, Type, Union

MULTIMUTATION_DICT: Dict[str, Type[MultiMutation]] = {
    "random": RandomMutation
}

ALPHABET_MUTATION: Dict[str, Type[Union[SequenceMutation, AlphabetMutation]]] = {
    "random-gene": RandomGeneMutation,
    "random-local": RandomLocalGeneMutation,
    "swap-gene": SwapMutation,
    "insert-gene": InsertMutation,
    "to-opt": ToOptMutation
}

REAL_MUTATION: Dict[str, Type[Union[SequenceMutation, RealNumberMutation]]] = {
    "random-range": RandomRangeMutation,
    "random-local": RandomLocalMutation,
    "swap-gene": SwapMutation,
    "insert-gene": InsertMutation,
    "to-opt": ToOptMutation
}

__all__ = [
    Mutation,
    MultiMutation,
    RandomMutation,
    SequenceMutation,
    SwapMutation,
    InsertMutation,
    ToOptMutation,
    AlphabetMutation,
    RandomGeneMutation,
    RandomLocalGeneMutation,
    RealNumberMutation,
    RandomRangeMutation,
    RandomLocalMutation,
    MULTIMUTATION_DICT,
    ALPHABET_MUTATION,
    REAL_MUTATION
]