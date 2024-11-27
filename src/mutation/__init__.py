from .mutation import Mutation, MultiMutation
from .alphabet import AlphabetMutation, RandomGeneMutation, SwapMutation, InsertMutation, ToOptMutation
from typing import Dict, Type

ALPHABET_MUTATION: Dict[str, Type[AlphabetMutation]] = {
    "random-gene": RandomGeneMutation,
    "swap-gene": SwapMutation,
    "insert-gene": InsertMutation,
    "to-opt": ToOptMutation
}

PERMUTATION_MUTATION: Dict[str, Type[AlphabetMutation]] = {
    "swap-gene": SwapMutation,
    "insert-gene": InsertMutation,
    "to-opt": ToOptMutation
}

__all__ = [
    "Mutation",
    "MultiMutation",
    "AlphabetMutation",
    "RandomGeneMutation",
    "SwapMutation",
    ALPHABET_MUTATION
]