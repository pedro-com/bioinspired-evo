from .mutation import Mutation, MultiMutation
from .alphabet import AlphabetMutation, RandomGeneMutation, SwapMutation
from typing import Dict, Type

ALPHABET_MUTATION: Dict[str, Type[AlphabetMutation]] = {
    "random-gene": RandomGeneMutation,
    "swap-mutation": SwapMutation
}

PERMUTATION_MUTATION: Dict[str, Type[AlphabetMutation]] = {
    "swap-mutation": SwapMutation
}

__all__ = [
    "Mutation",
    "MultiMutation",
    "AlphabetMutation",
    "RandomGeneMutation",
    "SwapMutation",
    ALPHABET_MUTATION
]