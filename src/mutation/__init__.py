from .mutation import Mutation, MultiMutation
from .alphabet import AlphabetMutation, RandomGeneMutation
from typing import Dict, Type

ALPHABET_MUTATION: Dict[str, Type[AlphabetMutation]] = {
    "random-gene": RandomGeneMutation
}

__all__ = [
    "Mutation",
    "MultiMutation",
    "AlphabetMutation",
    "RandomGeneMutation",
    ALPHABET_MUTATION
]