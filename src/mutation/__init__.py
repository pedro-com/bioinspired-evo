from .mutation import Mutation, MultiMutation
from .vocabulary import VocabularyMutations, RandomGeneMutation
from typing import Dict, Type

VOCABULARY_MUTATION: Dict[str, Type[VocabularyMutations]] = {
    "random-gene": RandomGeneMutation
}

__all__ = [
    "Mutation",
    "MultiMutation",
    "VocabularyMutations",
    "RandomGeneMutation",
    VOCABULARY_MUTATION
]