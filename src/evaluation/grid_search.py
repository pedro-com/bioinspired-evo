from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union, Dict, List, Any, Type, Tuple
import json

from ..evolutive import Evolutive

def load_document_file(file_path:str):
    with open(file_path, mode='r', encoding='utf-8') as fd:
        lines = fd.readlines()
    return [json.loads(line) for line in lines]

@dataclass
class GridSearch:
    save_file: str
    fit: Callable[Any]
    n_individuals: int
    n_generations: int
    evolution: Type[Evolutive]
    def evaluate()