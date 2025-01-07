from dataclasses import dataclass
from typing import Callable, Union, Dict, List, Any, Type, Tuple
import json
from itertools import product
import os
import time
import numpy as np

from .metrics import Metric
from ..evolutive import Evolutive
from ..utils import list_combinations

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)

def load_document_file(file_path:str):
    with open(file_path, mode='r', encoding='utf-8') as fd:
        lines = fd.readlines()
    return [json.loads(line) for line in lines]

def write_to_file(file_path:str, data: Dict):
    with open(file_path, mode='a', encoding='utf-8') as fd:
        fd.write(f'{json.dumps(data, cls=NumpyEncoder)}\n')

# STATIC_ARGS = ("alphabet", "cromolength", "phenotype", "maximize", "value_range")
LIST_ARGS = ("mutation", "crossover")
@dataclass
class GridSearch:
    save_file_name: str
    result_names: str
    fit: Callable[[Any], Any]
    n_individuals: int
    n_generations: int
    n_iterations: int
    Evolution: Type[Evolutive]
    static_evolution_kwargs: Dict[str, Any]
    evolution_kwargs: Dict[str, Union[Any, List[Any]]]
    evolution_metric: Metric
    exclude_combinations_list: tuple=()

    def __post_init__(self):
        self.file = f'{self.save_file_name}.jsonl'
        for list_arg in LIST_ARGS:
            if list_arg in self.evolution_kwargs:
                self.evolution_kwargs[list_arg] = list_combinations(self.evolution_kwargs[list_arg], self.exclude_combinations_list)
        self.evaluations = self.generate_evaluations()
        print(f"Number of Evaluations: {len(self.evaluations)}")
        if not os.path.exists(self.file):
            with open(self.file, 'w') as fd:
                fd.write("")
            self.results = []
        else:
            self.results = load_document_file(self.file)
        print(f"Current results: {len(self.results)}")

    def generate_evaluations(self):
        ev_keys = self.evolution_kwargs.keys()
        ev_values = self.evolution_kwargs.values()
        evaluations = [dict(zip(ev_keys, comb)) for comb in product(*ev_values)]
        for eval in evaluations:
            eval.update(self.static_evolution_kwargs)
        return evaluations

    def evolve(self, evolution_kwargs: Dict):
        evol = self.Evolution(n_individuals=self.n_individuals, **evolution_kwargs)
        results = []
        timings = 0
        for _ in range(self.n_iterations):
            start = time.perf_counter()
            results.append(evol.evolve(fit=self.fit, n_generations=self.n_generations))
            timings += time.perf_counter() - start
        res_metrics = self.evolution_metric([res["best"] for res in results])
        res_metrics["last_update_avg"] = sum(res["last_update"] for res in results) / self.n_iterations
        res_metrics["time_per_evol"] = timings / self.n_iterations
        return res_metrics
    
    def search(self):
        actual_res = len(self.results)
        for k in range(actual_res, len(self.evaluations)):
            print(f"Evaluation {k}: {self.evaluations[k]}")
            metrics = self.evolve(self.evaluations[k])
            metrics["id"] = f'{self.result_names}-{k}'
            self.results.append(metrics)
            print(f"Saving Evaluation {k}")
            write_to_file(self.file, metrics)
