from gen_evolutive import GeneticEvolutive
from typing import Callable, Union, Dict, List, Any, Type, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

def evaluate_evolutive(
        evolutive: GeneticEvolutive,
        fit: Callable,
        evaluation_func: Callable,
        n_generations: int,
        maximize: bool=True,
        n_tries: int = 10
        ):
    best_individuals = []
    for _ in range(n_tries):
        results = evolutive.evolve(fit, n_generations)
        best_individuals.append(results["best"])
    indv_scores = [(indv, evaluation_func(indv)) for indv in best_individuals]
    indv_scores.sort(key=lambda v: v[1], reverse=not maximize)
    scores = list(map(lambda v: v[1], indv_scores))
    return {
        "n_generations": n_generations,
        "best_individuals": list(map(lambda v: v[0], indv_scores)),
        "scores": scores,
        "max": np.max(scores),
        "min": np.min(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores)
    }

def to_pd_dataframe(evolutive_metrics: list, except_values: tuple=("best_individuals", "scores")):
    filtered_evaluation = [
        {key: dct[key] for key in dct if key not in except_values}
        for dct in evolutive_metrics
    ]
    return pd.DataFrame(filtered_evaluation)

def plot_scores_t1(evolutive_metrics: list, ax: plt.Axes=None, figsize: tuple=(10, 6)):
    if not ax:
        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = to_pd_dataframe(evolutive_metrics)
    sns.barplot(data=df, y="scores", hue="generations")

def plot_scores(evolutive_metrics: list, label:str, color:str="red", ax: plt.Axes=None, figsize: tuple=(10, 6)):
    if not ax:
        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(evolutive_metrics["scores"], color=color, linestyle='-o', label=label)
    ax.axhline(evolutive_metrics["mean"], color=color, linestyle='--')
    median = evolutive_metrics["scores"].index(evolutive_metrics["median"])
    ax.plot(median, evolutive_metrics["median"], color=color, linestyle='x')

@dataclass
class GridSearch:
    save_file: str
    evolution: Type[GeneticEvolutive]
    evolution_kwargs: Dict[str, Union[Any, List[Any], Tuple[Any]]]
    combinations: int=1

