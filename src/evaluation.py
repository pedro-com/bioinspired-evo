from evolutive import Evolutive
from typing import Callable, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_evolutive(evolutive: Evolutive, fit: Callable, evaluation_func: Callable, n_generations_values: Union[int, tuple], n_tries: int = 10):
    if isinstance(n_generations_values, int):
        n_generations_values = (n_generations_values,)
    evolutive_metrics = []
    for n_generations in n_generations_values:
        evaluation_dict = {"n_generations": n_generations, "best_individuals": []}
        for _ in range(n_tries):
            results = evolutive.evolve(fit, n_generations)
            evaluation_dict["best_individuals"].append(results["best"])
        evaluation_dict["scores"] = [evaluation_func(indv) for indv in evaluation_dict["best_individuals"]]
        evaluation_dict.update({
            "max": np.max(evaluation_dict["scores"]),
            "min": np.min(evaluation_dict["scores"]),
            "mean": np.mean(evaluation_dict["scores"]),
            "median": np.median(evaluation_dict["scores"]),
            "std": np.std(evaluation_dict["scores"])
        })
        evolutive_metrics.append(evaluation_dict)
    return evolutive_metrics

def to_pd_dataframe(evolutive_metrics: list, except_values: tuple=("best_individuals", "scores")):
    filtered_evaluation = [
        {key: dct[key] for key in dct if key not in ("best_individuals", "scores")}
        for dct in evolutive_metrics
    ]
    return pd.DataFrame(filtered_evaluation)

def plot_scores(evolutive_metrics: list, ax: plt.Axes=None, figsize: tuple=(10, 6)):
    if not ax:
        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = to_pd_dataframe(evolutive_metrics)
    sns.barplot(data=df, y="scores", hue="generations")
    

