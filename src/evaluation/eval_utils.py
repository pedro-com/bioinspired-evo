import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def to_pd_dataframe(evolutive_metrics: list, exclude_values: tuple):
    filtered_evaluation = [
        {key: dct[key] for key in dct if key not in exclude_values}
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