import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import pandas as pd
class Animation():
    def __init__(self, problem, n_generations: int, ind_per_generation: int):
        self.problem = problem
        self.n_generations = n_generations
        self.ind_per_generation = ind_per_generation
        self.scores = {}

    def evolve_step_by_step(self, evolutive, fit, trace: int=25, pareto_like: np.ndarray=None) -> pd.DataFrame:
        seed_population = []
        for idx, _ in enumerate(range(int(self.n_generations/self.ind_per_generation)), 1):
            res = evolutive.evolve(fit=fit, n_generations=self.ind_per_generation, trace=trace, seed_population=seed_population)
            fit_scores = np.array([self.problem.evaluate(p) for p in res["best"]])
            if len(fit_scores.shape) == 3:
                fit_scores = fit_scores[:, 0, :]
            self.scores[idx] = fit_scores
            seed_population=res["best_cromosome"] + res["population"]
            print('--------------')
        rows = []
        for key, array in self.scores.items():
            for row in array:
                rows.append({"Generacion": key * self.ind_per_generation, "X": row[0], "Y": row[1]})

        return self.get_animation(pd.DataFrame(rows), pareto_like)

    def get_animation(self, df, pareto_like:np.ndarray):
        fig, ax = plt.subplots()

        # Dibujar los puntos iniciales del Pareto
        if pareto_like is None:
            pareto = self.problem.pareto_front()
            ax.scatter(pareto[:, 0], pareto[:, 1], label='pareto', color='blue')
        else:
            ax.scatter(pareto_like[:, 0], pareto_like[:, 1], label='pareto', color='blue')
        # Crear el scatter plot vacío para los puntos animados
        scatter_anim = ax.scatter([], [], label='obtained', color='red')

        ax.legend()

        # Acumular puntos a medida que se avanza en la animación
        all_points = []

        # Animar por grupos
        def animar(frame):
            # Seleccionamos el grupo correspondiente al frame
            grupo = frame + 1
            data_grupo = df[df["Generacion"] == grupo * self.ind_per_generation]
            scatter_anim.set_offsets(data_grupo[["X", "Y"]].values)  # Actualiza los puntos
            # if grupo == 0:
            #     ax.set_title(f'Generación {(grupo +1) * IND_PER_GENERATION}', fontsize=14)
            # else:
            ax.set_title(f'Generación {grupo * self.ind_per_generation}', fontsize=14)

        # Crear la animación
        n_frames = df["Generacion"].nunique()  # Total de grupos

        return FuncAnimation(fig, animar, frames=n_frames, repeat=False, interval=750)

    '''
    def evolve_step_by_step(self, evolutive) -> pd.DataFrame:
        if self.problem.name() == 'MW7':
            for idx, _ in enumerate(range(int(self.n_generations/self.ind_per_generation)), 1):
                if idx == 1:
                    res = evolutive.evolve(fit=self.problem.evaluate, n_generations=self.ind_per_generation, trace=10, obtain_metrics=True)
                    fit_scores = np.array([self.problem.evaluate(p)[0] for p in res["best"]])
                    self.scores[idx] = fit_scores
                else:
                    res = evolutive.evolve(fit=self.problem.evaluate, n_generations=self.ind_per_generation, trace=10, obtain_metrics=True, seed_population=res["best_cromosome"] + res["population"])
                    fit_scores = np.array([self.problem.evaluate(p)[0] for p in res["best"]])
                    self.scores[idx] = fit_scores
                print('--------------')
        else:
            for idx, _ in enumerate(range(int(self.n_generations/self.ind_per_generation)), 1):
                if idx == 1:
                    res = evolutive.evolve(fit=self.problem.evaluate, n_generations=self.ind_per_generation, trace=10, obtain_metrics=True)
                    fit_scores = np.array([self.problem.evaluate(p) for p in res["best"]])
                    self.scores[idx] = fit_scores
                else:
                    res = evolutive.evolve(fit=self.problem.evaluate, n_generations=self.ind_per_generation, trace=10, obtain_metrics=True, seed_population=res["best_cromosome"] + res["population"])
                    fit_scores = np.array([self.problem.evaluate(p) for p in res["best"]])
                    self.scores[idx] = fit_scores
                print('--------------')

        rows = []
        for key, array in self.scores.items():
            for row in array:
                rows.append({"Generacion": key * self.ind_per_generation, "X": row[0], "Y": row[1]})

        return self.get_animation(pd.DataFrame(rows))
    '''