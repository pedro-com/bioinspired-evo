import random
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def dist(x, y): # distancia euclídea
	return ((x[0] - y[0])**2 + (x[1] - y[1])**2 ) ** 0.5

def globaldist(cities): # calcula la ditancia global de una solución
	acum = 0.0
	for i in range(len(cities)):
		acum += dist(cities[i-1],cities[i]) # -1 is the last element
	return acum

def fit_cities(cities): # fitness para TSP
	return 1 / (1 + globaldist(cities))

def plot_cities(cities_path: Iterable, pad:float=0.1, axes: plt.Axes=None):
	plt.ion()
	if axes is None:
		fig = plt.figure()
		axes = fig.add_subplot(111)
	circle = np.array(list(cities_path) + [cities_path[0]])
	min_v = np.min(circle, axis=0)
	max_v = np.max(circle, axis=0)
	axes.set_xlim(min_v[0] - pad, max_v[0] + pad)
	axes.set_ylim(min_v[1] - pad, max_v[1] + pad)
	axes.plot(circle[:, 0], circle[:, 1],'g')
	axes.plot(circle[:, 0], circle[:, 1],'ro')
	axes.set_title('Length=%5.3f' % globaldist(cities_path))

def random_cities(cities:int, dimensions:int):
	return np.random.random(size=(cities, dimensions))

def npdist(x, y):
	return np.sqrt(np.sum((x - y)**2))

def npglobaldist(cities):
	acum = 0.0
	for i in range(len(cities)):
		acum += dist(cities[i-1], cities[i]) # -1 is the last element
	return acum

def fit_npcities(cities):
	return 1 / npglobaldist(cities)

def plot_3dcities(cities_path: Iterable, frames:int=360, interval:int=50, pad:float=0.1, axes: plt.Axes=None):
    plt.ion()
    cities_path = np.array(list(cities_path))
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    circle = np.vstack([cities_path, cities_path[0]])
    min_v = np.min(circle, axis=0)
    max_v = np.max(circle, axis=0)
    
    axes.set_xlim(min_v[0] - pad, max_v[0] + pad)
    axes.set_ylim(min_v[1] - pad, max_v[1] + pad)
    axes.set_zlim(min_v[2] - pad, max_v[2] + pad)
    
    axes.plot(circle[:, 0], circle[:, 1], circle[:, 2], 'g-')
    axes.plot(circle[:, 0], circle[:, 1], circle[:, 2], 'ro')
    
    axes.set_title('Length=%5.3f' % globaldist(cities_path))
    def update(frame):
        axes.view_init(elev=10, azim=frame)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.close(fig)
    return anim
