import math
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def himmelblau(ch, onlyone=False):
	x = ch[0]
	y = ch[1]
	fxy = (x**2 + y - 11)**2 + (x + y**2 -7)**2
	if onlyone and (x>0 or y>0): # himmelblau modified to have only one global minimum (-3.77, -3.28)
		fxy += 0.5
	return fxy

def ackley(ch): # min in (0,0); huge set of local minima
	x = ch[0]
	y = ch[1]
	fxy = -20 * math.exp( -0.2 * (0.5 * (x**2 + y**2)) ** 0.5 ) \
		- math.exp (0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20
	return fxy

def beale(ch):
	x = ch[0]
	y = ch[1]
	term1 = (1.5 - x + x * y) ** 2
	term2 = (2.25 - x + x * y ** 2) ** 2
	term3 = (2.625 - x + x * y ** 3) ** 2
	return term1 + term2 + term3

def easom(ch):
	x = ch[0]
	y = ch[1]
	return -math.cos(x) * math.cos(y) * math.exp(-(x - math.pi)**2 - (y - math.pi)**2)

def goldstein_price_function(ch):
	x = ch[0]
	y = ch[1]
	term1 = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2))
	term2 = (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
	return term1 * term2

def rosenbrock(ch):
	x = ch[0]
	y = ch[1]
	a = 1
	b = 100
	term1 = (a - x) ** 2
	term2 = b * (y - x ** 2) ** 2
	return term1 + term2

def rosenbrock_ndim(x):
    a = 1
    b = 100
    sum = 0
    for i in range(len(x)-1):
        sum += (a - x[i])**2 + b * (x[i+1] - x[i]**2)**2
    return sum

def get_fit_function(function: Callable, non_zero_divisor: bool = False):
    def fit(x):
        divisor_adjustment = 1e-10 if non_zero_divisor else 0 # Avoid division by zero when target == None
        return 1 / (1 + function(x) + divisor_adjustment)
    return fit

def plot_contour(function: Callable, value_range: Tuple[int], points:int=400, marked_point: Tuple=None, figsize:Tuple=(12, 5)):
	x = np.linspace(value_range[0], value_range[1], points)
	y = np.linspace(value_range[0], value_range[1], points)
	X, Y = np.meshgrid(x, y)
	Z = np.array([function((x, y))for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
	plt.figure(figsize=figsize)
	filled_contour = plt.contourf(X, Y, Z, levels=50, cmap='plasma')  # Filled contours
	plt.colorbar(filled_contour)
	contour_lines = plt.contour(X, Y, Z, levels=50, colors='white', linewidths=0.5)
	plt.clabel(contour_lines, inline=True, fontsize=8)
	if marked_point is not None:
		plt.plot(marked_point[0], marked_point[1], 'ro')
		plt.text(marked_point[0] + 0.1, marked_point[1] + 0.1, f'({marked_point[0]:2.2f}, {marked_point[1]:2.2f})', color='r',
		   fontdict={"fontweight": "bold", "fontfamily": "DejaVu Sans", "fontsize": 12})
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.grid()


def plot_contour_3d(function: Callable, value_range: Tuple[int], points: int = 400, marked_point: Tuple = None, title: str = ''):

    x = np.linspace(value_range[0], value_range[1], points)
    y = np.linspace(value_range[0], value_range[1], points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([function((x, y)) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Plasma', opacity=0.8))

    if marked_point is not None:
        z_marked = function(marked_point)
        fig.add_trace(go.Scatter3d(
            x=[marked_point[0]],
            y=[marked_point[1]],
            z=[z_marked],
            mode='markers+text',
            marker=dict(size=6, color='red'),
            text=[f'({marked_point[0]:.2f}, {marked_point[1]:.2f}, {z_marked:.2f})'],
            textposition="top center",
			textfont=dict(size=10, color='black')  # Add text font properties
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1.25, y=1.25, z=0.5)
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

