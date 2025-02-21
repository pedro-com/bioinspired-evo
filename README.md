# bioinspired-evo
Evolución basada en métodos bioinspirados

* Crossovers:
    - Random split: Punto aleatorio y juntar dos listas.
    - Uniform crossover: Intercambiar valor a valor cada uno de los cromosomas.
    - Half cromosome Fixed positions in Ind1, add from INd2 the rest of the positions.

* Mutations:
    - Random gene mutation
    - Gene exchange (Permutation): Random selection of genes from 1st value, fill with the other value.
(1, 3), (4, 3), ....
    - Delete gene and insert random gene. OPT
    - To Opt: Random reverse.

Reference Page: <http://www.tomaszgwiazda.com/blendX.htm>

$$ch_1 = \beta p_1 + (1 - \beta) p_2$$
$$ch_2 = (1 - \beta) p_1 + \beta p_2$$
$$\beta \in [0, 1)^{dim(p)}$$

$$ch_1 = \beta p_1 + (1 + \alpha - \beta) p_2$$
$$ch_2 = (1 + \alpha - \beta) p_1 + \beta p_2$$
$$\beta \in [0, 1)^{dim(p)}$$
$$\alpha \in [0, 1]$$

$$gene_{mutation} \in [gene - \epsilon, gene + \epsilon]$$
$$\epsilon = muteps * len(alphabet)$$


## MultiObjective

* Dominance Function:
    * Obtains the number of points a specific point dominates
    $$[(p_i, i)...], i \in \mathbb{N}$$

* MultiObjectiveEvolutive:
    * ...

* MOGA:
    * Calcular fitness para las funciones g_i.
    $$F_i = N  - (|r_i| - 1)/2 - \sum_{k=1}^{r_i - 1}|k|$$
    * Mismo proceso evolutivo

* MOGA nichos

* Strenght Pareto



## Tareas

* Función de calculo de dominancia: Fernando Y Alejandro0
    - Dado una lista de puntos, obtener una matriz de dominancias (booleana), donde para cada valor aij = 1, si pi = pj o pi domina a pj.

* Función para calcular los frentes: Fernando Y Alejandro0
    - Usando la matriz de dominancia y los puntos, devolver un listado de listados de todos los frentes.

* Función para calcular distancia de Crowding: Fernando Y Alejandro0
    - Pasado una lista de valores (P_i, f_1(P_i), f_2(P_i)...)

PUNTOS = [(f1(P1), f2(P1)), (f1(P2), f2(P2)), (f1(P3), f2(P3), (f1(P4), f2(P4)))]

I = [0, 0, 0, 0]

for k in range(PUNTOS.shape[1]):
    args = np.argsort(PUNTOS[:, k])
    I[args[0]] = I[args[-1]] = inf
    for i in range(1, PUNTOS.shape[0] - 1):
        I[args[i]] += (I[args[i + 1]] - I[args[i - 1]]) / (I[args[-1]] - I[args[0]])

* Funciones a optimizar: Alejandro1
    - Escribir y probar que funcionen las funciones a optimizar.

* Función de reducción de Clustering para elitismo: Alejandro1

* Implementar evolución MOGA y NSAG2: Pedro0 + Evaluación

* Pedro1: Diapos + Notebook?


## Cosas que quedan

* Realizar la presentación: (Alex, Pedro y Fernando)
    - Diapos con el algoritmo que se utiliza (explicación + fotillos).
    - Resultados, tablitas con comparaciones + imágenes de los resultados

* Animación del algoritmo en funcionamiento. (Alex)
    - evolve se puede hacer a pasos (poniendo parte del número de generaciones y pasando la population anterior + best_cromosome para continuar con el previo).
    - .evolve : generations = 20 | 200generaciones algoritmo -> 10 veces -> .evolve(n_generations=20, seed_population=res["best_cromosome"] + res["population"])

* Función para plotear varios frentes de pareto de varias funciones: (Alejandro) OK
    * Poner colores y formas para cada frente.
    * Poner leyenda con el nombre de los frentes.
    * Recibe {nombre_frente: Puntos del frente}

* 3D plot del frente de pareto (parecido al de las rutas por ciudades pero con puntos). (Alejandro) OK

* Plot del mapa entre ciudades (Estaría guay que el tiempo estuviera representado como una escala de colores (por ejemplo azul a rojo y q los azules sean menos tiempo)). (Alex, Pedro y Fernando)

* Notebook de resultados:
    * Sección 1: Celdas primeras introduciendo los algoritmos que se han implementado con graficos (estaría hecho pero ponerlo bonito). (Alex, Pedro y Fernando)
    * Sección K: Una sección para cada problema que teníamos que resolver. (Alejandro)
        - Empezar con GridSearch -> Pedro | Evaluar 10 veces y sacar media de metricas.
        - Foto comparando pareto con un buen resultado del nuestro
        - Foto comparando pareto + nsga2 + nuestro en una gráfica

$$fit_f = \frac{2*(total_{indv} + \frac{indv_f - 1}{2} + \sum_{j=1}^{f - 1}indv_j)}{n_{indv} + 1}$$ 