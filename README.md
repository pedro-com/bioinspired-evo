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

* Función de calculo de dominancia:
    - Dado una lista de puntos, obtener una matriz de dominancias (booleana), donde para cada valor aij = 1, si pi = pj o pi domina a pj.

* Función para calcular los frentes:
    - Usando la matriz de dominancia y los puntos, devolver un listado de listados de todos los frentes.

* Función para calcular distancia de Crowding:
    - Pasado una lista de valores (P_i, f_1(P_i), f_2(P_i)...)

* Funciones a optimizar:
    - Escribir y probar que funcionen las funciones a optimizar.

* Función de reducción de Clustering para elitismo

* Implementar evolución MOGA y NSAG2