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


