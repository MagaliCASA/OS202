# Examen machine du 21 Mars 2023

## Configuration de l'ordinateur utilisé :

Nombre de coeurs physiques de la machine : 4
  
Nombre de coeurs logiques de la machine : 4

Quantité de mémoire cache L2 et L3 de la machine :

Cache L1d : 128 KiB
Cache L1i : 128 KiB
Cache L2 : 1 MiB
Cache L3 : 6 MiB

## Colorisation d'une photo noir et blanc

Le but de ce programme est de paralléliser un algorithme qui colorie une photo noir et blanc à partir d'une photo marquée.

### Parallélisation 

Dans un premier temps, faites une partition de l'image en nbp tranches d'images et demandez à chaque processus d'essayer de coloriser sa portion d'image à partir des conditions de Dirichlet correspondant à sa portion d'image et en construisant une matrice uniquement locale à cette portion d'image. Vous nommerez le fichier parallélisé colorize1.py



| 0    | 1      | 2         | 3         |
|------|--------|-----------|-----------|
| 0-63 | 64-127 | 128 - 191 | 192 - 255 |

Étant donné que l'on doit calculer le même nombre d'étapes (il n'y a pas d'autres 
conditions d'arrêts que le nombre d'itérations) puis enregistrer le résultat, je
ne vois pas de raison apriori qui pourrait expliquer que certains processus
soit plus long que d'autres.


2. Créer une courbe donnant l'accélération obtenue avec votre parallélisation (jusqu'à la limite du nombre de coeur physique présent sur votre ordinateur).

Sans parallélisation (sans mpi), le programme prend 0.440092 s pour générer les cellules et 
4,74 s pour enregistrer le résultat.

<img src="temps_enregistrement.png" alt="" width="600">

<img src="temps_calcul.png" alt="" width="600">

De plus l'hypothèse que les processus prennent environ la même durée se vérifie : l'écart type des temps de calcul 
est de l'ordre de 4 * 10^{-3} et celle du temps d'enregistrement de 6 * 10^{-2}.


_Remarque :_ Pour Thread=1, cela correspond à un seul thread avec mpi.

**Remarque** : Pour vérifier si les images contiennent des erreurs ou non, on peut vérifier que les fichiers images sont les mêmes qu'avec le code séquentiel en utilisant :

    md5sum -c check_resultats_md.md5sum
ou

    md5sum -c check_resultats_png.md5sum  # si vous avez choisi save_as_png

## Calcul d'une enveloppe convexe

On veut calculer l'enveloppe convexe d'un nuage de point sur le plan. Pour cela on utilise l'algorithme de Graham décrit dans le lien suivant :

    https://fr.wikipedia.org/wiki/Parcours_de_Graham

On obtient en sortie une sous-liste de points du nuage qui définissent l’enveloppe convexe. Ces points sont rangés de manière à parcourir le polygone de l’enveloppe dans le sens direct.

Le code séquentiel peut être trouvé dans le fichier `enveloppe_convexe.py`. En sortie, le code affiche les points et l'enveloppe convexe à l'écran.

Afin de paralléliser le code en distribué avec MPI, on veut distribuer les sommets sur plusieurs processus puis utiliser l’algorithme suivant :

- Calculer l’enveloppe convexe des sommets locaux de chaque processus
- Puis en échangeant deux à deux entre les processus les enveloppes convexes locales, calcul sur chacun la fusion des deux enveloppes convexes en remarquant que
l’enveloppe convexe de deux enveloppes convexe est l’enveloppe convexe de la réunion
des sommets définissant les deux enveloppes convexes.

1. Dans un premier temps, mettre en œuvre l’algorithme sur deux processus.

Sans parallélisation, la génération de points prend 0.0172 s et le calcul de l'enveloppe convexe prend
2,9552 s.

Il semblerait que la fonction de calcul d'enveloppe convexe soit erronée (à cause
des arrondis) : sans mpi, ajouter `enveloppe = calcul_enveloppe(enveloppe)` a pour conséquence
de retirer 3 points de l'enveloppe convexe. La fonction de vérification ne fonction alors plus.

2. Dans un deuxième temps, en utilisant un algorithme de type hypercube, de sorte qu’un processus fusionne son enveloppe convexe avec le processus se trouvant dans la direction d, mettre en œuvre l’algorithme sur `2**n` processus.



3. Mesurer les speed ups de votre algorithme en suivant le critère de Amdhal et de Gustafson. Interprétez votre résultat au regard de la complexité de l'algorithme et commentez.

_Non faite_

---

### Exemple sur 8 processus

- Les processus 0 à 7 calculent l’enveloppe convexe de leur nuage de points local.
- Le processus 0 et le processus 1 fusionnent leurs enveloppes, idem pour 2 avec 3, 4 avec 5 et 6 avec 7.
- Le processus 0 et le processus 2 fusionnent leurs enveloppes, idem pour 1 avec 3, 4 avec 6 et 5 avec 7.
- Le processus 0 et le processus 4 fusionnent leurs enveloppes, idem pour 1 avec 5, 2 avec 6 et 3 avec 7.

---
