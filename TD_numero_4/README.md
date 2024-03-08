# TD 4 - Jeu de la vie

Ce TD avait pour but de paralléliser le fameux jeu de la vie. 

Le premier fichier "lifegame.py" est une parallélisation du jeu en 2 processus : le processus 0 se charge de l'affichage des cellules, tandis que le processus 1 calcule l'évolution des cellules. 

Le second fichier "lifegame_2.py" est une parallélisation du jeu en 3 processus : le processus 0 se charge de l'affichage des cellules, pendant que le processus 1 calcule l'évolution des cellules dans la partie gauche de la fenêtre, et le processus 2 calcule l'évolution des cellules dans la partie gauche de la fenêtre. 

Le troisième fichier "lifegame_3_4processus.py" est une parallélisation du jeu en 4 processus : le processus 0 se charge de l'affichage des cellules ainsi que du calcul de l'évolution des cellules dans le coin en bas à droite de la fenêtre, tandis que processus 1 calcule l'évolution des cellules dans le coin en haut à gauche, le processus 2 calcule l'évolution des cellules dans le coin en haut à droite, et le processus 3 calcule l'évolution des cellules dans le coin en bas à gauche. 

Enfin, le dernier fichier "lifegame_3_5processus.py" est une parallélisation du jeu en 5 processus : le processus 0 se charge de l'affichage des cellules, tandis que processus 1 calcule l'évolution des cellules dans le coin en haut à gauche, le processus 2 calcule l'évolution des cellules dans le coin en haut à droite, le processus 3 calcule l'évolution des cellules dans le coin en bas à gauche, et le processus 4 calcule l'évolution des cellules dans le coin en bas à droite. 

Brièvement, le decoupage se fait dès le début, donc les processus ont tous un nombre de cellules égal, sauf pour le dernier processus qui peut éventuellement avoir des cellules en plus si le nombre de cellules n'est pas divisible par 4 (ou 2 dans le cas ou seuls 2 processus calculent l'évolution des cellules).

La subtilité du découpage des cellules est que pour calculer l'évolution des cellules au rang n+1 sur le bord de leur domaine, les processus ont besoin de connaître l'état des cellules au rang n. 
Ainsi, chaque processus effectue son calcul d'évolution des cellules de son domaine au rang n+1, comme suit : 
- Le processus i reçoit l'état des cellules frontalières avec son domaine au rang n, ce qui lui permet, par la suite, de calculer les cellules du bord de son domaine au rang n+1. 
- Le processus i procède au calcule de l'évolution des cellules de son domaine en prenant en compte les cellules fantômes
- Le processus i envoie au processus 0, l'état de ses cellules au rang n+1, sans prendre en compte les cellules fantômes.
- Le processus 0 s'occupe de l'affichage des cellules collectées. 

Ce problème est 'nearly embarassingly parallel', en effet, les processus doivent communiquer entre eux pour s'envoyer leurs cellules fantômes et ainsi pouvoir calculer l'évolution des cellules en bord de domaine.

Ce problème est également mal équilibré, en effet, un processus peut avoir plus de cellules à calculer si le nombre de cellules n'est pas divisible par le nombre de processus "calculateurs". Il n'y a pas d'équilibrage grâce un algorithme maître-esclave, qui semble inadapté pour ce genre de problème puisque les cellules doivent être calculées par morceaux. 

Tous les codes ont été vérifiés et donnent le résultat attendu. Malheureusement, le dernier fichier ("lifegame_3_5processus.py") n'a pas pu être testé jusqu'à l'affichage puisque je n'ai à disposition que des ordinateurs possédant 4 coeurs. 

j'ai utilisé la formule de . Elle m'a semblé la plus adaptée car 
Pour cela, j'ai calculé f, la proportion du programme (en temps CPU séquentiel) qui ne peut être parallélisé et fonctionne donc en séquentiel, en faisant appelle à la fonction time.time(). 
Ainsi j'obtiens un speed up de , ce qui était attendu, car j'ai multiplé par 4 le nombre de processus. 

Concernant le speed up, on se contentera du calcul pour le "lifegame_3_4processus".
Le speed-up représente la rapidité gagnée en ayant parallélisé la partie du code parallélisable. Il est calculé comme le rapport entre le temps du programme dans sa version séquentiel et sa version parallélisée. 

Pour le calculer j'ai choisi un nombre arbitraire d'itérations _n_ pour la boucle while : clé de la durée du code. Il faut donc mettre deux marqueurs de temps : un avant la boucle et un autre après la boucle (donc après _n_ itérations).
### Calcul du Speed-up (\(S(n)\))

On peut ainsi calculer le speed-up :

S(n) = T_s(n)/T_p(n)

avec \T_s\ le temps en séquentiel, \T_p\ le temps parallélisé, et *n* le nombre d'itérations.

## Résultats du Speed-up pour 4 processus

| Itérations | Durée Séquentielle (en *s*) | Durée Parallèle (en *s*) | Speed-up |
|------------|-----------------------------|--------------------------|----------|
| 30         | 8,05                        | 7,95                     | 1,01     |
| 100        | 27,5                        | 25,8                     | 1,07     |
| 300        | 78,7                        | 76,1                     | 1,03     |

*Tableau 1 : Résultat du speed-up pour 4 processus.*

On remarque que les speed-ups sont quasiment égaux à 1. Cela veut dire que le code parallélisé n'est que légèrement plus rapide, c'est quasiment le même temps d'exécution. Autrement dit la complexité ajoutée par la communication entre les processus et la synchronisation des datas entre ces derniers est trop élevée comparée au temps gagné par la parallélisation des calculs. Cependant, cela reste insatisfaisant et nécessite un travail supplémentaire pour paralléliser de façon plus efficace l'algorithme, que nous aurions continué à faire si nous avions eu du temps supplémentaire. 
