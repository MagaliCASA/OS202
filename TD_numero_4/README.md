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

Concernant le speed up, on 
