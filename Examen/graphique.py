import matplotlib.pyplot as plt


x = [i+1 for i in range(4)]
y = [29.31/68.75,29.31/43.3,29.31/34.71,29.31/30.89]


plt.plot(x,y)
plt.title('Temps d execution de colorize1.py en fonction du nombre de processeurs')
plt.xlabel('Nombre de processeurs')
plt.ylabel('Temps d execution total')
plt.savefig('colorize1.png')