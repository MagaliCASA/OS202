import matplotlib.pyplot as plt


x = [i+1 for i in range(4)]
y = [68.75,43.3,34.71,30.89]


plt.plot(x,y)
plt.title('Temps d execution de colorize1.py en fonction du nombre de processeurs')
plt.xlabel('Nombre de processeurs')
plt.ylabel('Temps d execution total')
plt.savefig('colorize1.png')