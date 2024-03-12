from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()

x = [i for i in range(size)]
y = [j for j in range(size)]

plt.plot(x,y)
plt.savefig('plot.png')