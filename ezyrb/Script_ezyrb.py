import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from pod import POD
from database import Database
from rbf import RBF
from reducedordermodel import ReducedOrderModel as ROM

snapshots = np.load('tut1_snapshots.npy')
param = np.load('tut1_mu.npy')
print(snapshots.shape, param.shape)

tri = np.load('tut1_triangles.npy')
coord = np.load('tut1_coord.npy')
triang = mtri.Triangulation(coord[0],coord[1],tri)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
ax = ax.flatten()
for i in range(8):
   ax[i].triplot(triang, 'b-', lw=0.1)
   ax[i].tripcolor(triang, snapshots[i])
   ax[i].set_title('($\mu_0={:5.2f}, \mu_1={:5.2f})$'.format(*param[i]))
plt.show()

db = Database(param, snapshots)
pod = POD('svd')
rbf=RBF()
rom = ROM(db, pod, rbf)
rom.fit();
new_mu = [8,   1]
pred_sol = rom.predict(new_mu)
plt.figure(figsize=(5, 5))
plt.triplot(triang, 'b-', lw=0.1)
plt.tripcolor(triang, pred_sol);
plt.show()
'''
from ipywidgets import interact

def plot_solution(mu0, mu1):
    new_mu = [mu0, mu1]
    pred_sol = rom.predict(new_mu)
    plt.figure(figsize=(8, 7))
    plt.triplot(triang, 'b-', lw=0.1)
    plt.tripcolor(triang, pred_sol)
    plt.colorbar()

interact(plot_solution, mu0=8, mu1=1);
'''
for pt, error in zip(rom.database.parameters, rom.loo_error()):
    print(pt, error)
print(rom.optimal_mu())