from matplotlib import pyplot as plt
import matplotlib
from scipy.io import loadmat
import numpy as np
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

data_grslra = np.load('predictions_old.npz')
data_slrabyf = np.load('predictions_old_slrabyf.npz')
data_arima = loadmat('arima/predictions_old_arima.mat')

x = data_grslra["x"]
l_grslra = data_grslra["l_grslra"]
m = data_grslra["m"]
N = data_grslra["N"]
N_f = data_grslra["N_f"]
t_grslra = data_grslra["t"]

l_slrabyf = data_slrabyf["l_slrabyf"]
t_slrabyf = data_slrabyf["t_slrabyf"]

l_arima = data_arima["x_hat"].flatten()
t_arima = data_arima["t"].flatten()

plt.figure(figsize=(15,5))
plt.hold(True)

plt.plot(x, label="data", linewidth=3, color='black',zorder=0)
plt.plot(l_grslra, label="GRSLRA", linewidth=2, color='g', zorder=3)
plt.plot(l_slrabyf, label="SLRAbyF", linewidth=2, color='r', zorder=1)
plt.plot(l_arima, label="ARIMA", linewidth=2, color='b', zorder=2)

#startpoint=60
startpoint = 2*m-1 + N_f

norm_data = np.linalg.norm(x[startpoint: N])

err_grslra = np.linalg.norm(x[startpoint : N] - l_grslra[startpoint : N]) / norm_data
err_slrabyf = np.linalg.norm(x[startpoint : N] - l_slrabyf[startpoint : N]) / norm_data
err_arima = np.linalg.norm(x[startpoint : N] - l_arima[startpoint : N]) / norm_data

avtime_grslra = np.mean(t_grslra[startpoint : N])
avtime_slrabyf = np.mean(t_slrabyf[startpoint : N])
avtime_arima = np.mean(t_arima[startpoint : N])

plt.xticks(np.arange(0,x.size, step=12), np.arange(1949,1961))
plt.yticks(np.arange(0,800, step=200), np.arange(0,8,2))
plt.ylabel("$\\times 10^5$ monthly passengers")
axes = plt.gca()
axes.set_ylim([0, 700])
axes.set_xlim(2*m - 1, N + 1)


plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()
plt.legend()

print "\nOld data set\n"

print "GRSLRA:\n"
print "relative error: ", err_grslra, "average time: ", avtime_grslra, "\n"

print "SLRAbyF:\n"
print "relative error: ", err_slrabyf, "average time: ", avtime_slrabyf, "\n"

print "ARIMA:\n"
print "relative error: ", err_arima, "average time: ", avtime_arima, "\n"

plt.savefig('airline_old.pdf', dpi=200)