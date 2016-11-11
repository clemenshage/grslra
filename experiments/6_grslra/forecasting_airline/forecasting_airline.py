from grslra import grslra_online, slra_by_factorization_online
import numpy as np
from grslra.structures import Hankel
from grslra.scaling import Scaling

# First data set is the classical Air Passengers data

csvfile = np.genfromtxt('AirPassengers.csv', delimiter=",", skip_header=1)
x = csvfile[0:-1,1]

p = 0.1
mu = 0.03

params_cg_U = {"t_init": 0.05}
grslra_params = {"VERBOSE": 1, "PRINT": None}
scaling = Scaling(centering=True)

N = x.size
k = 8
# corresponds to an observation of 3 years
m = 18
n = N - m + 1
# forecast the next half year
N_f = 6

hankel = Hankel(m, m + N_f)

l_grslra, angles, t = grslra_online(x, hankel, k, p, mu, N_f, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling)

np.savez('predictions_old.npz', l_grslra=l_grslra, angles=angles, x=x, m=m, N=N, N_f=N_f, t=t)

# Run the Python implementation of the SLRAbyF method (Ishteva et al., 2014)
l_slrabyf, t_slrabyf = slra_by_factorization_online(x, m, k, N_f)
np.savez('predictions_old_slrabyf.npz', l_slrabyf=l_slrabyf, t_slrabyf=t_slrabyf)


# Second data set is a more recent one
csvfile = np.genfromtxt('airline_passengers_total_1996-2014.csv', delimiter=",", skip_header=4, skip_footer=3)
x = csvfile[0:-1,2]

p = 0.1
mu = 0.03

params_cg_U = {"t_init": 0.05}
grslra_params = {"VERBOSE": 1, "PRINT": None}
scaling = Scaling(centering=True)

N = x.size
k = 8
m = 18
n = N - m + 1
N_f = 6

hankel = Hankel(m, m + N_f)

l_grslra, angles, t = grslra_online(x, hankel, k, p, mu, N_f, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling)

np.savez('predictions_new.npz', l_grslra=l_grslra, angles=angles, x=x, m=m, N=N, N_f=N_f, t=t)


l_slrabyf, t_slrabyf = slra_by_factorization_online(x, m, k, N_f)
np.savez('predictions_new_slrabyf.npz', l_slrabyf=l_slrabyf, t_slrabyf=t_slrabyf)

