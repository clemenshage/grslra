from scipy.io import loadmat
import numpy as np
import grslra
from grslra.structures import ConcatenatedSkewSymmetric
from grslra.structures import SkewSymmetric
from grslra.scaling import Scaling_Percentile
from grslra.tools import imagesc
from grslra.grslra_batch import grslra_test
from grslra.visualization import plot_lpnorm
from matplotlib import cm
from matplotlib import pyplot as plt


def plot_comparison(data, teamnames, filename, title, clim=None, cbar=True):
    fig, ax = plt.subplots(figsize=(19.2,10.8))
    cax = plt.imshow(data, interpolation='nearest', cmap=cm.coolwarm, clim=clim)
    ax.set_title(title)
    plt.xticks(xrange(24), teamnames, rotation='vertical')
    plt.yticks(xrange(24), teamnames)
    if cbar:
        fig.colorbar(cax)
    fig.subplots_adjust(bottom = 0.2)
    plt.savefig('figures/' + filename + '.png')
    plt.close()

def plot_ranking(ranking, teamnames, filename, title):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.set_title(title)
    ax1, = plt.plot(xrange(24), ranking)
    plt.xticks(xrange(24), teamnames, rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('figures/' + filename + '.png')
    plt.close()


teamnames= np.array(['Albania', 'Austria', 'Belgium', 'Croatia', 'Czech Republic', 'England', 'France', 'Germany', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Northern Ireland', 'Poland', 'Portugal', 'Romania', 'Russia', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'Wales'])

data = loadmat('../data/em.mat')
X = data["X"]
nofteams=X.shape[0]
# X[X>1] = 1
# X[X<-1] = -1



Omega = (data['row'].flatten() - 1, data['col'].flatten() - 1)
X_Omega = X[Omega]
Omega_full = np.zeros((nofteams, nofteams))
Omega_full[Omega] = 1
Omega_matches_full = np.triu(Omega_full,1)
Omega_matches = np.where(Omega_matches_full)
X_Omega_matches=X[Omega_matches]

plot_comparison(data["X"], teamnames, 'em_results', 'Actual goal differences of EURO 2016 matches')
plot_comparison(Omega_full, teamnames, 'em_matches', 'Observed matches of EURO 2016', clim=(-1, 1), cbar=False)

ranking_average = np.sum(data['X'], axis=1) / (np.sum(Omega_full, axis=1) - 1)
sorted_ix_average = np.argsort(ranking_average)
plot_ranking(ranking_average[sorted_ix_average], teamnames[sorted_ix_average], 'em_ranking_avgoals', 'Naive ranking according to average goal difference per match')

L_av= np.outer(ranking_average, np.ones((nofteams,))) - np.outer(np.ones((nofteams,)), ranking_average)
plot_comparison(L_av, teamnames, 'em_comparison_avgoals', 'Goal difference predictions based on average goal difference per match')
plot_comparison(Omega_matches_full * (X - L_av), teamnames, 'em_comparison_avgoals_error', 'RMSE Error of prediction based on average goal difference per match: ' + '{:1.3f}'.format(np.linalg.norm(X_Omega_matches - L_av[Omega_matches])))


structure = SkewSymmetric(nofteams)

scaling = None

k = 2
p = 0.1
mu_start = 1e-1
mu_end = 1e-8
kappa=0.1

grslra_params = {"mu_start": mu_start, "VERBOSE": None, "PRINT": None}
#plot_lpnorm(p, 0.1, mu_end)

for i in xrange(100):
    print i
    L_hat, U_hat, Y_hat, cost  = grslra_test(X_Omega, structure, k, p, mu_end, params=grslra_params, Omega=Omega, dimensions=(nofteams, nofteams), scaling=scaling, kappa=kappa)
    ranking = np.sum(L_hat, axis=1) / nofteams
    S_hat = Omega_matches_full * (X - L_hat)
    rmse = np.linalg.norm(S_hat[Omega_matches])
    sorted_ix = np.argsort(ranking)

    # plot_comparison(L_hat, teamnames, '{:02d}'.format(i) + '_L', '{:f}'.format(cost), clim=(-4, 4))
    # plot_comparison(S_hat, teamnames, '{:02d}'.format(i) + '_S', '{:f}'.format(cost), clim=(-4, 4))
    # plot_ranking(ranking[sorted_ix], teamnames[sorted_ix], '{:02d}'.format(i) + '_teamranking', 'Average predicted ranking according to $\ell_2$-approximation')

    if i:
        L.append(L_hat)
        U.append(U_hat)
        Y.append(Y_hat)
        S.append(S_hat)
        rankings.append(ranking)
        costs.append(cost)
        rmses.append(rmse)
    else:
        L = [L_hat, ]
        U = [U_hat, ]
        Y = [Y_hat, ]
        S = [S_hat, ]
        rankings = [ranking, ]
        costs = [cost, ]
        rmses = [rmse, ]

# sorted_ix_cost = np.argsort(costs)
#
# for i in xrange(10):
#     ix = sorted_ix_cost[i]
#     print ix
#     L_hat, U_hat, Y_hat, cost = grslra_test(X_Omega, structure, k, p, mu_end, params=grslra_params, Omega=Omega, dimensions=(nofteams, nofteams), scaling=scaling, kappa=kappa, U_init=U[ix], Y_init=Y[ix])
#
#     S_hat = Omega_matches_full * (X - L_hat)
#     rmse = np.linalg.norm(S_hat[Omega_matches])
#     plot_comparison(S_hat, teamnames, '{:02d}'.format(ix) + '_new', '{:f}'.format(cost), clim=(-2, 2))
#
#     if i:
#         L_new.append(L_hat)
#         U_new.append(U_hat)
#         Y_new.append(Y_hat)
#         S_new.append(S_hat)
#         rankings_new.append(ranking)
#         costs_new.append(cost)
#         rmses_new.append(rmse)
#     else:
#         L_new = [L_hat, ]
#         U_new = [U_hat, ]
#         Y_new = [Y_hat, ]
#         S_new = [S_hat, ]
#         rankings_new = [ranking, ]
#         costs_new = [cost, ]
#         rmses_new = [rmse, ]


ranking = np.mean(rankings, axis=0)
# cost = np.mean(costs)
sorted_ix = np.argsort(ranking)
# Z= np.outer(ranking, np.ones((nofteams,))) - np.outer(np.ones((nofteams,)), ranking)
#
plot_ranking(ranking[sorted_ix], teamnames[sorted_ix], 'em_ranking_Lp', 'Average predicted ranking according to $\ell_p$-approximation')
# plot_comparison(Z, teamnames, 'em_comparison_L2', 'Goal difference predictions ($\ell_2$ loss), ')
# plot_comparison(Omega_matches_full * (X - Z), teamnames, 'em_comparison_L2_error', 'RMSE Error of $\ell_2$ prediction : ' + '{:1.3f}'.format(np.linalg.norm(X_Omega_matches - Z[Omega_matches])))