from scipy.io import loadmat
import numpy as np
import grslra
from grslra.structures import ConcatenatedSkewSymmetric
from grslra.structures import SkewSymmetric
from grslra.scaling import Scaling_Percentile
from grslra.tools import imagesc
from grslra.grslra_batch import grslra_batch
from grslra.visualization import plot_lpnorm
from matplotlib import cm
from matplotlib import pyplot as plt
plt.rc('font', family='sans-serif', size=14)

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
#
# def plot_ranking(ranking, teamnames, resultfile, title):
#     fig, ax = plt.subplots(figsize=(19.2, 10.8))
#     ax.set_title(title)
#     ax1, = plt.plot(xrange(24), ranking)
#     plt.xticks(xrange(24), teamnames, rotation='vertical')
#     fig.subplots_adjust(bottom=0.2)
#     plt.savefig('figures/' + resultfile + '.png')
#     plt.close()


def print_comparison(data, teamnames, filename, title, clim=None, cbar=True):
    fig, ax = plt.subplots(figsize=(8,8))
    cax = plt.imshow(data, interpolation='nearest', cmap=cm.coolwarm, clim=clim)
    #ax.set_title(title)
    plt.xticks(xrange(24), teamnames, rotation='vertical')
    plt.yticks(xrange(24), teamnames)
    if cbar:
        fig.colorbar(cax,fraction=0.03)
    fig.subplots_adjust(left=0.2, bottom=0.2, top=0.95, right=0.95)
    plt.savefig('/home/cleha/data/Dropbox/work/euro2016/figures/' + filename + '.pdf')
    plt.close()

def print_ranking(ranking, teamnames, filename, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    #ax.set_title(title)
    ax1, = plt.plot(xrange(24), ranking)
    plt.xticks(xrange(24), teamnames, rotation='vertical')
    fig.subplots_adjust(bottom = 0.2, top=0.95)
    plt.savefig('/home/cleha/data/Dropbox/work/euro2016/figures/' + filename + '.pdf')
    plt.close()


def print_table(stats, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    colLabels = ("Team", "Goal difference", "Points")
    ax.axis('off')
    cellText=[]
    for i in xrange(len(stats)):
        cellText.append(['{:2d}'.format(i+1), stats[i][0], '{:+2d}'.format(stats[i][1]), '{:2d}'.format(stats[i][2])])

    points = np.array(stats["points"], dtype=np.float)
    points -= points.min()
    points /= points.max()
    points *= 0.5
    points += 0.25

    cvector = points

    # do the table
    cellColours = cm.coolwarm(np.tile(np.atleast_2d(cvector).T, 4))
    colWidths = [0.1, 0.3,0.1,0.1]
    table = ax.table(cellText=cellText, loc='center', colWidths=colWidths, cellColours = cellColours)#, colLabels=colLabels, rowColours=rowColours, colWidths=colWidths)
    table.scale(1.75,1.75)
    #table.set_fontsize(18)
    #fig.subplots_adjust(left=0.2, bottom = 0.2, top=0.8, right=0.8)
    plt.savefig('/home/cleha/data/Dropbox/work/euro2016/figures/' + filename + '.pdf')


def get_stats(X):
    dtype = [('name', 'a15'), ('goaldiff', int), ('points', int)]
    for i in xrange(nofteams):
        goaldiff = 0
        points = 0
        for j in xrange(nofteams):
            if i==j:
                continue
            diff = np.round(X[i, j])
            goaldiff += diff
            if diff > 0:
                points += 3
            elif diff == 0:
                points += 1
            else:
                pass
        entry = (teamnames[i], goaldiff, points)

        if i:
            stats.append(entry)
        else:
            stats = [entry, ]
    arr = np.array(stats, dtype=dtype)
    arr_sorted = np.sort(arr, order=['points', 'goaldiff'])[::-1]
    return arr_sorted


teamnames= np.array(['Albania', 'Austria', 'Belgium', 'Croatia', 'Czech Republic', 'England', 'France', 'Germany', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'North. Ireland', 'Poland', 'Portugal', 'Romania', 'Russia', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'Wales'])

data = loadmat('../data/em.mat')
X = data["X"]
nofteams = X.shape[0]

# X[X>1] = 1
# X[X<-1] = -1

Omega = (data['row'].flatten() - 1, data['col'].flatten() - 1)
Omega_full = np.zeros((nofteams, nofteams))
Omega_full[Omega] = 1
Omega_matches_full = np.triu(Omega_full,1)
Omega_matches = np.where(Omega_matches_full)
nofmatches = Omega_matches[0].size

k = 2

print_comparison(data["X"], teamnames, 'em_results', 'Actual goal differences of EURO 2016 matches')
print_comparison(Omega_full, teamnames, 'em_matches', 'Observed matches of EURO 2016', clim=(-1, 1), cbar=False)

ranking_average = np.sum(data['X'], axis=1) / (np.sum(Omega_full, axis=1) - 1)
sorted_ix_average = np.argsort(ranking_average)
print_ranking(ranking_average[sorted_ix_average], teamnames[sorted_ix_average], 'em_ranking_avgoals', 'Naive ranking according to average goal difference per match')

L_av= np.outer(ranking_average, np.ones((nofteams,))) - np.outer(np.ones((nofteams,)), ranking_average)
print_comparison(L_av, teamnames, 'em_comparison_avgoals', 'Goal difference predictions based on average goal difference per match', clim=(-4, 4))
print_comparison(Omega_matches_full * (X - L_av), teamnames, 'em_comparison_avgoals_error', 'RMSE misfit of estimation based on average goal difference per match: ' + '{:1.3f}'.format(np.linalg.norm(X[Omega_matches] - L_av[Omega_matches])), clim=(-4, 4))
print_table(get_stats(L_av), 'em_table_avgoals')


data=loadmat('rankfit.mat')
sorted_ix = np.argsort(data["ranking"].flatten())
print_comparison(data["L"], teamnames, 'em_comparison_rankfit', 'Predicted goal differences according to fitted ranking', clim=(-4, 4))
print_ranking(data["ranking"].flatten()[sorted_ix], teamnames[sorted_ix], 'em_ranking_rankfit', 'Predicted ranking according to fitted ranking')
print_comparison(Omega_matches_full * (X - data["L"]), teamnames, 'em_comparison_rankfit_error', 'RMSE misfit of fitted ranking:' + '{:1.3f}'.format(np.linalg.norm(X[Omega_matches] - data["L"][Omega_matches])), clim=(-4, 4))
print_table(get_stats(data["L"]), 'em_table_rankfit')

U, _, _ = np.linalg.svd(X)
L_PCA = np.dot(U[:,:k], np.dot(U[:,:k].T, X))

print_comparison(L_PCA, teamnames, 'em_comparison_pca', 'Predicted goal differences according to PCA', clim=(-4, 4))
print_table(get_stats(L_PCA), 'em_table_pca')

structure = SkewSymmetric(nofteams)
p = 2.0
mu_start = 1e-8
mu_end = 1e-8
kappa=None
grslra_params = {"mu_start": mu_start, "VERBOSE": 1, "PRINT": 0}
#TODO: scaling
scaling = None

L_hat, U_hat, Y_hat = grslra_batch(X[Omega], structure, k, p, mu_end, params=grslra_params,
                                        Omega=Omega, dimensions=(nofteams, nofteams), scaling=scaling,
                                        kappa=kappa, PCA_INIT=True)

ranking = np.sum(L_hat, axis=1) / nofteams
S_hat = Omega_matches_full * (X - L_hat)
sorted_ix = np.argsort(ranking)

print_comparison(L_hat, teamnames, 'em_comparison_prediction_2_nokappa', 'Predicted goal differences according to $\ell_2$ measure', clim=(-4, 4))
print_ranking(ranking[sorted_ix], teamnames[sorted_ix], 'em_ranking_prediction_2_nokappa', 'Predicted ranking according to $\ell_2$ measure')
print_comparison(Omega_matches_full * (X - L_hat), teamnames, 'em_comparison_prediction_error_2_nokappa', 'RMSE misfit of $\ell_2$ prediction:' + '{:1.3f}'.format(np.linalg.norm(X[Omega_matches] - L_hat[Omega_matches])), clim=(-4, 4))
print_table(get_stats(L_hat), 'em_table_prediction_2_nokappa')


err = np.zeros((nofmatches,))
predictions = np.zeros((nofmatches,))

for i in xrange(nofmatches):
    Omega_test = Omega_matches[0][i], Omega_matches[1][i]
    Omega_train_full = Omega_matches_full.copy()
    Omega_train_full[Omega_test] = 0
    Omega_train = np.where(Omega_train_full + Omega_train_full.T + np.eye(nofteams))

    L_hat, _, _ = grslra_batch(X[Omega_train], structure, k, p, mu_end, params=grslra_params,
                                            Omega=Omega_train, dimensions=(nofteams, nofteams), scaling=scaling,
                                            kappa=kappa, PCA_INIT=True)

    predictions[i] = L_hat[Omega_test]
    err[i] = X[Omega_test] - predictions[i]

    title = teamnames[Omega_test[0]], teamnames[Omega_test[1]], X[Omega_test], predictions[i]#, predictions_PCA[i]
    plot_comparison(L_hat, teamnames, 'predictions/' + '{:02d}'.format(i), title, clim=(-4, 4))

    tmp=1

np.savez('predictions', err=err, predictions=predictions)

print "RMSE prediction error of rankfit: ", np.linalg.norm(data["err"].flatten())
print "RMSE prediction error of L1 approximation: ", np.linalg.norm(err)