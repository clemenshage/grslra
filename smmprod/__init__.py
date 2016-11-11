# # -*- coding: utf-8 -*-
import numpy as np
from . import _smmprod


def smmprod_c(A, B, Omega):
    # out wird hier preallocated, kann in Schleifen dann wiederverwendet werden
    out = np.zeros(Omega[0].shape[0])
    _smmprod.smmprod(A, B, Omega, out)
    return out

# def smmprod(A, B, Omega):
#     A_rows = A[Omega[0]]
#     B_cols = B.T[Omega[1]]
#     return np.sum(A_rows * B_cols, axis=1)
#
#
# def smmprod2(A, B, Omega):
#     A_rows = A[Omega[0]]
#     B_cols = B.T[Omega[1]]
#     # Inplace Multiplikation nach A_rows, damit fällt Speicher Allokation weg
#     np.multiply(A_rows, B_cols, A_rows)
#     return np.sum(A_rows, axis=1)
#
#
# def smmprod3(A, B, Omega):
#     # out wird hier preallocated, kann in Schleifen dann wiederverwendet werden
#     out = np.zeros(Omega.shape[1])
#     _smmprod.smmprod(A, B, Omega, out)
#     return out
#
#
# def smmprod_loop(A, B, Omega):
#     card_Omega = np.size(Omega[0])
#     result = np.zeros(card_Omega)
#     for k in range(card_Omega):
#         result[k] = np.dot(A[Omega[0][k]], B.T[Omega[1][k]])
#     return result
#
#
# def smmprod_loop2(A, B, Omega):
#     card_Omega = np.size(Omega[0])
#     result = np.zeros(card_Omega)
#     # B nur einmal transponieren
#     B = B.T
#     # über Omega.T iterieren, günstigere Index-Extraction
#     for index, idx in enumerate(Omega.T):
#         result[index] = np.dot(A[idx[0]], B[idx[1]])
#     return result
