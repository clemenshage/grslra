import numpy as np
from ._lpnorm import lpnorm, lpnormgrad


def lpnormgrad_c_openmp(mat, mu, p):
    out = np.zeros(mat.shape)
    lpnormgrad(mat, mu, p, True, out)
    return out


def lpnormgrad_py_simple(mat, mu, p):

    return p * mat * np.exp(np.log(mat * mat + mu) * (p / 2. - 1))

def lpnorm_py_simple(mat, mu, p, temp=None):
    ''' Straightforward lpnorm implementation. '''
    return np.sum(np.exp(np.log(np.multiply(mat, mat) + mu) * (p / 2.)))


def lpnorm_c_openmp(mat, mu, p, temp=None):
    ''' Proxy function with static configuration '''
    return lpnorm(mat, mu, p, False, True)


def lpnorm_py_inplace(mat, mu, p, temp):
    ''' Lpnorm implmementation that does not allocate new memory blocks.

        This variant uses the temp matrix for storing intermediate results.
    '''
    np.multiply(mat, mat, temp)
    np.log(temp + mu, temp)
    np.exp(temp * p / 2., temp)
    return np.sum(temp)


def lpnorm_c(mat, mu, p, temp=None):
    ''' Proxy function with static configuration '''

    return lpnorm(mat, mu, p, False, False)


def lpnorm_c_sse(mat, mu, p, temp=None):
    ''' Proxy function with static configuration '''

    return lpnorm(mat, mu, p, True, False)





def lpnorm_c_sse_openmp(mat, mu, p, temp=None):
    ''' Proxy function with static configuration '''

    return lpnorm(mat, mu, p, True, True)

__all__ = [
    'lpnorm',
    'lpnorm_py_simple',
    'lpnorm_py_inplace',
    'lpnorm_c',
    'lpnorm_c_sse',
    'lpnorm_c_openmp',
    'lpnorm_c_sse_openmp',
]
