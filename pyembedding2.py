#!/usr/bin/env python

import os
import sys
import random
import numpy
from numba import jit

def make_embedding(x, E, tau=1, verify_all=False):
    """Create a matrix of lagged vectors of size E from vector x.
    
    The lagged vectors of length E form the rows of the matrix.
    
    >>> x = numpy.array([3.0, 1.7, 4.3, 5.4, 8.8, 9.6])
    >>> X = make_embedding(x, 3, verify_all=True)
    >>> X.shape == (4,3)
    True
    >>> numpy.sum(X[0,:] == [3.0, 1.7, 4.3])
    3
    >>> numpy.sum(X[1,:] == [1.7, 4.3, 5.4])
    3
    >>> numpy.sum(X[2,:] == [4.3, 5.4, 8.8])
    3
    >>> numpy.sum(X[3,:] == [5.4, 8.8, 9.6])
    3
    >>> X2 = make_embedding(x, 3, tau=2, verify_all=True)
    >>> X2.shape[0]
    2
    >>> X2.shape[1]
    3
    >>> numpy.sum(X2[0,:] == [3.0, 4.3, 8.8])
    3
    >>> numpy.sum(X2[1,:] == [1.7, 5.4, 9.6])
    3
    """
    # Preconditions
    assert E > 0
    assert tau > 0
    assert len(x.shape) == 1
    N = x.shape[0]
    vec_span = tau * (E - 1) + 1
    assert N >= vec_span
    
    vec_count = N - tau * (E - 1)
    X = numpy.zeros((vec_count, E))
    for i in range(E):
        start = tau * i
        X[:,i] = x[start:(start + vec_count)]
    
    # Postcondition: boundary checks
    assert numpy.sum(X[0,:] == x[0:tau*E:tau]) == E
    if verify_all:
        for i in range(vec_count):
            assert numpy.sum(X[i,:] == x[i:i + tau*E:tau]) == E
    assert numpy.sum(X[-1,:] == x[-(1 + tau*(E-1))::tau]) == E
    
    return X

def squared_euclidean_distance(A, B):
    """Compute the squared Euclidean distance between all rows of A and all rows of B.
    
    >>> x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> X = make_embedding(x, 3, verify_all=True)
    >>> D = squared_euclidean_distance(X, X)
    >>> numpy.array_equal(D, [[0, 3, 12, 27], [3, 0, 3, 12], [12, 3, 0, 3], [27, 12, 3, 0]])
    True
    """
    D = numpy.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        D[i,:] = ((B - A[i:(i+1),:].repeat(B.shape[0], axis=0))**2).sum(axis=1)
    return D

def euclidean_distance(A, B):
    return numpy.sqrt(squared_euclidean_distance(A, B))

@jit
def assign_diagonal(X, offset, val):
    """Initialize a value to all elements of a diagonal of X
    >>> x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> X = make_embedding(x, 3, verify_all=True)
    >>> D = squared_euclidean_distance(X, X)
    >>> inf_val = float('inf')
    >>> assign_diagonal(D, 0, inf_val)
    >>> numpy.array_equal(D, [[inf_val, 3, 12, 27], [3, inf_val, 3, 12], [12, 3, inf_val, 3], [27, 12, 3, inf_val]])
    True
    >>> assign_diagonal(D, 1, inf_val)
    >>> numpy.array_equal(D, [[inf_val, inf_val, 12, 27], [3, inf_val, inf_val, 12], [12, 3, inf_val, inf_val], [27, 12, 3, inf_val]])
    True
    >>> assign_diagonal(D, -1, inf_val)
    >>> numpy.array_equal(D, [[inf_val, inf_val, 12, 27], [inf_val, inf_val, inf_val, 12], [12, inf_val, inf_val, inf_val], [27, 12, inf_val, inf_val]])
    True
    """
    for i in range(X.shape[0]):
        if (i + offset) >= 0 and (i + offset) < X.shape[1]:
            X[i, i + offset] = val
    
def find_neighbors(D, n_neighbors):
    """Find nearest neighbors from a distance matrix.
    >>> x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> X = make_embedding(x, 3, verify_all=True)
    >>> D = squared_euclidean_distance(X, X)
    >>> assign_diagonal(D, 0, float('inf'))
    >>> N, DN = find_neighbors(D, 1)
    >>> numpy.array_equal(N, [[1], [0], [1], [2]])
    True
    >>> assign_diagonal(D, 1, float('inf'))
    >>> assign_diagonal(D, -1, float('inf'))
    >>> N, DN = find_neighbors(D, 1)
    >>> numpy.array_equal(N, [[2], [3], [0], [1]])
    True
    """
    N = numpy.zeros((D.shape[0], n_neighbors), dtype=int)
    DN = numpy.zeros((D.shape[0], n_neighbors), dtype=float)
    for i in range(D.shape[0]):
        N[i,:] = numpy.argpartition(D[:,i], range(n_neighbors))[:n_neighbors]
        DN[i,:] = D[i, N[i,:]]
    return N, DN

def nichkawde_embedding(x, E_max, neighbor_offset_min):
    assert len(x.shape) == 1
    
    # Make a full embedding with maximum embedding dimension E_max;
    # actual embedding will consist of columns of this full embedding.
    X_full = make_embedding(x, E_max, tau=1)[:,::-1]
    
    # Initial member of embedding is simply x[t] = x[t - 0]
    taus = (0,)
    while taus[-1] < (E_max - 1):
        X = X_full[:,taus]
        D = numpy.sqrt(squared_euclidean_distance(X, X))
        for offset in range(neighbor_offset_min):
            assign_diagonal(D, offset, float('inf'))
            if offset > 0:
                assign_diagonal(D, -offset, float('inf'))
        
        N, DN = find_neighbors(D, 1)
        
        max_deriv = 0.0
        max_deriv_tau = None
        for tau_next in range(taus[-1] + 1, E_max):
            Xnext = X_full[:,tau_next]
            Xnext_N = Xnext[N[:,0]]
            Dnext = numpy.abs(Xnext - Xnext_N)
            deriv = Dnext / DN[:,0]
            geo_mean_deriv = numpy.exp(numpy.log(deriv).mean())
            
            if geo_mean_deriv > max_deriv:
                max_deriv = geo_mean_deriv
                max_deriv_tau = tau_next
        
        assert max_deriv_tau is not None
        taus = taus + (max_deriv_tau,)
        
        print taus

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    
    import doctest
    doctest.testmod(verbose=True)
    
    x = numpy.random.normal(size=1000)
    X = nichkawde_embedding(x, 5, 2)
