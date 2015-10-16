#!/usr/bin/env python

import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import random
import numpy
from scipy.spatial import cKDTree
import subprocess
import tempfile
import shutil
import time
from collections import OrderedDict
from numba import jit

class Embedding:
    '''

    >>> a = Embedding([1, 2, 3, 4], delays=(0, 1))
    >>> a.embedding_mat.tolist()
    [[2, 1], [3, 2], [4, 3]]
    >>> a.t.tolist()
    [1, 2, 3]
    >>> b = Embedding([1, 2, 3, 4], delays=(0, 2))
    >>> b.embedding_mat.tolist()
    [[3, 1], [4, 2]]
    >>> b.t.tolist()
    [2, 3]
    >>> c = Embedding([1, 2, 3, 4], delays=(-1, 1))
    >>> c.embedding_mat.tolist()
    [[3, 1], [4, 2]]
    >>> c.t.tolist()
    [1, 2]
    >>> d = Embedding([1, 2, 3, 4], delays=(1, -1))
    >>> d.embedding_mat.tolist()
    [[1, 3], [2, 4]]
    >>> d.t.tolist()
    [1, 2]
    >>> e = Embedding([1, 2, 3, 4], delays=range(4))
    >>> e.embedding_mat.tolist()
    [[4, 3, 2, 1]]
    >>> e.t.tolist()
    [3]
    >>> try:
    ...     f = Embedding([1, 2, 3, 4], delays=range(5))
    ...     assert False
    ... except AssertionError:
    ...     pass
    '''
    def __init__(self, x, delays, embedding_mat=None, t=None):
        self.x = numpy.array(x)
        self.delays = tuple(delays)

        if embedding_mat is None:
            assert t is None
            self.construct_embedding_matrix()
        else:
            self.embedding_mat = embedding_mat
            self.t = t
        if self.embedding_mat.shape[0] == 0:
            self.kdtree = None
        else:
            self.kdtree = cKDTree(self.embedding_mat)

    def construct_embedding_matrix(self):
        min_delay = numpy.min(self.delays)
        max_delay = numpy.max(self.delays)

        t_list = []
        embedding_list = []
        for i in range(self.x.shape[0]):
            if (i - max_delay < 0) or (i - min_delay) >= self.x.shape[0]:
                continue
            delay_vector = numpy.array([self.x[i - delay] for delay in self.delays])
            if numpy.any(numpy.logical_or(numpy.isnan(delay_vector), numpy.isinf(delay_vector))):
                sys.stderr.write('Warning: found nan or inf at index {0}; leaving out'.format(i))
                continue
            t_list.append(i)
            embedding_list.append(delay_vector)
        assert len(embedding_list) > 0

        self.t = numpy.array(t_list)
        self.embedding_mat = numpy.array(embedding_list)
        assert self.embedding_mat.shape[1] == len(self.delays)

    def sampled_embedding(self, n, replace=True, rng=numpy.random):
        '''
        >>> a = Embedding([1, 2, 3, 4], delays=(0, 1))
        >>> b = a.sampled_embedding(2, replace=False)
        >>> b.delay_vector_count
        2
        >>> c = a.sampled_embedding(2, replace=True)
        >>> c.delay_vector_count
        2
        >>> d = a.sampled_embedding(3, replace=True)
        >>> d.delay_vector_count
        3
        >>> try:
        ...     e = a.sampled_embedding(3, replace=False)
        ...     assert False
        ... except AssertionError:
        ...     pass
        >>> f = a.sampled_embedding(4, replace=True)
        >>> f.delay_vector_count
        4
        '''
        assert n > 0
        if replace == False:
            assert n < self.embedding_mat.shape[0]

        inds = rng.choice(self.embedding_mat.shape[0], size=n, replace=replace)

        return Embedding(self.x, self.delays, embedding_mat=self.embedding_mat[inds,:], t=self.t[inds])

    def find_neighbors(self, neighbor_count, query_vectors, theiler_window=1, t_query=None):
        '''

        :param neighbor_count:
        :param query_vectors:
        :param theiler_window:
        :param t_query:
        :return:

        >>> a = Embedding([1, 2, 3, 4], delays=(0, 1))
        >>> dn1_a, tn1_a = a.find_neighbors(1, [[1.9, 1.1]])
        >>> dn1_a.shape
        (1, 1)
        >>> '{0:.4f}'.format(dn1_a[0,0])
        '0.1414'
        >>> tn1_a.shape
        (1, 1)
        >>> tn1_a[0,0]
        1
        >>> dn2_a, tn2_a = a.find_neighbors(1, [[2, 1]], theiler_window=None, t_query=None)
        >>> '{0:.4f}'.format(dn2_a[0,0])
        '0.0000'
        >>> tn2_a[0,0]
        1
        >>> dn3_a, tn3_a = a.find_neighbors(1, [[2, 1]], theiler_window=0, t_query=[1])
        >>> '{0:.4f}'.format(dn3_a[0,0])
        '1.4142'
        >>> tn3_a[0,0]
        2
        >>> dn4_a, tn4_a = a.find_neighbors(4, [[2,1]], theiler_window=None, t_query=None)
        >>> tn4_a[0,:].tolist()
        [1, 2, 3, -1]
        >>> dn5_a, tn5_a = a.find_neighbors(1, [[2,1]], theiler_window=1, t_query=[1])
        >>> tn5_a[0,:].tolist()
        [3]
        >>> dn6_a, tn6_a = a.find_neighbors(1, [[2,1]], theiler_window=2, t_query=[1])
        >>> tn6_a[0,:].tolist()
        [-1]
        >>> dn6_a[0,0]
        inf
        >>> dn7_a, tn7_a = a.find_neighbors(1, [[2,1]], theiler_window=3, t_query=[1])
        >>> tn7_a[0,:].tolist()
        [-1]
        >>> dn7_a[0,0]
        inf

        >>> b = Embedding([1, 2, 3, 5, 8, 13, 21], delays=(0,2))
        >>> dn1_b, tn1_b = b.find_neighbors(1, [[3, 1], [8, 3], [21, 8]], theiler_window=None, t_query=None)
        >>> dn1_b[:,0].tolist()
        [0.0, 0.0, 0.0]
        >>> tn1_b[:,0].tolist()
        [2, 4, 6]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=0, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[3, 4, 5], [2, 4, 5], [3, 5, 2], [4, 6, 3], [5, 4, 3]]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=1, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[4, 5, 6], [5, 6, -1], [2, 6, -1], [3, 2, -1], [4, 3, 2]]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=2, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[5, 6, -1], [6, -1, -1], [-1, -1, -1], [2, -1, -1], [3, 2, -1]]
        '''
        if not isinstance(query_vectors, numpy.ndarray):
            query_vectors = numpy.array(query_vectors)
        if t_query is not None and not isinstance(t_query, numpy.ndarray):
            t_query = numpy.array(t_query)

        assert neighbor_count > 0
        assert query_vectors.shape[0] > 0
        assert query_vectors.shape[1] == self.embedding_dimension
        assert (t_query is None and theiler_window is None) or theiler_window >= 0
        assert t_query is None or t_query.shape[0] == query_vectors.shape[0]

        # Start with infinite distances and missing neighbor times (-1)
        dist = numpy.ones((query_vectors.shape[0], neighbor_count), dtype=float) * float('inf')
        tn = -numpy.ones((query_vectors.shape[0], neighbor_count), dtype=int)
        unfinished_ind = numpy.arange(query_vectors.shape[0])

        # Query kd-tree until every point has neighbor_count neighbors outside the Theiler window,
        # or, if not enough neighbors exist, the maximum available number outside the Theiler window.
        k = neighbor_count
        while len(unfinished_ind) > 0:
            # Query points that need more neighbors
            dist_i, ind_i = self.kdtree.query(query_vectors[unfinished_ind,:], k=k)

            # Fix output of query function (returns 1D array if k==1)
            if len(ind_i.shape) == 1:
                assert len(dist_i.shape) == 1
                dist_i = dist_i.reshape((ind_i.shape[0], 1))
                ind_i = ind_i.reshape((ind_i.shape[0], 1))

            # Unavailable neighbors indicated by kdtree.n
            is_missing = ind_i == self.kdtree.n
            ind_missing = numpy.nonzero(is_missing)
            ind_present = numpy.nonzero(numpy.logical_not(is_missing))

            tn_i = -numpy.ones((len(unfinished_ind), k), dtype=int)
            tn_i[ind_present] = self.t[ind_i[ind_present]]

            # If there are no times to check against Theiler window, we're done immediately
            if t_query is None:
                dist = dist_i
                tn = tn_i
                break

            # Identify too-close-in-time neighbors; label them -2
            tq_i = t_query[unfinished_ind]
            for ni in range(k):
                too_close = numpy.logical_and(
                    tn_i[:,ni] != -1,
                    numpy.abs(tn_i[:,ni] - tq_i) <= theiler_window
                )
                tn_i[too_close,ni] = -2

            # dist = dist_i
            # tn = tn_i
            # break

            # Calculate number of valid neighbors
            n_valid = (tn_i >= 0).sum(axis=1)
            min_n_valid = numpy.min(n_valid)

            has_enough = n_valid >= neighbor_count
            has_too_close = (tn_i == -2).sum(axis=1) > 0
            has_missing = (tn_i == -1).sum(axis=1) > 0

            # Efficiently assign distances for rows that don't need to be modified
            not_too_close_no_missing = numpy.logical_not(numpy.logical_or(has_too_close, has_missing))
            dist[unfinished_ind[not_too_close_no_missing],:] = dist_i[not_too_close_no_missing,:neighbor_count]

            # Assign distances one-by-one for rows that have run out of valid neighbors,
            # or that have enough valid neighbors but also some that are too close in time
            ready_needs_modification = numpy.logical_or(
                has_missing,
                numpy.logical_and(has_too_close, has_enough)
            )
            for ind in numpy.nonzero(ready_needs_modification)[0]:
                dist_ind = dist_i[ind,:]
                tn_ind = tn_i[ind,:]
                valid_ind = tn_ind >= 0
                n_valid_ind = valid_ind.sum()

                dist_ind[:n_valid_ind] = dist_ind[valid_ind]
                tn_ind[:n_valid_ind] = tn_ind[valid_ind]

                dist_ind[n_valid_ind:] = float('inf')
                tn_ind[n_valid_ind:] = -1

                dist[unfinished_ind[ind],:] = dist_ind[:neighbor_count]
                tn[unfinished_ind[ind],:] = tn_ind[:neighbor_count]

            not_ready = numpy.logical_and(
                has_too_close,
                numpy.logical_not(numpy.logical_or(has_enough, has_missing))
            )
            assert not_too_close_no_missing.sum() + ready_needs_modification.sum() + not_ready.sum() == len(unfinished_ind)

            unfinished_ind = unfinished_ind[not_ready]
            k = k + neighbor_count - min_n_valid


        return dist, tn

    def get_embedding_dimension(self):
        return len(self.delays)

    embedding_dimension = property(get_embedding_dimension)

    def get_delay_vector_count(self):
        return self.embedding_mat.shape[0]

    delay_vector_count = property(get_delay_vector_count)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    import doctest
    doctest.testmod(verbose=True)
