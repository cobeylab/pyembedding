#!/usr/bin/env python

import doctest
import unittest
import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import random
import numpy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import subprocess
import tempfile
import shutil
import time
from collections import OrderedDict
from numba import jit
import jsonobject

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

        self.kdtree = None

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

    def find_neighbors_from_embedding(self, neighbor_count, embedding, theiler_window=1, return_indices=False, use_kdtree=True):
        '''
        :param neighbor_count:
        :param embedding:
        :param theiler_window:
        :return:

        >>> a = Embedding([1, 2, 3, 5, 8, 13, 21], delays=(0,2))
        >>> dn, tn = a.find_neighbors_from_embedding(3, a, theiler_window=3)
        >>> tn.tolist()
        [[5, 6, -1], [6, -1, -1], [-1, -1, -1], [2, -1, -1], [3, 2, -1]]

        >>> b = Embedding([1, 2, 1, 2, 1, 2, 1], delays=(0,))
        >>> dn, tn = b.find_neighbors_from_embedding(1, b, theiler_window=1)
        >>> tn[:,0].tolist()
        [2, 3, 0, 1, 0, 1, 0]
        '''
        assert theiler_window >= 0
        return self.find_neighbors(neighbor_count, embedding.embedding_mat, theiler_window=theiler_window, t_query=embedding.t, return_indices=return_indices, use_kdtree=use_kdtree)

    def find_neighbors(self, neighbor_count, query_vectors, theiler_window=0, t_query=None, return_indices=False, use_kdtree=True):
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
        >>> dn2_a, tn2_a = a.find_neighbors(1, [[2, 1]], theiler_window=0, t_query=None)
        >>> '{0:.4f}'.format(dn2_a[0,0])
        '0.0000'
        >>> tn2_a[0,0]
        1
        >>> dn3_a, tn3_a = a.find_neighbors(1, [[2, 1]], theiler_window=1, t_query=[1])
        >>> '{0:.4f}'.format(dn3_a[0,0])
        '1.4142'
        >>> tn3_a[0,0]
        2
        >>> dn4_a, tn4_a = a.find_neighbors(4, [[2,1]], theiler_window=0, t_query=None)
        >>> tn4_a[0,:].tolist()
        [1, 2, 3, -1]
        >>> dn5_a, tn5_a = a.find_neighbors(1, [[2,1]], theiler_window=2, t_query=[1])
        >>> tn5_a[0,:].tolist()
        [3]
        >>> dn6_a, tn6_a = a.find_neighbors(1, [[2,1]], theiler_window=3, t_query=[1])
        >>> tn6_a[0,:].tolist()
        [-1]
        >>> dn6_a[0,0]
        inf
        >>> dn7_a, tn7_a = a.find_neighbors(1, [[2,1]], theiler_window=4, t_query=[1])
        >>> tn7_a[0,:].tolist()
        [-1]
        >>> dn7_a[0,0]
        inf

        >>> b = Embedding([1, 2, 3, 5, 8, 13, 21], delays=(0,2))
        >>> dn1_b, tn1_b = b.find_neighbors(1, [[3, 1], [8, 3], [21, 8]], theiler_window=0, t_query=None)
        >>> dn1_b[:,0].tolist()
        [0.0, 0.0, 0.0]
        >>> tn1_b[:,0].tolist()
        [2, 4, 6]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=1, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[3, 4, 5], [2, 4, 5], [3, 5, 2], [4, 6, 3], [5, 4, 3]]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=2, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[4, 5, 6], [5, 6, -1], [2, 6, -1], [3, 2, -1], [4, 3, 2]]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=3, t_query=[2, 3, 4, 5, 6])
        >>> tn2_b.tolist()
        [[5, 6, -1], [6, -1, -1], [-1, -1, -1], [2, -1, -1], [3, 2, -1]]
        >>> dn2_b, tn2_b = b.find_neighbors(3, [[3, 1], [5, 2], [8, 3], [13, 8], [21, 8]], theiler_window=3, t_query=[2, 3, 4, 5, 6], use_kdtree=False)
        >>> tn2_b.tolist()
        [[5, 6, -1], [6, -1, -1], [-1, -1, -1], [2, -1, -1], [3, 2, -1]]

        >>> rng = numpy.random.RandomState(seed=1)
        >>> x = numpy.random.normal(0, 1, 100)
        >>> c = Embedding(x, delays=(0,))
        >>> dnk_c, tnk_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=True)
        >>> dns_c, tns_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=False)
        >>> numpy.array_equal(dnk_c, dns_c)
        True
        >>> numpy.array_equal(tnk_c, tns_c)
        True
        >>> dnk_c, tnk_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=True)
        >>> dns_c, tns_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=False)
        >>> numpy.array_equal(dnk_c, dns_c)
        True
        >>> numpy.array_equal(tnk_c, tns_c)
        True
        >>> dnk_c, tnk_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=True)
        >>> dns_c, tns_c = c.find_neighbors_from_embedding(1, c, theiler_window=0, use_kdtree=False)
        >>> numpy.array_equal(dnk_c, dns_c)
        True
        >>> numpy.array_equal(tnk_c, tns_c)
        True
        '''
        if not isinstance(query_vectors, numpy.ndarray):
            query_vectors = numpy.array(query_vectors)
        if t_query is not None and not isinstance(t_query, numpy.ndarray):
            t_query = numpy.array(t_query)

        assert theiler_window >= 0
        assert neighbor_count > 0
        assert query_vectors.shape[0] > 0
        assert query_vectors.shape[1] == self.embedding_dimension
        assert theiler_window == 0 or t_query is not None
        assert t_query is None or t_query.shape[0] == query_vectors.shape[0]

        if use_kdtree:
            return self.find_neighbors_kdtree(neighbor_count, query_vectors, theiler_window=theiler_window, t_query=t_query, return_indices=return_indices)
        else:
            return self.find_neighbors_stupid(neighbor_count, query_vectors, theiler_window=theiler_window, t_query=t_query, return_indices=return_indices)

    def find_neighbors_kdtree(self, neighbor_count, query_vectors, theiler_window=0, t_query=None, return_indices=False):
        if self.kdtree is None:
            self.kdtree = cKDTree(self.embedding_mat)

        # Start with infinite distances and missing neighbor times (-1)
        dist = numpy.ones((query_vectors.shape[0], neighbor_count), dtype=float) * float('inf')
        tn = -numpy.ones((query_vectors.shape[0], neighbor_count), dtype=int)
        indn = -numpy.ones((query_vectors.shape[0], neighbor_count), dtype=int)
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
            ind_i[ind_missing] = -1

            tn_i = -numpy.ones((len(unfinished_ind), k), dtype=int)
            tn_i[ind_present] = self.t[ind_i[ind_present]]

            # If there are no times to check against Theiler window, we're done immediately
            if theiler_window == 0:
                dist = dist_i
                tn = tn_i
                indn = ind_i
                break

            # Identify too-close-in-time neighbors; label them -2
            tq_i = t_query[unfinished_ind]
            for ni in range(k):
                too_close = numpy.logical_and(
                    tn_i[:,ni] != -1,
                    numpy.abs(tn_i[:,ni] - tq_i) < theiler_window
                )
                tn_i[too_close,ni] = -2

            # Calculate number of valid neighbors
            n_valid = (tn_i >= 0).sum(axis=1)
            min_n_valid = numpy.min(n_valid)

            has_enough = n_valid >= neighbor_count
            has_too_close = (tn_i == -2).sum(axis=1) > 0
            has_missing = (tn_i == -1).sum(axis=1) > 0

            # Efficiently assign distances for rows that don't need to be modified
            not_too_close_no_missing = numpy.logical_not(numpy.logical_or(has_too_close, has_missing))
            dist[unfinished_ind[not_too_close_no_missing],:] = dist_i[not_too_close_no_missing,:neighbor_count]
            tn[unfinished_ind[not_too_close_no_missing],:] = tn_i[not_too_close_no_missing,:neighbor_count]
            indn[unfinished_ind[not_too_close_no_missing],:] = ind_i[not_too_close_no_missing,:neighbor_count]

            # Assign distances one-by-one for rows that have run out of valid neighbors,
            # or that have enough valid neighbors but also some that are too close in time
            ready_needs_modification = numpy.logical_or(
                has_missing,
                numpy.logical_and(has_too_close, has_enough)
            )
            for ind in numpy.nonzero(ready_needs_modification)[0]:
                dist_ind = dist_i[ind,:]
                tn_ind = tn_i[ind,:]
                indn_ind = ind_i[ind,:]

                valid_ind = tn_ind >= 0
                n_valid_ind = valid_ind.sum()

                dist_ind[:n_valid_ind] = dist_ind[valid_ind]
                tn_ind[:n_valid_ind] = tn_ind[valid_ind]
                indn_ind[:n_valid_ind] = indn_ind[valid_ind]

                dist_ind[n_valid_ind:] = float('inf')
                tn_ind[n_valid_ind:] = -1
                indn_ind[n_valid_ind:] = -1

                dist[unfinished_ind[ind],:] = dist_ind[:neighbor_count]
                tn[unfinished_ind[ind],:] = tn_ind[:neighbor_count]
                indn[unfinished_ind[ind],:] = indn_ind[:neighbor_count]
                # sys.stderr.write('{0}\n'.format(indn_ind[:neighbor_count]))

            not_ready = numpy.logical_and(
                has_too_close,
                numpy.logical_not(numpy.logical_or(has_enough, has_missing))
            )
            assert not_too_close_no_missing.sum() + ready_needs_modification.sum() + not_ready.sum() == len(unfinished_ind)

            unfinished_ind = unfinished_ind[not_ready]
            k = k + neighbor_count - min_n_valid

        if return_indices:
            return dist, tn, indn
        else:
            return dist, tn

    def find_neighbors_stupid(self, neighbor_count, query_vectors, theiler_window=0, t_query=None, return_indices=False):
        dmat = cdist(query_vectors, self.embedding_mat)

        dn = numpy.ones((query_vectors.shape[0], neighbor_count)) * float('inf')
        tn = -numpy.ones((query_vectors.shape[0], neighbor_count), dtype=int)
        indn = -numpy.ones((query_vectors.shape[0], neighbor_count), dtype=int)

        for i in range(dmat.shape[0]):
            dn_i = []
            tn_i = []
            indn_i = []
            for j in numpy.argsort(dmat[i,:]):
                # sys.stderr.write('{0}: dmat[i,j] = {1}, dt = {2}\n'.format(j, dmat[i,j], t_query[i] - self.t[j]))
                if (theiler_window == 0 and t_query is None) or numpy.abs(t_query[i] - self.t[j]) >= theiler_window:
                    indn_i.append(j)
                    tn_i.append(self.t[j])
                    dn_i.append(dmat[i,j])
                if len(tn_i) == neighbor_count:
                    break
            dn[i,:len(dn_i)] = dn_i
            tn[i,:len(tn_i)] = tn_i
            indn[i,:len(tn_i)] = indn_i
        # sys.stderr.write('dn = {0}'.format(dn))
        # sys.stderr.write('tn = {0}'.format(tn))
        if return_indices:
            return dn, tn, indn
        else:
            return dn, tn

    def subembedding(self, delay_sub_indices):
        return Embedding(
            self.x,
            [self.delays[i] for i in delay_sub_indices],
            embedding_mat=self.embedding_mat[:,delay_sub_indices],
            t=self.t
        )

    def identify_nichkawde_subembedding(self, theiler_window):
        # Initial member of embedding is the first column
        sub_inds = (0,)
        sub_emb = self.subembedding(sub_inds)
        while sub_inds[-1] < len(self.delays) - 1:
            dn, tn, indn = sub_emb.find_neighbors_from_embedding(1, sub_emb, theiler_window=theiler_window, return_indices=True)
            valid = indn[:,0] >= 0

            max_deriv = 0.0
            max_deriv_ind = None
            for next_sub_ind in range(sub_inds[-1] + 1, len(self.delays)):
                emb_mat_next = self.embedding_mat[valid, next_sub_ind]
                emb_mat_next_neighbors = emb_mat_next[indn[valid,0]]
                dist_next = numpy.abs(emb_mat_next - emb_mat_next_neighbors)

                deriv = dist_next / dn[valid,0]

                # Take geometric mean over nonzero distances
                geo_mean_deriv = numpy.exp(numpy.log(deriv[deriv != 0.0]).mean())

                if geo_mean_deriv > max_deriv:
                    max_deriv = geo_mean_deriv
                    max_deriv_ind = next_sub_ind

            if max_deriv_ind is None:
                raise Exception('Embedding identification failed: could not calculate max derivative.')
            sub_inds = sub_inds + (max_deriv_ind,)
            sub_emb = self.subembedding(sub_inds)

        return sub_emb

    def ccm(self, query_embedding, y_full, neighbor_count=None, theiler_window=1, use_kdtree=True):

        y_actual, y_pred = self.simplex_predict_using_embedding(
            query_embedding, y_full, neighbor_count=neighbor_count, theiler_window=theiler_window, use_kdtree=use_kdtree
        )
        invalid = numpy.isnan(y_pred)
        valid = numpy.logical_not(invalid)
        valid_count = valid.sum()

        sd_actual = numpy.std(y_actual[valid])
        sd_pred = numpy.std(y_pred[valid])

        if valid_count == 0:
            corr = float('nan')
        elif sd_actual == 0 and sd_pred == 0:
            corr = 1.0
        elif sd_actual == 0 or sd_pred == 0:
            corr = 0.0
        else:
            corr = numpy.corrcoef(y_actual[valid], y_pred[valid])[0,1]

        return jsonobject.JSONObject([
            ('correlation', corr),
            ('valid_count', valid_count),
            ('sd_actual', sd_actual),
            ('sd_predicted', sd_pred)
        ]), y_actual, y_pred


    def simplex_predict_using_embedding(self, query_embedding, y, neighbor_count=None, theiler_window=0, use_kdtree=True):
        return self.simplex_predict(query_embedding.embedding_mat, y, query_embedding.t, neighbor_count=neighbor_count, theiler_window=theiler_window, use_kdtree=use_kdtree)

    def simplex_predict(self, X, y, t, neighbor_count=None, theiler_window=0, use_kdtree=True):
        '''

        :param t:
        :param y_t:
        :param neighbor_count:
        :param query_vectors:
        :param theiler_window:
        :return:

        >>> a = Embedding([1, 2, 1, 2, 1, 2, 1], delays=(0,))
        >>> y = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=2, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=3, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=4, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=10, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=3, theiler_window=0)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=1)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=2)[1].tolist()
        [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=3)[1].tolist()
        [2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=4)[1].tolist()
        [2.0, 1.0, 2.0, nan, 2.0, 1.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=5)[1].tolist()
        [2.0, 2.0, nan, nan, nan, 2.0, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=6)[1].tolist()
        [2.0, nan, nan, nan, nan, nan, 2.0]
        >>> a.simplex_predict(a.embedding_mat, y, a.t, neighbor_count=1, theiler_window=7)[1].tolist()
        [nan, nan, nan, nan, nan, nan, nan]
        '''

        if neighbor_count is None:
            neighbor_count = self.embedding_dimension + 1

        if not isinstance(X, numpy.ndarray):
            X = numpy.array(X)
        if not isinstance(y, numpy.ndarray):
            y = numpy.array(y)
        if not isinstance(t, numpy.ndarray):
            t = numpy.array(t)

        assert X.shape[0] == t.shape[0]
        assert y.shape[0] == self.x.shape[0]
        assert X.shape[1] == self.embedding_dimension

        dn, tn = self.find_neighbors(neighbor_count, X, theiler_window=theiler_window, t_query=t, use_kdtree=use_kdtree)

        assert numpy.isnan(dn).sum() == 0
        invalid = numpy.isinf(dn[:,0])
        valid = numpy.logical_not(invalid)

        # Adjust rows where min distance is 0 so that zero distances are set to 1 and other distances are set to inf
        min_is_zero = dn[:,0] == 0
        dn[min_is_zero,0] = 1
        for i in range(1, neighbor_count):
            is_nonzero_with_zero_min = numpy.logical_and(min_is_zero, dn[:,i] > 0)
            dn[is_nonzero_with_zero_min,i] = float('inf')
            dn[dn[:,i] == 0, i] = 1

        # Calculate weights from distances normalized by nearest neighbor
        y_pred = numpy.zeros(X.shape[0])

        weights = numpy.zeros_like(dn)
        for i in range(neighbor_count):
            weights[valid,i] = numpy.exp(-dn[valid,i] / dn[valid,0])
        weights_sum = numpy.sum(weights, axis=1)
        for i in range(neighbor_count):
            y_pred[valid] += y[tn[valid,i]] * weights[valid,i] / weights_sum[valid]
        y_pred[invalid] = float('nan')

        return y[t], y_pred

    def get_embedding_dimension(self):
        return len(self.delays)

    embedding_dimension = property(get_embedding_dimension)

    def get_delay_vector_count(self):
        return self.embedding_mat.shape[0]

    delay_vector_count = property(get_delay_vector_count)

class TestEmbedding(unittest.TestCase):
    def test_nichkawde_stochastic(self):
        rng = numpy.random.RandomState(seed=256)
        x = numpy.random.normal(0, 1, 100)
        e = Embedding(x, delays=(0,1,2,))
        ne = e.identify_nichkawde_subembedding(5)
        sys.stderr.write('{0}\n'.format(ne.delays))

    def test_find_neighbors_stochastic(self):
        rng = numpy.random.RandomState(seed=1)
        x = numpy.random.normal(0, 1, 100)
        c = Embedding(x, delays=(0,))

        for neighbor_count in range(1, 10):
            for theiler_window in range(10):
                dnk_c, tnk_c, indnk = c.find_neighbors_from_embedding(neighbor_count, c, theiler_window=theiler_window, return_indices=True, use_kdtree=True)
                dns_c, tns_c, indns = c.find_neighbors_from_embedding(neighbor_count, c, theiler_window=theiler_window, return_indices=True, use_kdtree=False)

                # sys.stderr.write('{0}\n'.format(indnk))
                # sys.stderr.write('{0}\n'.format(indns))
                self.assertTrue(numpy.array_equal(dnk_c, dns_c))
                self.assertTrue(numpy.array_equal(tnk_c, tns_c))
                self.assertTrue(numpy.array_equal(indnk, indns))

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    doctest.testmod(verbose=True)
