#!/usr/bin/env python

import doctest
import unittest
import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import numpy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from collections import OrderedDict

class ProjectionEmbedding:
    '''

    >>> a = ProjectionEmbedding([1, 2, 3, 4], 2, 2, projection_mat=[[1, 0], [0, 1]])
    >>> a.embedding_mat.tolist()
    [[2, 1], [3, 2], [4, 3]]
    >>> a = ProjectionEmbedding([1, 2, 3, 4], 2, 2, projection_mat=[[1, 1], [0, 1]])
    >>> a.embedding_mat.tolist()
    [[3, 1], [5, 2], [7, 3]]
    >>> a = ProjectionEmbedding([1, 2, 3, 4], 3, 2, projection_mat=[[1, 1, 0], [0, 1, 1]])
    >>> a.embedding_mat.tolist()
    [[5, 3], [7, 5]]
    '''
    def __init__(self, x, dmax, d, embedding_mat=None, t=None, projection_mat=None, rng=numpy.random):
        self.x = numpy.array(x)
        self.dmax = dmax
        self.d = d
        assert d <= dmax

        if projection_mat is None:
            projection_mat = rng.normal(0.0, 1.0, size=(d, dmax))
        elif not isinstance(projection_mat, numpy.ndarray):
            projection_mat = numpy.array(projection_mat)

        assert projection_mat.shape[0] == d
        assert projection_mat.shape[1] == dmax
        self.projection_mat = projection_mat

        if embedding_mat is None:
            assert t is None
            self.construct_embedding_matrix()
        else:
            self.embedding_mat = embedding_mat
            self.t = t

        self.kdtree = None

    def construct_embedding_matrix(self):
        max_delay = self.dmax - 1

        t_list = []
        embedding_list = []
        for i in range(self.x.shape[0]):
            if (i - max_delay < 0):
                continue
            delay_vector_unprojected = self.x[(i - max_delay) : (i + 1)][::-1]
            assert delay_vector_unprojected.shape[0] == self.dmax

            if numpy.any(numpy.logical_or(numpy.isnan(delay_vector_unprojected), numpy.isinf(delay_vector_unprojected))):
                continue

            delay_vector = numpy.dot(self.projection_mat, delay_vector_unprojected)
            assert delay_vector.shape[0] == self.d

            t_list.append(i)
            embedding_list.append(delay_vector)

        if len(embedding_list) == 0:
            self.t = numpy.array(t_list, dtype=float)
            self.embedding_mat = numpy.zeros((0, self.d), dtype=float)
        else:
            self.t = numpy.array(t_list)
            self.embedding_mat = numpy.array(embedding_list)
            assert self.embedding_mat.shape[1] == self.d

    def sample_embedding(self, n, match_valid_vec=None, replace=True, rng=numpy.random):
        '''
        >>> a = ProjectionEmbedding([1, 2, 3, 4], 2, 2)
        >>> b = a.sample_embedding(2, replace=False)
        >>> b.delay_vector_count
        2
        >>> c = a.sample_embedding(2, replace=True)
        >>> c.delay_vector_count
        2
        >>> d = a.sample_embedding(3, replace=True)
        >>> d.delay_vector_count
        3
        >>> try:
        ...     e = a.sample_embedding(3, replace=False)
        ...     assert False
        ... except AssertionError:
        ...     pass
        >>> f = a.sample_embedding(4, replace=True)
        >>> f.delay_vector_count
        4
        '''
        assert n > 0
        if replace == False:
            assert n < self.embedding_mat.shape[0]
        
        if match_valid_vec is not None:
            valid_ind_mask = numpy.logical_not(numpy.isnan(match_valid_vec))[self.t]
            valid_inds = numpy.arange(self.embedding_mat.shape[0])[valid_ind_mask]
            
            if valid_inds.shape[0] == 0:
                return None
            
            inds = rng.choice(valid_inds, size=n, replace=replace)
        else:
            inds = rng.choice(self.embedding_mat.shape[0], size=n, replace=replace)

        return ProjectionEmbedding(self.x, self.dmax, self.d, embedding_mat=self.embedding_mat[inds,:], t=self.t[inds], projection_mat = self.projection_mat)

    def find_neighbors_from_embedding(self, neighbor_count, embedding, theiler_window=0, return_indices=False, use_kdtree=True):
        '''
        :param neighbor_count:
        :param embedding:
        :param theiler_window:
        :return:

        >>> a = ProjectionEmbedding([1, 2, 3, 5, 8, 13, 21], dmax=3, d=2, projection_mat=[[1, 0, 0], [0, 0, 1]])
        >>> dn, tn = a.find_neighbors_from_embedding(3, a, theiler_window=3)
        >>> tn.tolist()
        [[5, 6, -1], [6, -1, -1], [-1, -1, -1], [2, -1, -1], [3, 2, -1]]

        >>> b = ProjectionEmbedding([1, 2, 1, 2, 1, 2, 1], dmax=1, d=1, projection_mat=[[1]])
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

        >>> a = ProjectionEmbedding([1, 2, 3, 4], dmax=2, d=2, projection_mat=[[1, 0], [0, 1]])
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

        >>> b = ProjectionEmbedding([1, 2, 3, 5, 8, 13, 21], dmax=3, d=2, projection_mat=[[1, 0, 0], [0, 0, 1]])
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
        >>> c = ProjectionEmbedding(x, 1, 1, projection_mat=[[1]])
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

    def ccm(self, query_embedding, y_full, neighbor_count=None, theiler_window=1, use_kdtree=True):
        return self.simplex_predict_summary(query_embedding, y_full, neighbor_count=neighbor_count, theiler_window=theiler_window, use_kdtree=use_kdtree)
    
    def simplex_predict_summary(self, query_embedding, y_full, neighbor_count=None, theiler_window=1, use_kdtree=True):
        y_actual, y_pred = self.simplex_predict_using_embedding(
            query_embedding, y_full, neighbor_count=neighbor_count, theiler_window=theiler_window, use_kdtree=use_kdtree
        )   
        corr, valid_count, sd_actual, sd_pred = correlation_valid(y_actual, y_pred)
        
        return OrderedDict([
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

        >>> a = ProjectionEmbedding([1, 2, 1, 2, 1, 2, 1], 1, 1, projection_mat=[[1]])
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
        return self.d

    embedding_dimension = property(get_embedding_dimension)

    def get_delay_vector_count(self):
        return self.embedding_mat.shape[0]

    delay_vector_count = property(get_delay_vector_count)


def tajima_cross_embedding(cause, effect, theiler_window, neighbor_count = None, corr_threshold = 1.00, rng = numpy.random):
    '''
    >>> x = numpy.random.normal(0, 1, size=1000)
    >>> y = numpy.random.normal(0, 1, size=1000)
    >>> emb_xy = tajima_cross_embedding(x, y, 10, 20)
    >>> emb_yx = tajima_cross_embedding(x, y, 10, 20)
    '''
    if not isinstance(cause, numpy.ndarray):
        cause = numpy.array(cause)
    if not isinstance(effect, numpy.ndarray):
        effect = numpy.array(effect)
    assert len(cause.shape) == 1
    assert len(effect.shape) == 1
    assert cause.shape[0] == effect.shape[0]
    
    dmax = 16
    max_dmax = int(effect.shape[0] / 2)
    
    max_corr_all_dmax = None
    max_corr_dmax_all_dmax = None
    max_corr_d_all_dmax = None
    pm_all_dmax = None
    corrs_all_dmax = None
    
    while dmax <= max_dmax:
        sys.stderr.write('dmax = {}\n'.format(dmax))
        projection_matrix = rng.normal(0.0, 1.0, size=(dmax, dmax))
        
        corrs = numpy.zeros(dmax, dtype=float)
        for d in range(1, dmax + 1):
            pm_d = projection_matrix[:d, :]
            emb_d = ProjectionEmbedding(effect, dmax, d, projection_mat=pm_d)

            ccm_result, y_actual, y_pred = emb_d.ccm(emb_d, cause, neighbor_count=neighbor_count, theiler_window=theiler_window)
            corrs[d-1] = ccm_result['correlation']
            sys.stderr.write('d = {}, corr = {}\n'.format(d, corrs[d-1]))
        
        max_corr_d = numpy.argmax(corrs) + 1
        max_corr = numpy.max(corrs)
        
        if max_corr_all_dmax is None or max_corr > max_corr_all_dmax:
            max_corr_all_dmax = max_corr
            max_corr_dmax_all_dmax = dmax
            max_corr_d_all_dmax = max_corr_d
            pm_all_dmax = projection_matrix
            corrs_all_dmax = corrs
            
            dmax *= 2
        else:
            break
    
    sys.stderr.write('best dmax, d: {}, {}\n'.format(max_corr_dmax_all_dmax, max_corr_d_all_dmax))
    
    threshold_corr = corr_threshold * max_corr_all_dmax
    corr_d = numpy.argmax(corrs_all_dmax >= threshold_corr) + 1
    corr = corrs_all_dmax[corr_d - 1]
    
    pm = pm_all_dmax[:corr_d,:]
    sys.stderr.write('min d >= threshold: d = {}, corr = {}\n'.format(corr_d, corr))

    return ProjectionEmbedding(effect, max_corr_dmax_all_dmax, corr_d, projection_mat=pm)



def correlation_valid(x, y):
    invalid = numpy.logical_or(numpy.isnan(x), numpy.isnan(y))
    valid = numpy.logical_not(invalid)
    valid_count = valid.sum()

    if valid_count == 0:
        corr = float('nan')
        sd_x = float('nan')
        sd_y = float('nan')
    else:
        sd_x = numpy.std(x[valid])
        sd_y = numpy.std(y[valid])
        
        if sd_x == 0 and sd_y == 0:
            corr = 1.0
        elif sd_x == 0 or sd_y == 0:
            corr = 0.0
        else:
            corr = numpy.corrcoef(x[valid], y[valid])[0,1]
    
    return corr, valid_count, sd_x, sd_y
