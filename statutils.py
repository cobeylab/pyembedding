import sys
import numpy
from scipy.interpolate import interp1d

def bootstrap(vec, func, n_bootstraps):
    results = list()
    for i in xrange(n_bootstraps):
        vec_bs = numpy.random.choice(vec, replace=True)
        results.append(func(vec_bs))
    return results


def inverse_quantile(x, y):
    '''Calculates the location of each member of y in x

    :param x:
    :param y:
    :return:

    >>> inverse_quantile([1,2,3], 2.5).tolist()
    0.75
    >>> inverse_quantile([1,2,3], [2.5]).tolist()
    [0.75]
    >>> inverse_quantile([1,1,1], [0.99, 1, 1.01]).tolist()
    [0.0, 0.5, 1.0]
    >>> inverse_quantile([1], [0.5, 1.0, 1.5]).tolist()
    [0.0, 0.5, 1.0]
    >>> inverse_quantile([1, 2, 3], [[0.5, 1.0], [1.5, 2.5]]).tolist()
    [[0.0, 0.0], [0.25, 0.75]]
    '''
    if not isinstance(x, numpy.ndarray):
        x = numpy.array(x)
    if not isinstance(y, numpy.ndarray):
        y = numpy.array(y)
    x.sort()

    assert len(x.shape) == 1

    if x[0] == x[-1]:
        inv_q_y = numpy.ones_like(y) * float('nan')
        inv_q_y[y == x[0]] = 0.5
    else:
        quantiles = numpy.linspace(0.0, 1.0, num=len(x), endpoint=True)
        interp_func = interp1d(x, quantiles, bounds_error=False)
        inv_q_y = interp_func(y)
    inv_q_y[y < x[0]] = 0.0
    inv_q_y[y > x[-1]] = 1.0
    assert numpy.all(numpy.logical_not(numpy.isnan(y)))

    return inv_q_y

