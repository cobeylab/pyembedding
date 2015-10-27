#!/usr/bin/env python

import os
import sys
import numpy
from cStringIO import StringIO

def ndarray_to_npy_buffer(x):
    f = StringIO()
    numpy.save(f, x)
    buf = buffer(f.getvalue())
    f.close()
    return buf

def npy_buffer_to_ndarray(x):
    f = StringIO(x)
    arr = numpy.load(f)
    f.close()
    return arr
