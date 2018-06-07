#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#
# This is the Cython translation of computing the entropy over multiple
# values. This is much faster that computing in raw Python.
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

import numpy as np
cimport numpy as np

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef np.npy_intp INTP
ctypedef np.npy_float FLOAT
ctypedef np.float64_t DOUBLE
ctypedef np.float32_t FLOAT32
ctypedef np.int64_t LONG


np.import_array()


# Make this a cpdef so we can test it directly in our py framework
cpdef FLOAT C_entropy(np.ndarray[DOUBLE, ndim=1, mode='c'] x,
                      np.ndarray[FLOAT32, ndim=1, mode='c'] cts):
    # compute the entropy score
    cdef INTP n = cts.shape[0]
    if n == 1:
        return 0.0

    # allocate the pr_C array
    cdef FLOAT running_sum = 0.0
    cdef FLOAT n_x = float(x.shape[0])
    cdef FLOAT pr_Ci

    # get the class probas and compute the entropy (update the running sum)
    for i in range(n):
        pr_Ci = cts[i] / n_x  # probability of class i
        running_sum += (-pr_Ci * np.log2(pr_Ci))
        #print(pr_Ci)
        #print(running_sum)

    return running_sum


def entropy_bin_bounds(np.ndarray[DOUBLE, ndim=1, mode='c'] x,
                       np.ndarray[LONG, ndim=1, mode='c'] unq,
                       np.ndarray[FLOAT32, ndim=1, mode='c'] cts,
                       INTP n_bins):
    """Compute the bin boundaries via entropy"""
    # start by computing the entropy of the entire thing
    cdef FLOAT overall_entropy = C_entropy(x, cts)

    # for each of the unique values, compute the entropy
    # TODO:

