
import cv2
cimport cython
import functools
import numpy as np
cimport numpy as np
from kmc2 import kmc2
from libc.stdio cimport printf
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

DTYPE = np.float64
INT32_DTYPE = np.int32
INT64_DTYPE = np.int64
UINT8_DTYPE = np.uint8
ctypedef np.float64_t DTYPE_t
ctypedef np.uint8_t UINT8_DTYPE_t
ctypedef np.int64_t INT64_DTYPE_t
ctypedef np.int32_t INT32_DTYPE_t
ctypedef np.int_t INT_DTYPE_t

def reshape_clusters(func=None):
    if not func:
        return functools.partial(reshape_clusters)
    @functools.wraps(func)
    def cluster_reshape(*args, **kwargs):
        cdef np.ndarray clusters
        cdef tuple shape
        clusters, shape = func(*args, **kwargs)
        return clusters.reshape(shape)
    return cluster_reshape

@cython.boundscheck(False)
@reshape_clusters
def k_means(np.ndarray[DTYPE_t, ndim=2] x, int k, tuple size):
    """
    Cluster input vectors using K-Means.
    This implementation seeds initial means using the kmc2 algorithm
    https://github.com/obachem/kmc2

    Parameters
    ----------
    x: array_like
        Input values to be clustered (stored column-wise).
    k: int
        Number of clusters to form.

    Returns
    -------
    classes: array_like
        Cluster class assignment for each input vector.
    """
    cdef np.ndarray[DTYPE_t, ndim=2] seeding = kmc2(x.T, k)
    cdef np.ndarray quantization = MiniBatchKMeans(k, init=seeding).fit_predict(x.T)
    return quantization.T, size

@cython.boundscheck(False)
@reshape_clusters
def gmm(np.ndarray[DTYPE_t, ndim=2] x, int k, tuple size):
    """
    Approximate Gaussian Mixture Model over input values.
    This implementation seeds initial means using the kmc2 algorithm
    https://github.com/obachem/kmc2

    Parameters
    ----------
    x: array_like
        Input values to be clustered (stored column-wise).
    k: int
        Number of Gaussian distributions to fit.

    Returns
    -------
    classes: array_like
        Distribution class assignment for each input vector (most likely).
    """
    cdef np.ndarray[DTYPE_t, ndim=2] seeding = kmc2(x.T, k)
    model = GaussianMixture(n_components=k, means_init=seeding).fit(x.T)
    return model.predict(x.T).T, size


@cython.boundscheck(False)
@reshape_clusters
def hierarchical(np.ndarray[DTYPE_t, ndim=2] x, int k, tuple size):
    """
    Cluster input values by hierarchical clustering.
    This routine is based on Euclidean distance and Ward linkage
    (Euclidean similarity).

    Parameters
    ----------
    x: array_like
        Input values to be clustered (stored column-wise).
    k: int
        Number of clusters to form.

    Returns
    -------
    classes: array_like
        Cluster class assignment for each input vector.
    """
    model = AgglomerativeClustering(n_clusters=k).fit(x.T)
    return model.predict(x.T).T, size


@cython.boundscheck(False)
def complement(np.ndarray[DTYPE_t, ndim=2] x):
    """
    Array complement calculation.

    Parameters
    ----------
    x: array_like
        Input array to invert.

    Returns
    -------
    complement: array_like
        Complement of input array.
    """
    return np.max(x) - x


@cython.boundscheck(False)
def morphological_reconstruction(np.ndarray[DTYPE_t, ndim=2] marker,
                                 np.ndarray[DTYPE_t, ndim=2] mask):
    """
    Geodesic morphological reconstruction based on dilation.
    Adapted from Gala library: https://github.com/janelia-flyem/gala

    Parameters
    ----------
    marker: array_like
        Initial regional minima marker matrix.
    mask: array_like
        Reference matrix to reconstruct.

    Returns
    -------
    marker: array_like
        Geodesical reconstruction of original array from reference markers.
    """
    cdef np.ndarray[UINT8_DTYPE_t, ndim=2] cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    cdef bint diff = True
    cdef np.ndarray[DTYPE_t, ndim=2] markernew
    while diff:
        markernew = cv2.dilate(marker, cross, iterations=1)
        markernew = np.minimum(markernew, mask)
        diff = np.max(markernew-marker) > 0
        marker = markernew
    return marker


@cython.boundscheck(False)
def regional_minima(np.ndarray[DTYPE_t, ndim=2] x):
    """
    Calculate regional minima on input matrix.
    Adapted from Gala library: https://github.com/janelia-flyem/gala

    Parameters
    ----------
    x: array_like
        Input array.

    Returns
    -------
    markers: array_like
        Boolean matrix that indicates for every coordinate if the value on the
        input is a regional minimum.
    """
    cdef np.ndarray[DTYPE_t, ndim=1] values = np.unique(x)
    cdef np.ndarray[DTYPE_t, ndim=1] fil = ndi.minimum_filter(values, size=(3,))
    cdef np.ndarray[DTYPE_t, ndim=1] diff = values - fil
    cdef DTYPE_t delta = np.min(diff[1:])
    cdef np.ndarray[DTYPE_t, ndim=2] marker = complement(x)
    cdef np.ndarray[DTYPE_t, ndim=2] mask = marker + delta
    return marker == morphological_reconstruction(marker, mask)


@cython.boundscheck(False)
def impose_minima(np.ndarray[DTYPE_t, ndim=2] x,
                  np.ndarray minima):
    """
    Impose regional minima markers on input matrix.
    Adapted from Gala library: https://github.com/janelia-flyem/gala

    Parameters
    ----------
    x: array_like
        Input matrix on which minima is going to be imposed.
    minima: array_like
        Boolean matrix that indicates for every coordinate if the value on the
        input is a regional minimum.
    """
    cdef DTYPE_t m = np.max(x)
    cdef np.ndarray[DTYPE_t, ndim=2] mask = m - x
    cdef np.ndarray[DTYPE_t, ndim=2] marker = np.zeros_like(mask, dtype=DTYPE)
    marker[minima] = mask[minima]
    return m - morphological_reconstruction(marker, mask)


@cython.boundscheck(False)
def hminima(np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h):
    """
    H-minima transform.

    Suppresses all minima on the input matrix whose depth is less than h.
    Adapted from Gala library: https://github.com/janelia-flyem/gala

    Parameters
    ----------
    x: array_like
        Input matrix on which minima is going to be suppressed.
    h: float
        Lower bound of the minima minimum acceptable depth.

    Returns
    -------
    hminima: array_like
        Copy of the original array with the minima suppressed.
    """
    cdef DTYPE_t maxval = np.max(x)
    cdef np.ndarray[DTYPE_t, ndim=2] xinv = maxval - x
    return maxval - morphological_reconstruction(xinv - h, xinv)

@cython.boundscheck(False)
def watershed(img, grad, k):
    """
    Calculate watershed transform based on grayscale gradient and
    regional h-minima imposition, the function iteratively chooses
    a height and counts the number of different regions segmented,
    until the number is at least k.

    Parameters
    ----------
    img: array_like
        Original image to segment.
    grad: array_like
        Grayscale gradient of the image.
    k: int
        Number of segmentation regions to consider.

    Returns
    -------
    seg: array_like
        Watershed segmentation of original image, which contains at least
        k segmentation regions.
    """
    cdef DTYPE_t h = 1
    cdef int num_seg
    cdef np.ndarray seg
    cdef np.ndarray[INT32_DTYPE_t, ndim=2] markers
    cdef np.ndarray bool_markers = regional_minima(hminima(grad, 0))
    markers = ndi.label(bool_markers)[0]
    seg = cv2.watershed(img, markers)
    num_seg = len(np.unique(seg))
    cdef DTYPE_t alpha = 0.05
    cdef DTYPE_t prev_est = num_seg
    cdef DTYPE_t best_est_h = 0
    cdef int best_est_diff = 1000000
    cdef int best_est_seg = 1000000
    cdef int diff
    while num_seg > k:
        bool_markers = regional_minima(hminima(grad, h))
        markers = ndi.label(bool_markers)[0]
        seg = cv2.watershed(img, markers)
        num_seg = len(np.unique(seg))
        if num_seg == prev_est:
            alpha *= 10
        print(num_seg)

        prev_est = num_seg
        diff = num_seg-k
        if diff > 0:
            best_est_diff = diff
            best_est_h = h
            best_est_seg = num_seg
            alpha *= 10
        elif diff < 0:
            h = best_est_h
            num_seg = best_est_seg
            alpha /= 100
        h += alpha*(num_seg-k)
    print("Depth: %g" % (h))
    return seg
