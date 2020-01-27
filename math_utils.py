
import numpy as np
import time
import unittest

'''
 Lines are already been pruned to same angle 
 We are looking for quadrilaterals

 '''


def lines2quadrilaterals(lines):
    def same_line(l1, l2):
        return l1[1] == l2[1] and l1[2] == l2[2] and l1[0] == l2[0] and l1[3] == l2[3]

    cands = []
    for s1, s2 in combinations(lines, 2):
        p1, p2 = map(Point, [(s1[0], s1[1]), (s1[2], s1[3])])
        p3, p4 = map(Point, [(s2[0], s2[1]), (s2[2], s2[3])])
        poly = Polygon(p1, p2, p3, p4)
        print(('check ', poly))
        passit = True
        for other in lines:
            if same_line(s1, other) or same_line(s2, other): continue
            p1, p2 = map(Point, [(other[0], other[1]), (other[2], other[3])])
            print(('against ', p1, p2))
            if poly.encloses_point(p1) or poly.encloses_point(p2):
                passit = False
                print(('bad ', poly))
                break
        print(('good ', poly))
        if passit: cands.append(poly)
    return cands


#from caput.weighted_median import weighted_median, moving_weighted_median


def py_weighted_median(data, weights):
    """Flattens the given arrays and calculates a weighted median from that."""
    data = np.reshape(data, np.prod(np.shape(data)))
    weights = np.reshape(weights, np.prod(np.shape(weights)))

    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()

    # remove values with 0-weights
    choice = weights != 0
    data = data[choice]
    weights = weights[choice]

    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        try:
            idx_upper = (
                np.intersect1d(
                    np.where(cs_weights > midpoint)[0],
                    np.where(cs_weights != 0)[0],
                    True,
                )[0]
                + 1
            )
        except IndexError:
            idx_upper = len(data)
            # skip zero-weights
            try:
                while weights[idx_upper] == 0:
                    idx_upper += 1
            except IndexError:
                pass
        try:
            idx_lower = (
                np.intersect1d(
                    np.where(cs_weights[-1] - cs_weights > midpoint)[0],
                    np.where(cs_weights != 0)[0],
                    True,
                )[-1]
                + 1
            )
        except IndexError:
            idx_lower = 0
            # skip zero-weights
            try:
                while weights[idx_lower] == 0:
                    idx_lower -= 1
            except IndexError:
                pass
        if idx_upper == len(data) and idx_lower == -1:
            # All weights are 0.
            return 0
        w_median = np.mean(s_data[idx_lower:idx_upper])
    return w_median


def py_mwm_1d(values, weights, size):
    """Moving weighted median (one-dimensional)."""
    medians = []

    # slide a window of size <size> over the value array and get weighted median inside window
    for i in range(len(values)):

        # size is bigger than value array
        if i + size // 2 >= len(values) and i - size // 2 < 0:
            medians.append(py_weighted_median(values, weights))

        # window is sliding into the value array
        elif i - size // 2 < 0:
            medians.append(
                py_weighted_median(
                    values[: i + size // 2 + 1], weights[: i + size // 2 + 1]
                )
            )

        # window is sliding over the end of the value array
        elif i + size // 2 >= len(values):
            medians.append(
                py_weighted_median(values[i - size // 2 :], weights[i - size // 2 :])
            )

        # the normal case: window is inside the value array
        else:
            medians.append(
                py_weighted_median(
                    values[i - size // 2 : i + size // 2 + 1],
                    weights[i - size // 2 : i + size // 2 + 1],
                )
            )

    return medians


def py_mwm_nd(values, weights, size):
    """Moving weighted median (n-dimensional)."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    medians = np.ndarray(values.shape)

    # window radius around the index we want to get a median for
    r = np.floor_divide(size, 2).astype(int)

    # iterate over n-dim array
    for index, value in np.ndenumerate(values):

        # get the edge indides of the window
        lbound = np.subtract(index, r, dtype=int)
        hbound = np.add(index, r + 1, dtype=int)

        # make sure they are inside the array
        lbound = np.maximum(lbound, 0)
        hbound = np.minimum(hbound, np.shape(values))

        window = tuple([slice(i, j, 1) for (i, j) in zip(lbound, hbound)])
        medians[index] = py_weighted_median(
            values[window].flatten(), weights[window].flatten()
        )
    return medians

def slidingWindow(sequence, windowSize=2, stepSize=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    Parameters
    a  - a 1D array
    windowSize - the window size, in samples
    stepSize - the step size, in samples. If not provided, window and step size
             are equal.
    """

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((isinstance(windowSize, type(0))) and (isinstance(stepSize, type(0)))):
        raise Exception(
            "**ERROR** type(windowSize) and type(stepSize) must be int.")
    if stepSize > windowSize:
        raise Exception(
            "**ERROR** stepSize must not be larger than windowSize.")
    if windowSize > len(sequence):
        print("**ERROR** windowSize must not be larger than sequence length.")
        return None

    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - windowSize) / stepSize) + 1

    outputWindows = []
    for i in range(0, numOfChunks * stepSize, stepSize):
        outputWindows.append(sequence[i:i + windowSize])

    return outputWindows

def smooth(x, window_len=3, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.
    output:
            the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def medianFilter(x, k=9):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

class TestMWM(unittest.TestCase):
    def test_the_test(self):
        mwm = py_mwm_1d(
            values=[1, 2, 3, 4, 5, 6, 7, 8], weights=[1, 2, 3, 4, 5, 6, 7, 8], size=3
        )
        assert mwm == [2, 2.5, 3, 4, 5, 6, 7, 8]

        assert (
            py_weighted_median([1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 1])
            == 4.5
        )

        assert py_weighted_median([1, 3, 3, 7], [0, 7, 7, 0]) == 3

    def test_the_nd_test(self):

        # 2D
        values = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        mwm = py_mwm_nd(values, values, (3, 3))
        np.testing.assert_array_equal(
            [[2.0, 2.5, 3.0], [2.5, 2.5, 2.5], [3.0, 2.5, 2.0]], mwm
        )

    def test_wm(self):
        assert (
            weighted_median([1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 1]) == 4.5
        )

        assert (
            weighted_median(
                [1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 1], method="lower"
            )
            == 1
        )

        assert (
            weighted_median(
                [1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 1], method="higher"
            )
            == 8
        )

        values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        weights = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = np.asarray([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.float64)
        np.testing.assert_array_equal(
            py_weighted_median(values, values), weighted_median(values, values)
        )

        values = np.asarray([0.1, 0, 7.7, 0, 9.999, 42, 0, 1, 9, 1], dtype=np.float64)
        weights = np.asarray([0, 7.5, 0.33, 23.23, 0, 4, 7, 8, 9, 0], dtype=np.float64)
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = [0, 0, 7, 0, 9]
        weights = [0, 7, 1, 23, 0]
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = [0, 7, 0, 9, 42]
        weights = [7, 1, 23, 0, 4]
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = [7, 0, 9, 42, 0]
        weights = [1, 23, 0, 4, 7]
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = [0, 9, 42, 0, 1]
        weights = [23, 0, 4, 7, 8]
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

        values = [0, 4, 6, 7, 8, 8]
        weights = [7, 6, 2, 0, 8, 7]
        np.testing.assert_array_equal(
            py_weighted_median(values, weights), weighted_median(values, weights)
        )

    def test_weighted_median_methods(self):
        values = [[9, 2], [5, 5], [2, 9]]
        weights = [[3, 0], [5, 0], [8, 0]]
        np.testing.assert_equal(weighted_median(values, weights, method="lower"), 2)
        np.testing.assert_equal(weighted_median(values, weights, method="higher"), 5)

    def test_1d_mwm_int(self):
        values = [0, 0, 7, 0, 9, 42, 0, 1, 9, 1]
        weights = [0, 7, 1, 23, 0, 4, 7, 8, 9, 0]
        np.testing.assert_array_almost_equal(
            py_mwm_1d(values, weights, 5), moving_weighted_median(values, weights, 5)
        )

    def test_1d_mwm(self):
        values = [0.1, 0, 7.7, 0, 9.999, 42, 0, 1, 9, 1]
        weights = [0, 7.5, 0.33, 23.23, 0, 4, 7, 8, 9, 0]
        np.testing.assert_array_equal(
            py_mwm_1d(values, weights, 5), moving_weighted_median(values, weights, 5)
        )

    # These two are just for measuring performance:
    def test_1d_mwm_big(self):
        N = 100
        values = np.random.random_sample(N)
        weights = np.random.random_sample(N)
        t0 = time.time()
        res_py = py_mwm_1d(values, weights, 5)
        t_py = time.time() - t0
        t0 = time.time()
        res_cython = moving_weighted_median(values, weights, 5)
        t_cython = time.time() - t0
        np.testing.assert_array_equal(res_py, res_cython)
        print(
            "1D moving weighted median with {} elements took {}s / {}s".format(
                N, t_py, t_cython
            )
        )

    def test_2d_mwm_big(self):
        N = 100
        M = 100
        N_w = 5
        values = np.random.random_sample((N, M))
        weights = np.random.random_sample((N, M))
        t0 = time.time()
        moving_weighted_median(values, weights, (N_w, N_w))
        t_cython = time.time() - t0
        print(
            "2D moving {}x{} weighted median with {}x{} elements took {}s".format(
                N_w, N_w, N, N, t_cython
            )
        )

    def test_zero_weights(self):
        values = [1, 1, 1]
        weights = [0, 0, 0]
        np.testing.assert_array_equal(
            [0, 0, 0], moving_weighted_median(values, weights, 1)
        )

    def test_2d_mwm_small(self):
        values = [[9, 2], [5, 5], [2, 9]]
        weights = [[3, 0], [5, 0], [8, 0]]
        np.testing.assert_array_equal(
            py_mwm_nd(values, weights, 3),
            moving_weighted_median(values, weights, (3, 3)),
        )

        values = [
            [8.0, 6.0, 1.0, 5.0],
            [6.0, 6.0, 8.0, 10.0],
            [8.0, 4.0, 4.0, 7.0],
            [10.0, 1.0, 7.0, 8.0],
        ]
        weights = [
            [8.0, 6.0, 1.0, 5.0],
            [6.0, 6.0, 8.0, 10.0],
            [8.0, 4.0, 4.0, 7.0],
            [10.0, 1.0, 7.0, 8.0],
        ]
        np.testing.assert_array_equal(
            py_mwm_nd(values, weights, 3),
            moving_weighted_median(values, weights, (3, 3)),
        )
        values = [[9, 4], [2, 5]]
        weights = [[4, 8], [4, 5]]
        np.testing.assert_array_equal(
            py_mwm_nd(values, weights, 3),
            moving_weighted_median(values, weights, (3, 3)),
        )

    def test_2d_mwm_int(self):
        values = np.asarray(np.random.randint(0, 10, (14, 8)), np.float64)
        weights = np.asarray(np.random.randint(0, 10, (14, 8)), np.float64)
        np.testing.assert_array_equal(
            py_mwm_nd(values, weights, (3, 3)),
            moving_weighted_median(values, weights, (3, 3)),
        )

    def test_2d_mwm(self):
        values = np.random.rand(14, 8)
        weights = np.random.rand(14, 8)
        np.testing.assert_array_equal(
            py_mwm_nd(values, weights, 3),
            moving_weighted_median(values, weights, (3, 3)),
        )

    # weights are all zeros for a region that is smaller than the window
    def test_small_zero_weight(self):

        window = (3, 3)
        zero_shape = (2, 2)
        shape = (10, 10)

        data = np.ones(shape, dtype=np.float64)
        weight = np.ones_like(data)
        weight[1 : zero_shape[0] + 1, 1 : zero_shape[1] + 1] = 0.0

        np.testing.assert_array_equal(
            data, moving_weighted_median(data, weight, window)
        )

    # weights are all zeros for a region that is the size of the window, this has historically
    # caused a segfault
    def test_med_zero_weight(self):

        window = (3,)
        zero_shape = (3,)
        shape = (10,)

        data = np.ones(shape, dtype=np.float64)
        weight = np.ones_like(data)
        weight[1 : zero_shape[0] + 1] = 0.0

        result = data.copy()
        result[2] = np.nan

        np.testing.assert_array_equal(
            result, moving_weighted_median(data, weight, window)
        )