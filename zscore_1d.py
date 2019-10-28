#!/usr/bin/env python
# Implementation of algorithm from http://stackoverflow.com/a/22640362/6029703
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, interpolate


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])

    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1
                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
                avgFilter[i] = np.mean(filteredY[(i - lag):i])
                stdFilter[i] = np.std(filteredY[(i - lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])
    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


from numpy import genfromtxt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import medfilt


def find_inflections(xdata, ydata, plot=False):
    hextremes = argrelextrema(ydata, np.greater, 0, 2)
    peaks = []
    for p in hextremes:
        peaks.append(p)
    xpeaks = xdata[tuple(peaks)]
    ypeaks = ydata[tuple(peaks)]

    lextremes = argrelextrema(ydata, np.less, 0, 2)
    valleys = []
    for v in lextremes:
        valleys.append(v)
    xvalleys = xdata[tuple(valleys)]
    yvalleys = ydata[tuple(valleys)]

    medy = np.median(ydata[0:110])
    medydata = np.ones(len(ydata)) * medy
    print(medy)
    ## compute distance of all valleys from median
    medyvalleys = np.zeros(len(yvalleys))
    medxvalleys = np.zeros(len(xvalleys))
    mind = medy
    minx = 0
    minidx = -1
    for idx, vv in enumerate(yvalleys):
        if vv > medy: continue
        dd = vv - medy
        dd = dd * dd
        dd = np.sqrt(dd)
        print('%d,%d,%f,%f' % (idx, xvalleys[idx], vv, dd))
        medyvalleys[idx] = dd
        medxvalleys[idx] = xvalleys[idx]
        if dd < mind:
            mind = dd
            minidx = idx
            minx = xvalleys[idx]

    print('%f,%f,%f' % (minidx, minx, mind))

    if plot:
        plt.figure()
        plt.plot(xdata, ydata)
        plt.plot(xdata, medydata, 'yo')
        plt.plot(xpeaks, ypeaks, 'ro')
        plt.plot(xvalleys, yvalleys, 'go')
        plt.show()

    dr = {'inflection': (minidx, minx, mind),
          'median': medy,
          'xvalleyes': xvalleys,
          'yvalleyes': yvalleys,
          'xpeaks': xpeaks,
          'ypeaks': ypeaks,

          'medxvalleyes': medxvalleys,
          'medyvalleyes': medyvalleys,
          }
    return dr


if __name__ == '__main__':
    # Data
    my_data = genfromtxt('./projects/pupil/images/sig2.txt', delimiter=',')
    dshape = my_data.shape
    if len(dshape) == 1:
        y = my_data
        x = np.arange(len(y))
    else:
        y = my_data[:, 1]
        x = my_data[:, 0]
    result = find_inflections(x, y, True)
    print(result)

    # # Settings: lag = 30, threshold = 5, influence = 0
    # lag = 5
    # threshold = 1
    # influence = 1
    # # Run algo with settings from above
    # result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    # # Plot result
    # pylab.subplot(211)
    # pylab.plot(np.arange(1, len(y) + 1), y)
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"], color="cyan", lw=2)
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    # pylab.plot(np.arange(1, len(y) + 1),
    #            result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    # pylab.subplot(212)
    # pylab.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    # pylab.ylim(-1.5, 1.5)
    # pylab.show()
