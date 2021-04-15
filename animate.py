import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pylab

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))




plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()

def animate(i):
    lag = 50
    threshold = 3.9
    influence = 0
    data = pd.read_csv('newdata.csv')
    x = data['time']
    y = data['heart_rate']
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    plt.cla()
    pylab.subplot(211)
    pylab.plot(x,y)
    pylab.plot(x,
           result["avgFilter"], color="cyan", lw=2)
    pylab.plot(x,
           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    pylab.plot(x,
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    pylab.subplot(212)
    pylab.step(x, result["signals"], color="red", lw=2)
    pylab.ylim(-1.5, 1.5)
    plt.tight_layout()


#print("There may be a problem at times:", np.nonzero(result['signals'])[0])

ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
