from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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


fig = plt.figure()
#creating a subplot
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)


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

    ax1.clear()
    ax1.plot(x,y)
    ax1.plot(x,
           result["avgFilter"], color="cyan", lw=2)
    ax1.plot(x,
           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    ax1.plot(x,
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    ax2.step(x, result["signals"], color="red", lw=2)
    ax2.set_ylim(-1.5, 1.5)
    fig.tight_layout()


#print("There may be a problem at times:", np.nonzero(result['signals'])[0])

ani = FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()
