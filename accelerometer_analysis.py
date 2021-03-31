import pandas as pd # version 1.0.5
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import numpy as np
from scipy import fft, ifft, fftpack
from scipy import optimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import re
import fbprophet as fb

def file_ext_to_list(empty_list, ext):
    for root, dirs_list, files_list in os.walk(dir):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == ext:
                empty_list.append(file_name)

def readFile(infile):
    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()
        lines = lines[:-2]
        x  = []
        y  = []
        z  = []
        tfps = []


    for line in lines[25:]:
        linedata = regex.findall(line)
        x.append(float(linedata[0]))
        y.append(float(linedata[1]))
        z.append(float(linedata[2]))
        tfps.append(float(linedata[3]))


    data = {'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 'tfps': np.array(tfps)}
    return data

def plot_dataframe(csvfilename, dataframe):
    # print(dataframe_motion.head())
    # plot on same axes

    plt.figure()
    ax = plt.gca()
    ax.plot(dataframe.index, dataframe.x, label='x-axis', color='red')
    ax.plot(dataframe.index, dataframe.y, label='y-axis', color='blue')
    ax.plot(dataframe.index, dataframe.z, label='z-axis', color='green')
    plt.vlines(x=40, ymin=-3, ymax=3, colors='black', ls=':', lw=2, label='flailing start point')
    plt.title(csvfilename[:-4].replace('_', " ").title())
    plt.xlabel('time (sec)')
    plt.ylabel('position relative to origin')
    plt.legend()
    plt.show()


# get txt files in list to iterate over later
# dir is directory where script is run from; also the directory which contains the log files

dir = '/Users/vedangjoshi/PycharmProjects/MDM3_Phase_C/Accelerometer_Data'
txt_extension = '.txt'

txt_file_name_list = []
file_ext_to_list(txt_file_name_list, txt_extension)

dataframe_list_swimming = []
dataframe_list_flailing = []
for txtfile in txt_file_name_list:
    dataframe_motion = pd.DataFrame(readFile(dir+'/'+txtfile))
    dataframe_motion = dataframe_motion.rename_axis("index")
    print(dataframe_motion)
    df_swimming = dataframe_motion.iloc[:450, :]
    df_flailing = dataframe_motion.iloc[451:, :]
    dataframe_list_swimming.append(df_swimming)
    dataframe_list_flailing.append(df_flailing)


# graphs to show seasonal_decompose
def seasonal_decompose(arr, csvfilename):
    decomposition = sm.tsa.seasonal_decompose(arr, model='additive', period=10)
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.title(csvfilename[:-4].replace('_', " ").title())
    plt.xlabel('time (sec)')
    plt.show()


# Augmented Dickey-Fuller Test to detect if data stationary or not

def augmented_dickey_fuller_test(timeseries, csvfilename, attr):
    print(csvfilename[:-4].replace('_', " ").title()+' %s'%(attr))
    dickey_fuller_test = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = %f'%(dickey_fuller_test[0]))
    print('P-value = %f'%(dickey_fuller_test[1]))
    print('Critical values :')
    for k, v in dickey_fuller_test[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not'
        if v < dickey_fuller_test[0] else '', 100-int(k[:-1])))

for i,j in zip(txt_file_name_list, dataframe_list_flailing):
    seasonal_decompose(arr=j.x, csvfilename=i)
    seasonal_decompose(arr=j.y, csvfilename=i)
    seasonal_decompose(arr=j.z, csvfilename=i)

    augmented_dickey_fuller_test(j.x, csvfilename=i, attr='x')
    augmented_dickey_fuller_test(j.y, csvfilename=i, attr='y')
    augmented_dickey_fuller_test(j.z, csvfilename=i, attr='z')

    pd.plotting.lag_plot(j.x)
    plt.show()
    pd.plotting.lag_plot(j.y)
    plt.show()
    pd.plotting.lag_plot(j.z)
    plt.show()
