import numpy as np
import pywt
import pywt.data
from transforming_data import *
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = train_data, train_labels, test_data, test_labels
print('\n train data \n', train_signals_ucihar, '\n train data shape \n', test_signals_ucihar.shape)
print( '\n train labels \n',train_labels_ucihar,'\n train labels length \n',len(test_labels_ucihar))
window = 10 # 10 data point per window
scales = range(1,window+1)


print('here', test_signals_ucihar.shape, train_signals_ucihar.shape, len(train_labels_ucihar))
waveletname = 'morl'
train_size = train_signals_ucihar.shape[0] * window
test_size= test_signals_ucihar.shape[0]*window


# This is where it needs to change to sliding window

# MIGIHT NEED TO CHANGE THE 9 HERE TO HOW MANY DIMENSIONS OUR DATA IS....
# PROBABLY 4... ACCELEROMETER X,Y,Z AND HEART RATE
train_data_cwt = np.ndarray(shape=(train_size, window, window, DIMENSIONS))

# # DIMENSIONS is the number of variables we consider in the CNN (e.g. accelerometer x,y,z,heart rate ...)
# DIMENSIONS = 3
# #all_data = [drowning, breast]
# # MAX_SCALE is the number of data point in each chunk that the labels are assigned to
# MAX_SCALE = 100

# train size 1228 * 10
# test_size = 306 *10
for ii in range(0,train_size):
    if ii % 80 == 0:
        print(ii)
    for jj in range(0,DIMENSIONS):
        signal = train_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:window]
        train_data_cwt[ii, :, :, jj] = coeff_
print('made it past train')

test_data_cwt = np.ndarray(shape=(test_size, window, window, DIMENSIONS))
for ii in range(0,test_size):
    if ii % 5 == 0:
        print(ii)
    for jj in range(0,DIMENSIONS):
        signal = test_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:window]
        test_data_cwt[ii, :, :, jj] = coeff_

uci_har_labels_train = list(map(lambda x: int(x) - 1, train_labels_ucihar))
uci_har_labels_test = list(map(lambda x: int(x) - 1, test_labels_ucihar))

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])

print('the data\n\n', x_train, '\n\nthe labels\n', y_train)
