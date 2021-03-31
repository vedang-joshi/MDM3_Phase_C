import numpy as np
import pywt
import pywt.data
from load_UCI_HAR_dataset import *

folder_ucihar = '/Users/jakebeard/Documents/GitHub/UCIHARDataset/'
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)
print(train_signals_ucihar, 'labels',train_labels_ucihar, test_signals_ucihar, 'labels',test_labels_ucihar)
MAX_SCALE = 127
scales = range(1,MAX_SCALE+1)
DIMENSIONS = 9
waveletname = 'morl'
train_size = 5000
test_size= 500

# MIGIHT NEED TO CHANGE THE 9 HERE TO HOW MANY DIMENSIONS OUR DATA IS....
# PROBABLY 4... ACCELEROMETER X,Y,Z AND HEART RATE
train_data_cwt = np.ndarray(shape=(train_size, MAX_SCALE, MAX_SCALE, DIMENSIONS))

for ii in range(0,train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,DIMENSIONS):
        signal = train_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:MAX_SCALE]
        train_data_cwt[ii, :, :, jj] = coeff_

test_data_cwt = np.ndarray(shape=(test_size, MAX_SCALE, MAX_SCALE, DIMENSIONS))
for ii in range(0,test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,DIMENSIONS):
        signal = test_signals_ucihar[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:MAX_SCALE]
        test_data_cwt[ii, :, :, jj] = coeff_

uci_har_labels_train = list(map(lambda x: int(x) - 1, train_labels_ucihar))
uci_har_labels_test = list(map(lambda x: int(x) - 1, test_labels_ucihar))

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])

print(x_train, y_train)
