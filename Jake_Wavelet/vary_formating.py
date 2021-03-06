
def formating_main(train_data, train_labels, test_data, test_labels,MAX_SCALE,DIMENSIONS):
    import numpy as np
    import pywt
    import pywt.data


    train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = train_data, train_labels, test_data, test_labels
    print('\n train data \n', train_signals_ucihar, '\n train data shape \n', test_signals_ucihar.shape)
    print( '\n train labels \n',train_labels_ucihar,'\n train labels length \n',len(test_labels_ucihar))

    scales = range(1,MAX_SCALE+1)


    print('here', test_signals_ucihar.shape, train_signals_ucihar.shape, len(train_labels_ucihar))
    waveletname = 'morl'
    train_size = train_signals_ucihar.shape[0]
    test_size= test_signals_ucihar.shape[0]

    # MIGIHT NEED TO CHANGE THE 9 HERE TO HOW MANY DIMENSIONS OUR DATA IS....
    #┬áPROBABLY 4... ACCELEROMETER X,Y,Z AND HEART RATE
    train_data_cwt = np.ndarray(shape=(train_size, MAX_SCALE, MAX_SCALE, DIMENSIONS))


    for ii in range(0,train_size):
        if ii % 80 == 0:
            print(ii)
        for jj in range(0,DIMENSIONS):
            signal = train_signals_ucihar[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:MAX_SCALE]
            train_data_cwt[ii, :, :, jj] = coeff_
    print('made it past train')

    test_data_cwt = np.ndarray(shape=(test_size, MAX_SCALE, MAX_SCALE, DIMENSIONS))
    for ii in range(0,test_size):
        if ii % 5 == 0:
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
    return x_train,y_train,x_test,y_test
