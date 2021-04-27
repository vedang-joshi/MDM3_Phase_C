''' This file will format our data so it can be used in the wavelet transform and CNN '''
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data_reader import readFile
from vary_cnn import cnn_main
from vary_formating import formating_main



drown_data2= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast_drowning1.txt'
breast_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast_swimming1.txt'
drown_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl_drowning1.txt'
crawl_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl_swimming1.txt'
cdrown_data5 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__5_drowning.txt'
crawl_data5 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__5_swimming.txt'
cdrown_data13 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__13_drowning.txt'
crawl_data13 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__13_swimming.txt'
cdrown_data14 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__14_drowning.txt'
crawl_data14 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__14_swimming.txt'
cdrown_data15 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__15_drowning.txt'
crawl_data15 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__15_swimming.txt'
cdrown_data16 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__16_drowning.txt'
crawl_data16 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__16_swimming.txt'
cdrown_data17 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__17_drowning.txt'
crawl_data17 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__17_swimming.txt'
cdrown_data18 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__18_drowning.txt'
crawl_data18 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__18_swimming.txt'
cdrown_data19 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__19_drowning.txt'
crawl_data19 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__19_swimming.txt'
cdrown_data20 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__20_drowning.txt'
crawl_data20 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/crawl__20_swimming.txt'



breast_data4 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__4_swimming.txt'
drown_data4 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__4_drowning.txt'
breast_data5= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__5_swimming.txt'
drown_data5= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__5_drowning.txt'
breast_data6= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__6_swimming.txt'
drown_data6= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__6_drowning.txt'
breast_data7= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__7_swimming.txt'
drown_data7= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__7_drowning.txt'


breast_data10= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__10_swimming.txt'
drown_data10 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__10_drowning.txt'
breast_data11= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__11_swimming.txt'
drown_data11 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__11_drowning.txt'
breast_data12= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__12_swimming.txt'
drown_data12= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Breast/breast__12_drowning.txt'



drowning = [1,drown_data1,drown_data2,drown_data4,drown_data5,drown_data6,drown_data7,drown_data10,drown_data11,drown_data12,cdrown_data5,cdrown_data13,cdrown_data14,cdrown_data15,cdrown_data16,cdrown_data17,cdrown_data18,cdrown_data19,cdrown_data20]
breast = [2, breast_data1, breast_data4, breast_data5, breast_data6, breast_data7, breast_data10, breast_data11, breast_data12]
crawl = [3, crawl_data1, crawl_data5, crawl_data13, crawl_data14, crawl_data15, crawl_data16, crawl_data17, crawl_data18, crawl_data19, crawl_data20]
all_data = [drowning,breast,crawl]

fraction_of_data_that_is_test_data = 1/5

# DIMENSIONS is the number of variables we consider in the CNN (e.g. accelerometer x,y,z,heart rate ...)
DIMENSIONS = 3
#all_data = [drowning, breast]
# MAX_SCALE is the number of data point in each chunk that the labels are assigned to
m_scales = [300,200,100,50,25,20]
drs = []
trds = []
tsds = []
trdsl = []
tsdsl = []
testscores = []
trainscores = []
for MAX_SCALE in m_scales:
    labels = []
    formatted_data = []
    for data in all_data:
        for set_of_data in data[1:]:
            trash_data, read_data = readFile(set_of_data)
            split_by_secs = MAX_SCALE
            remainder = len(read_data) % split_by_secs
            read_data = read_data[:-remainder]
            split_data = np.array([read_data[x:x+split_by_secs] for x in range(0, len(read_data), split_by_secs)])
            print('shape', split_data.shape)
            split_data.transpose()
            formatted_data.append(split_data)
            labels_list = [data[0]] * len(split_data)
            labels.append(labels_list)




    formatted_data = [j for i in formatted_data for j in i]
    labels = [j for i in labels for j in i]

    print('len before', len(labels))

    ''' PICK RANDOM INDEX TO SPLIT INTO TRAIN AND TEST DATA '''
    test_labels = []
    test_data = []
    for i in range(0,int(len(labels)*(fraction_of_data_that_is_test_data))):
        random_index = random.randrange(0,len(labels))
        test_label = labels[random_index]
        test_point = formatted_data[random_index]
        test_labels.append(test_label)
        test_data.append(test_point)
        labels.pop(random_index)
        formatted_data.pop(random_index)


    #test_labels = [j for i in test_labels for j in i]
    #test_data = [j for i in test_data for j in i]
    print('len after ', len(labels))
    train_labels = labels
    train_data = np.asarray(formatted_data)
    test_data = np.asarray(test_data)
    x_train,y_train,x_test,y_test = formating_main(train_data, train_labels, test_data, test_labels,MAX_SCALE,DIMENSIONS)
    test_score, train_score = cnn_main(DIMENSIONS, MAX_SCALE,x_train,y_train,x_test, y_test)
    testscores.append(test_score)
    trainscores.append(train_score)
    print(test_data)

    print(test_labels)
print(testscores)
print(trainscores)
plt.plot(m_scales, testscores)
plt.title('Varying data chunks used in sliding window')
plt.ylabel('Accuracy of test scores')
plt.xlabel('Size of data in chunks')
plt.show()
