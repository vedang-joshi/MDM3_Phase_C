''' This file will format our data so it can be used in the wavelet transform and CNN '''
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data_reader import readFile
from vary_cnn import cnn_main
from vary_formating import formating_main
import pandas as pd
import ast

from os import listdir
from os.path import isfile, join

mypath = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Vedang_Accelerometer_Crawl/26-28Apr_accelerometer_heart_rate_data'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
swimming_data = [2]
drowning_data = [1]
for i in onlyfiles:
    if "swim" in i:
        print('here swim', i)
        data = mypath + '/' + str(i)
        swimming_data.append(data)
    elif "drown" in i:
        print('here drown', i)
        data = mypath + '/' + str(i)
        drowning_data.append(data)
    elif 'heart' in i:
        print('here heart', i)
        heart_rate_data = mypath + '/' + str(i)
    else:
        print('Wrong name')

heart_rate_df = pd.read_csv(heart_rate_data)


# drowning = [1,drown_data1,drown_data2,drown_data4,drown_data5,drown_data6,drown_data7,drown_data10,drown_data11,drown_data12,cdrown_data5,cdrown_data13,cdrown_data14,cdrown_data15,cdrown_data16,cdrown_data17,cdrown_data18,cdrown_data19,cdrown_data20]
# breast = [2, breast_data1, breast_data4, breast_data5, breast_data6, breast_data7, breast_data10, breast_data11, breast_data12]
# crawl = [3, crawl_data1, crawl_data5, crawl_data13, crawl_data14, crawl_data15, crawl_data16, crawl_data17, crawl_data18, crawl_data19, crawl_data20]
all_data = [swimming_data,drowning_data]

fraction_of_data_that_is_test_data = 1/5

# DIMENSIONS is the number of variables we consider in the CNN (e.g. accelerometer x,y,z,heart rate ...)
DIMENSIONS = 4
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
HR_i_count = 0

label_idx = 0
for MAX_SCALE in m_scales:
    labels = []
    formatted_data = []
    for data in all_data:

        for set_of_data in data[1:]:
            trash_data, read_data = readFile(set_of_data)
            split_by_secs = MAX_SCALE
            remainder = len(read_data) % split_by_secs
            print('remainder', remainder)
            if remainder == 0:
                read_data = read_data
            else:
                read_data = read_data[:-remainder]
            print(read_data)
            length_data = len(read_data)
            hr_swim = heart_rate_df['Heart_rate_swimming'].iloc[HR_i_count]
            hr_swim = ast.literal_eval(hr_swim)
            hr_drown = heart_rate_df['Heart_rate_drowning'].iloc[HR_i_count]
            hr_drown = ast.literal_eval(hr_drown)
            split_data = np.array([read_data[x:x+split_by_secs] for x in range(0, len(read_data), split_by_secs)])
            print(split_data)
            if label_idx == 0:
                # use swim data
                num_hr_data_points = int(MAX_SCALE/len(hr_swim))
                original_num_hr_data_points = num_hr_data_points
                print(num_hr_data_points)
                init_data_points = 0
                print('hape', split_data.shape)
                hr_matrix = np.zeros((split_data.shape[0],split_data.shape[1],DIMENSIONS))
                for hr in hr_swim[:-1]:
                    for i in range(init_data_points,num_hr_data_points):

                        hr_matrix[:,i,-1]=hr
                    init_data_points +=original_num_hr_data_points
                    num_hr_data_points+=original_num_hr_data_points
                    print(num_hr_data_points)

                hr_matrix[:,init_data_points:,-1]=hr_swim[-1]

                #split_data-1 # add remaining hr to rest of data


            elif label_idx == 1:
                # use drown data
                # use swim data
                 num_hr_data_points = int(MAX_SCALE/len(hr_drown))
                 original_num_hr_data_points = num_hr_data_points
                 init_data_points = 0
                 hr_matrix = np.zeros((split_data.shape[0],split_data.shape[1],DIMENSIONS))
                 for hr in hr_drown[:-1]:
                     for i in range(init_data_points,num_hr_data_points):
                         hr_matrix[:,i,-1]=hr
                     init_data_points +=original_num_hr_data_points
                     num_hr_data_points+=original_num_hr_data_points
                     print(num_hr_data_points)

                 hr_matrix[:,init_data_points:,-1]=hr_drown[-1]

            hr_matrix[:,:,:-1] = split_data
            split_data = hr_matrix

            print('shape', split_data.shape)
            split_data.transpose()
            formatted_data.append(split_data)
            labels_list = [data[0]] * len(split_data)
            labels.append(labels_list)
            HR_i_count += 1
        label_idx += 1
        HR_i_count = 0





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
