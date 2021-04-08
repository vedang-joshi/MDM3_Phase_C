''' This file will format our data so it can be used in the wavelet transform and CNN '''
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data_reader import readFile

drown_data2= '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast_drowning1.txt'
brest_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast_swimming1.txt'
drown_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl_drowning1.txt'
crawl_data1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl_swimming1.txt'

drowning = [1,drown_data1,drown_data2]
brest = [2, brest_data1, crawl_data1]
crawl = [3, crawl_data1]
all_data = [drowning,brest,crawl]

fraction_of_data_that_is_test_data = 1/5

# DIMENSIONS is the number of variables we consider in the CNN (e.g. accelerometer x,y,z,heart rate ...)
DIMENSIONS = 3
#all_data = [drowning, brest]
# MAX_SCALE is the number of data point in each chunk that the labels are assigned to
MAX_SCALE = 100
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
print(test_data)

print(test_labels)
