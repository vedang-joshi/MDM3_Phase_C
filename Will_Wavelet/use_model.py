from tensorflow.keras.models import load_model
from data_reader import readFile
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from train_model import *
from load_data import *

model = load_model('scaled_model.h5')


drownData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_drowning1.txt'
swimData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_swimming1.txt'
swimData2 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/crawl_swimming1.txt'

Fs = 20
frame_size = Fs*4
hop_size = Fs*2
N_FEATURES = 3

# TRAIN MODEL

files = [drownData1, swimData1, swimData2]

X_train, X_test, y_train, y_test = prep_train_data(files, frame_size, hop_size, N_FEATURES)

input_shape = X_train[0].shape

cnn_model = build_model1("relu", input_shape)
#cnn_model.save('initial_model.h5')

model, history = compile_and_fit_model(cnn_model, X_train, y_train, X_test, y_test, 30)

#model.save('scaled_model.h5')

LABEL_NAMES = ['Drowning','Swimming']

#model = load_model('initial_model.h5')
# make predictions for test data
y_pred = model.predict_classes(X_test)

# determine the total accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#create_confusion_matrix(y_pred, y_test)


# TEST NEW DATA

X1_test = prep_test_data(drownData1, frame_size, hop_size, N_FEATURES)

prob = model.predict_proba(X1_test)

for i in prob:
    if i[0] > 0.5:
        print("Drowning: ", round(i[0]*100), '%')
    elif i[1] > 0.5:
        print("Swimming: ", round(i[1]*100), '%')
