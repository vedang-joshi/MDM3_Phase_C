from data_reader import readFile
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split

fsDataswim1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_swimming1.txt'
fdDatadrown1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/breast_drowning1.txt'
fdData2 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/crawl_drowning1.txt'
fsData2 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/crawl_swimming1.txt'

# def get_frames(df, frame_size, hop_size, N_FEATURES):
#
#
#     frames = []
#     labels = []
#     for i in range(0, len(df) - frame_size, hop_size):
#         x = df['x'].values[i: i + frame_size]
#         y = df['y'].values[i: i + frame_size]
#         z = df['z'].values[i: i + frame_size]
#
#         # Retrieve the most often used label in this segment
#         label = stats.mode(df['label'][i: i + frame_size])[0][0]
#         frames.append([x, y, z])
#         labels.append(label)
#
#     # Bring the segments into a better shape
#     frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
#     labels = np.asarray(labels)
#
#     return frames, labels

def prep_train_data(files, frame_size, hop_size, N_FEATURES):
    data = pd.DataFrame()
    for i in files:
        read1 = readFile(i)
        df1 = pd.DataFrame(read1)
        df1 = df1.drop('tfps', axis=1)
        if "swim" in i:
            df1['activity'] = "Swimming"
        elif "drown" in i:
            df1['activity'] = "Drowning"
        else:
            print('Wrong name')
        data = pd.concat([data, df1])

    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')
    labels = data['activity'].value_counts()
    def round_down(n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier
    lab = min(labels)
    val = int(round_down(lab, -2))

    drowning = data[data['activity'] == 'Drowning'].head(val)
    swimming = data[data['activity'] == 'Swimming'].head(val)

    balanced_data = pd.DataFrame()
    balanced_data = pd.concat([drowning, swimming])
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled = pd.DataFrame(data=X, columns=['x', 'y', 'z'])
    scaled['label'] = y.values

    df = scaled
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    X = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    y = np.asarray(labels)

    # X, y = get_frames(scaled, frame_size, hop_size, N_FEATURES)
    # print('first =', X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    print("shape=", X_train[0].shape, X_test[0].shape)

    data08 = int(round(X.shape[0] * 0.8))
    data02 = int(round(X.shape[0] * 0.2))
    print(data08, data02)
    X_train = X_train.reshape(data08, frame_size, N_FEATURES, 1)
    X_test = X_test.reshape(data02, frame_size, N_FEATURES, 1)

    print(X_train[0].shape, X_test[0].shape)

    return X_train, X_test, y_train, y_test

def prep_test_data(file, frame_size, hop_size, N_FEATURES):
    read1 = readFile(file)
    df = pd.DataFrame(read1)
    df = df.drop('tfps', axis=1)
    data = df
    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')
    df = data

    X = df[['x', 'y', 'z']]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    df = scaled_X

    frames = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        frames.append([x, y, z])

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)

    frames = frames.reshape(frames.shape[0], frame_size, N_FEATURES, 1)

    return frames


# files = [fsDataswim1, fdDatadrown1]


# X_train, X_test, y_train, y_test = prep_train_data(files)
#
# print(X_train, X_test, y_train, y_test)






