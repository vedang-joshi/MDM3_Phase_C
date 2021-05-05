
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics


#
# scaled_X = pd.read_csv('scaled_data.csv')
# swimDrown = pd.read_csv('swimDrown_data.csv')
#
def get_frames(df, frame_size, hop_size, N_FEATURES):


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
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

def prep_train_data(files):
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

    X, y = get_frames(scaled, frame_size, hop_size, N_FEATURES)
    print('first =', X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    print("shape=", X_train[0].shape, X_test[0].shape)

    data08 = int(round(X.shape[0] * 0.8))
    data02 = int(round(X.shape[0] * 0.2))
    print(data08, data02)
    X_train = X_train.reshape(data08, frame_size, N_FEATURES, 1)
    X_test = X_test.reshape(data02, frame_size, N_FEATURES, 1)

    print(X_train[0].shape, X_test[0].shape)

    return X_train, X_test, y_train, y_test


##MODEL

def build_model1(activation, input_shape):
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation = activation, input_shape = input_shape))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (2, 2), activation=activation))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation = activation))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))

    return model


def build_model2(activation, input_shape):
    #K.set_image_dim_ordering('th')
    model = Sequential()

    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, (5, 5), activation=activation, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(64, (5, 2), activation=activation, padding='same', kernel_initializer="he_normal"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())

    # 3 Full connected layer
    model.add(Dense(128, activation=activation, kernel_initializer="he_normal"))
    model.add(Dense(54, activation=activation, kernel_initializer="he_normal"))
    model.add(Dense(6, activation='softmax'))  # 6 classes

    # summarize the model
    print(model.summary())
    return model


def compile_and_fit_model(model, X_train, y_train, X_test, y_test, n_epochs):
    # compile the model
    model.compile(
        optimizer=Adam(learning_rate = 0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', 'accuracy'])

    # define callbacks
    callbacks = [
        ModelCheckpoint(filepath='best_model.h5', monitor='val_sparse_categorical_accuracy', save_best_only=True)]

    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        #batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))

    return model, history

# input_shape = X_train[0].shape
#
# cnn_model = build_model1("relu", input_shape)
# #cnn_model.save('initial_model.h5')
#
# model, history = compile_and_fit_model(cnn_model, X_train, y_train, X_test, y_test, 30)
#
# model.save('scaled_model.h5')


##VISULISE

def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

#plot_learningCurve(history, 15)


#Confusion Matrix

LABEL_NAMES = ['Drowning','Swimming'] #'Breast', 'Crawl'
def create_confusion_matrix(y_pred, y_test):
    # calculate the confusion matrix
    confmat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

    n_labels = len(LABEL_NAMES)
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # loop over data dimensions and create text annotations.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i, y=j, s=confmat[i, j], va='center', ha='center')

    # avoid that the first and last row cut in half
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

#model = load_model('initial_model.h5')
# make predictions for test data
# y_pred = model.predict_classes(X_test)
# # determine the total accuracy
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

#create_confusion_matrix(y_pred, y_test)


# prob_d = model.predict_proba(X_test)[:,0]
# prob_s = model.predict_proba(X_test)[:,1]
# print(prob_d, prob_s)

