def cnn_main(DIMENSIONS, MAX_SCALE,x_train,y_train,x_test, y_test):

    import keras
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras.callbacks import History

    import matplotlib.pyplot as plt


    history = History()


    img_x = MAX_SCALE
    img_y = MAX_SCALE
    img_z = DIMENSIONS
    input_shape = (img_x, img_y, img_z)
    print('input_shape', input_shape)

    num_classes = DIMENSIONS

    ''' NOT SURE WHAT batch_size IS '''
    batch_size = 16
    epochs = 10

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    # Error here
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

    model.save('Jakes_CNN_model.h5')

    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    return test_score, train_score


'''
*** Epoch 1/10
*** 5000/5000 [==============================] - 235s 47ms/step - loss: 0.3963 - acc: 0.8876 - val_loss: 0.6006 - val_acc: 0.8780
*** Epoch 2/10
*** 5000/5000 [==============================] - 228s 46ms/step - loss: 0.1939 - acc: 0.9282 - val_loss: 0.3952 - val_acc: 0.8880
*** Epoch 3/10
*** 5000/5000 [==============================] - 224s 45ms/step - loss: 0.1347 - acc: 0.9434 - val_loss: 0.4367 - val_acc: 0.9100
*** Epoch 4/10
*** 5000/5000 [==============================] - 228s 46ms/step - loss: 0.1971 - acc: 0.9334 - val_loss: 0.2662 - val_acc: 0.9320
*** Epoch 5/10
*** 5000/5000 [==============================] - 231s 46ms/step - loss: 0.1134 - acc: 0.9544 - val_loss: 0.2131 - val_acc: 0.9320
*** Epoch 6/10
*** 5000/5000 [==============================] - 230s 46ms/step - loss: 0.1285 - acc: 0.9520 - val_loss: 0.2014 - val_acc: 0.9440
*** Epoch 7/10
*** 5000/5000 [==============================] - 232s 46ms/step - loss: 0.1339 - acc: 0.9532 - val_loss: 0.2884 - val_acc: 0.9300
*** Epoch 8/10
*** 5000/5000 [==============================] - 237s 47ms/step - loss: 0.1503 - acc: 0.9488 - val_loss: 0.3181 - val_acc: 0.9340
*** Epoch 9/10
*** 5000/5000 [==============================] - 250s 50ms/step - loss: 0.1247 - acc: 0.9504 - val_loss: 0.2403 - val_acc: 0.9460
*** Epoch 10/10
*** 5000/5000 [==============================] - 238s 48ms/step - loss: 0.1578 - acc: 0.9508 - val_loss: 0.2133 - val_acc: 0.9300
*** Train loss: 0.11115437872409821, Train accuracy: 0.959
*** Test loss: 0.21326758581399918, Test accuracy: 0.93
'''
