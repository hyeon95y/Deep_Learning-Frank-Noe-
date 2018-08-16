# 전체적인 구조 만들기(RNN SLIDES 참조)
# 가능하다면 RESIDUAL NETWORK 적용하기
# 도움을 줄수 있는 TRICKS 찾아서 적용하기




import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import keras
import os

num_classes = 5
max_length = 2000

with np.load(os.path.join(os.path.dirname(__file__), "rnn-challenge-data.npz")) as fh:
    print(fh.files)
    data_x = fh['data_x']
    data_y = fh['data_y']
    val_x = fh['val_x']
    val_y = fh['val_y']
    test_x = fh['test_x']
    print(data_x.shape)
    print(data_x[0].shape)
    print(data_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    print(test_x.shape)
    print(data_x[0])
    print(data_y[0])

    # shuffle the training data
    number_datapoints = data_x.shape[0]
    indexes = np.arange(number_datapoints)
    np.random.shuffle(indexes)

    # assign them
    x_train_temp = np.zeros(shape=(400,2000), dtype=str)
    x_train = np.zeros(shape=(400,2000))
    y_train = data_y
    x_test_temp = np.zeros(shape=(100, 2000), dtype=str)
    x_test = np.zeros(shape=(100, 2000))
    y_test = val_y

    # making list for each sequence x_train, x_test
    for i in range (0, len(x_train_temp)) :
        #x_train[i] = np.array(list(data_x[indexes[i]]))
        temp = np.array(list(data_x[indexes[i]]))
        x_train_temp[i][:len(temp)] = temp
        # A, C, G, T -> [-2, -1, 1, 2]
        for j in range (0, len(x_train[i])) :
            if x_train_temp[i][j] == 'A' :
                x_train[i][j] = int(-2)
            elif x_train_temp[i][j] == 'C' :
                x_train[i][j] = int(-1)
            elif x_train_temp[i][j] == 'G' :
                x_train[i][j] = 1
            elif x_train_temp[i][j] == 'T' :
                x_train[i][j] = 2
            else :
                x_train[i][j] = 0
    for i in range (0, len(x_test_temp)) :
        #x_test[i] = np.array(list(val_x[i]))
        temp = np.array(list(val_x[i]))
        x_test_temp[i][:len(temp)] = temp
        # A, C, G, T -> [-2, -1, 1, 2]
        for j in range (0, len(x_test[i])) :
            if x_test_temp[i][j] == 'A' :
                x_test[i][j] = int(-2)
            elif x_test_temp[i][j] == 'C' :
                x_test[i][j] = int(-1)
            elif x_test_temp[i][j] == 'G' :
                x_test[i][j] = 1
            elif x_test_temp[i][j] == 'T' :
                x_test[i][j] = 2
            else :
                x_test[i][j] = 0



    print(x_train.shape)
    print(x_train[0][0])
    print(x_train[0][1])

    # regularization


    # categorize labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # reshape
    x_train = np.reshape(x_train, (x_train.shape[0],  x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # input_shape
    print('* x_train.shape', x_train.shape)
    print('* x_train[0]', x_train[0])
    print('* y_train.shape', y_train.shape)
    print('* y_train[0]', y_train[0])
    print('* x_test.shape', x_test.shape)
    print('* x_test[0]', x_test[0])
    print('* y_test.shape', y_test.shape)
    print('* y_test[0]', y_test[0])

# 일단은 LSTM으로 시도해봄
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

model = Sequential()
model.add(LSTM(100, input_shape=(2000, 1)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train,
          epochs=500,
          batch_size=64,
          verbose=2,
          shuffle=False,
          validation_data=(x_test, y_test)
          )

# MODEL SAVE - FOR CODING
model.save('rnn_challenge.h5')

# MODEL LOAD - FOR CODING
from keras.models import load_model
model = load_model('rnn_challenge.h5')

# show visually
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
fig = plt.gcf()
fig.savefig('plt.png')

# model prediction
prediction = model.predict(x_test)
print(x_test)
