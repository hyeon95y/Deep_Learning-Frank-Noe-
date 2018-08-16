# import modules
import numpy as np
import keras
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Embedding

# fix random seed for reproducibility
np.random.seed(7)

'''
def pad_sequences(input) :
    output = np.zeros(shape=(len(input), max_sequence_length), dtype=str)
    for i in range (0, len(input)) :
        temp = np.array(list(input[i]))
        output[i][:len(temp)] = temp
    return output
'''

def cut_sequences(input) :
    output = np.zeros(shape=(len(input), sequence_length), dtype=str)
    for i in range (0, len(input)) :
        temp = np.array(list(input[i]))
        output[i][:sequence_length] = temp[:sequence_length]
    return output

def from_str_to_float(input) :
    output = np.zeros(shape=(len(input), sequence_length), dtype=float)
    for i in range(0, len(input)) :
        for j in range(0, len(input[0])) :
            output[i][j] = ACGT(input[i][j])
    return output

def ACGT(x):
    return {'A': '1', 'C': '2', 'T': '3', 'G' : '4'}.get(x, '0')



# load data
with np.load(os.path.join(os.path.dirname(__file__), "rnn-challenge-data.npz")) as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    val_x = fh['val_x']
    val_y = fh['val_y']
    test_x = fh['test_x']

    '''
    # pad input pad_sequences
    max_sequence_length = 2000
    str_data_x = pad_sequences(data_x)
    str_val_x = pad_sequences(val_x)
    str_test_x = pad_sequences(test_x)
    '''

    # instead of padding, cut it as the shortest length
    sequence_length = 400
    str_data_x = cut_sequences(data_x)
    str_val_x = cut_sequences(val_x)
    str_test_x = cut_sequences(test_x)

    # change str->float, ''->zero
    x_train = from_str_to_float(str_data_x)
    x_test = from_str_to_float(str_val_x)
    x_predict = from_str_to_float(str_test_x)

    # categorize labels
    num_classes = 5
    y_train = keras.utils.to_categorical(data_y, num_classes)
    y_test = keras.utils.to_categorical(val_y, num_classes)

# create the dense network models
triplet_code = 3
num_aminoacid = 20
model = Sequential()
model.add(Embedding(num_aminoacid, triplet_code, input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# train
hist = model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=1000,
          batch_size=64,
          verbose = 1
          )

# final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# save/load model
from keras.models import load_model
filename = 'ACGT.h5'
#model.save(filename)
model = load_model(filename)

# see the history
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

# make a prediction
prediction = model.predict(x_predict)
print(prediction)

# save prediction as a required format

prediction_submit = np.zeros(len(prediction))

# reduce the dimension
for i in range(0, len(prediction)) :
    maxvalue = np.max(prediction[i])
    index = np.where(prediction[i] == maxvalue)
    prediction_submit[i] = index[0]

# make sure that you have the right format
assert prediction_submit.ndim == 1
assert prediction_submit.shape[0] == 250
print(prediction_submit)

# save
np.save('prediction.npy', prediction_submit)
