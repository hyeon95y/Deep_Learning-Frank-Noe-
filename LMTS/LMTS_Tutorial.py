# REFERENCE
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# CHECK VERSION OF PYTHON FOR ATOM
import sys
print("Version ",sys.version)

# IMPORT MODULES
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 5000
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=3,
          batch_size=64
          )

# Final evaluation of the models
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# save model

from keras.models import load_model
filename = 'test.h5'
model.save(filename)
model = load_model(filename)
