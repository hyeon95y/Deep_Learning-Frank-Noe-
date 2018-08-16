# 1. LOAD PACKAGES
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 2. LOAD DATAS FROM NUMPY
import numpy as np
import matplotlib.pyplot as plt
path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/';
with np.load(path + 'prediction-challenge-01-data.npz') as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']

# 3. CREATE DATASET
(x_train, y_train) = (data_x[:16000], data_y[:16000])
(x_test, y_test) = (data_x[16000:], data_y[16000:])
x_test_assign = test_x
x_train = x_train.reshape(16000, 784).astype('float32') / 255.0
x_test = x_test.reshape(4000, 784).astype('float32') / 255.0
x_test_assign = x_test_assign.reshape(2000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])

# 4. CONSIST MODEL
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 5. SET TRAINING METHOD
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. TRAIN THE MODEL
hist = model.fit(x_train, y_train, epochs=50, batch_size=100)

# 7. SEE HOW IT WORKS
print('\n## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['acc'])

# 8. EVALUATE THE MODEL
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)
print('\n## evaluation loss and_metrics ##')
print(loss_and_metrics)


# 9. SAVE AS THE REQUIRED FORMAT
xhat = x_test_assign
yhat = model.predict(xhat)
prediction = yhat
yhat_submit = np.zeros(2000)

# REDUCE THE DIMENSION OF YHAT AS REQURIED FORMAT
for i in range(0, 2000) :
    maxyhat = np.max(yhat[i])
    indexyhat = np.where(yhat[i] == maxyhat)
    #print('indexyhat : ', indexyhat[0])
    yhat_submit[i] = indexyhat[0]
    
prediction = yhat_submit    

# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert prediction.ndim == 1
assert prediction.shape[0] == 2000

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('prediction.npy', prediction)
