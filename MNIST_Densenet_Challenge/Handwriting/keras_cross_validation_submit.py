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


# 3. CREATE DATASET FOR CROSS VALIDATION

datasize = 20000
k = 5

x_train = [None] * k
y_train = [None] * k
x_test = [None] * k
y_test = [None] * k

# FOR CROSS VALIDATION
for i in range (1, k+1) :
    start = int(datasize * ((i-1)/k))
    end = int(datasize * (i/k))
    if i==1 : # THE FIRST ONE IS TEST DATA
        
        x_test[i-1] = data_x[ : end]
        y_test[i-1] = data_y[ : end]
        x_train[i-1] = data_x[ end : ]
        y_train[i-1] = data_y[ end : ]
        
    elif i==k : # THE LAST ONE IS TEST DATA
        x_test[i-1] = data_x[ start : ]
        y_test[i-1] = data_y[ start : ]
        x_train[i-1] = data_x[ : start ]
        y_train[i-1] = data_y[ : start ]

    else :
        x_test[i-1] = data_x[ start : int(datasize * (i/k))]
        y_test[i-1] = data_y[ start : int(datasize * (i/k))]
        x_train[i-1] = np.concatenate((data_x[ : start ], data_x[ end : ]))
        y_train[i-1] = np.concatenate((data_y[ : start ], data_y[ end : ]))
        
    x_train[i-1] = x_train[i-1].reshape(int(datasize*(k-1)/k), 784).astype('float32') / 255.0
    x_test[i-1] = x_test[i-1].reshape(int(datasize*(1/k)), 784).astype('float32') / 255.0
    y_train[i-1] = np_utils.to_categorical(y_train[i-1])
    y_test[i-1] = np_utils.to_categorical(y_test[i-1])
    print(x_train[i-1].shape, x_train[i-1].dtype)
    print(y_train[i-1].shape, y_train[i-1].dtype)
    print(x_test[i-1].shape, x_test[i-1].dtype)
    print(y_test[i-1].shape, y_test[i-1].dtype)
# x_test, y_test : 0 ~ k-1 
# x_train, y_train : 0 ~ k-1

# FOR SUBMISSION
x_test_assign = test_x
x_test_assign = x_test_assign.reshape(2000, 784).astype('float32') / 255.0

# FOR COMPARSION
error = [None] * k

# REPEAT K TIMES FOR CROSS-VALIDATION
for i in range (1, k+1) :
    # 4. CONSIST MODEL
    model = Sequential()
    model.add(Dense(units=64, input_dim=28*28, activation='relu'))
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
    print("TRAIN THE MODEL")
    print(x_train[i-1].shape, x_train[i-1].dtype)
    print(y_train[i-1].shape, y_train[i-1].dtype)
    print(x_test[i-1].shape, x_test[i-1].dtype)
    print(y_test[i-1].shape, y_test[i-1].dtype)
    hist = model.fit(x_train[i-1], y_train[i-1], epochs=50, batch_size=20)

    # 8. EVALUATE THE MODEL
    loss_and_metrics = model.evaluate(x_test[i-1], y_test[i-1], batch_size=20)
    print('\n## evaluation loss and_metrics ## ', i)
    print(loss_and_metrics)
    error[i-1] = loss_and_metrics

print('## errors ##')
for i in range (1, k+1) :
    print(error[i-1])

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
    
#for i in range(0, 2000) :
#    print(yhat_submit[i])
prediction = yhat_submit    
 
#print(prediction.ndim)
#print(prediction.shape[0])

# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert prediction.ndim == 1
assert prediction.shape[0] == 2000

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('prediction.npy', prediction)
