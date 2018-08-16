'''
Created on 2018. 5. 22.

@author: HyeonWoo
'''
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import keras

img_rows, img_cols = 32, 32
batch_size = 25
epochs = 40
num_classes = 3

path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/'
with np.load(path + 'prediction-challenge-02-data.npz') as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']
    print(data_x.shape, data_x.dtype)
    print(data_y.shape, data_y.dtype)
    print(test_x.shape, test_x.dtype)
    data_x = np.transpose(data_x, (0, 2,3,1))
    test_x = np.transpose(test_x, (0, 2,3,1))
    print(data_x.shape, data_x.dtype)
    print(data_y.shape, data_y.dtype)
    print(test_x.shape, test_x.dtype) 
    print(data_y)
    
    number_datapoints = data_x.shape[0]

    indexes = np.arange(number_datapoints)
    np.random.shuffle(indexes)  # What I misesed

    length_train = int(number_datapoints*0.9)
    length_vali = number_datapoints - length_train

    print('here come the indexes')
    #print(indexes)
    print(indexes.shape)
    #print(indexes[:length_train])
    print(indexes[:length_train].shape)
    x_train = data_x[indexes[:length_train]]
    y_train = data_y[indexes[:length_train]]
    print('\nx_train and blah blah \n')
    print(x_train.shape)
    print(y_train.shape)
    print(x_train.shape[0])
    print(y_train.shape[0])
    print('\n')

    x_test = data_x[indexes[length_train:]]
    y_test = data_y[indexes[length_train:]]
    x_predict = test_x
    print('x_test and blah blah')
    print(x_test.shape)
    print(y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    
    x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
    

    print(x_train.shape)
    print(x_test.shape)
    print(input_shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print(y_test)
    y_test = keras.utils.to_categorical(y_test, num_classes)
'''
import matplotlib.pyplot as plt
plt.imshow(x_train[10])
plt.title(y_train[10])
plt.show()
''' 
    
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_predict = x_predict.astype('float32')
#x_train /= 255
#x_test /= 255
 #   print('x_train shape:', x_train.shape)
 #   print(x_train.shape[0], 'train sampels')
 #   print(x_test.shape[0], 'test samples')
 #   print(x_predict.shape[0], 'rediction samples')

    
    
# TRAINING DATA: INPUT (x) AND OUTPUT (y)
# 1. INDEX: IMAGE SERIAL NUMBER (6000)
# 2. INDEX: COLOR CHANNELS (3)
# 3/4. INDEX: PIXEL VALUE (32 x 32)
print(data_x.shape, data_x.dtype)
print(data_y.shape, data_y.dtype)

# TEST DATA: INPUT (x) ONLY
print(test_x.shape, test_x.dtype)

# TRAIN MODEL ON data_x, data_y
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(loss=keras.losses.categorical_crossentropy, 
              #optimizer=keras.optimizers.Adadelta(),
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_data=(x_test, y_test)
          )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# PREDICT prediction FROM test_x

# 9. SAVE AS THE REQUIRED FORMAT
prediction = model.predict(x_predict)
yhat_submit = np.zeros(300)

# REDUCE THE DIMENSION OF YHAT AS REQURIED FORMAT
for i in range(0, 300) :
    maxyhat = np.max(prediction[i])
    indexyhat = np.where(prediction[i] == maxyhat)
    #print('indexyhat : ', indexyhat[0])
    yhat_submit[i] = indexyhat[0]
    
prediction = yhat_submit

# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert prediction.ndim == 1
assert prediction.shape[0] == 300

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('prediction6.npy', prediction)
