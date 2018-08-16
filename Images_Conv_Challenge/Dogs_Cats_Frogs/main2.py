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
epochs = 25
num_classes = 3

path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/'
with np.load(path + 'prediction-challenge-02-data.npz') as fh:
    
    #기본적인 데이터 불러오는 과정
    data_x = fh['data_x']
    data_y = fh['data_y']
    test_x = fh['test_x']
    
    #채널을 맨 뒤로 빼줌
    data_x = np.transpose(data_x, (0, 2,3,1))
    test_x = np.transpose(test_x, (0, 2,3,1))
    
    #전체 개수를 센다
    number_datapoints = data_x.shape[0]

    #인덱스의 어레이를 만들고 셔플함
    indexes = np.arange(number_datapoints)
    np.random.shuffle(indexes)  # What I misesed

    #트레이닝과 테스트 셋으로 data_x, data_y를 나누기 위한 변수 
    length_train = int(number_datapoints*0.9)
    length_vali = number_datapoints - length_train
    
    x_train = data_x[indexes[:length_train]]
    y_train = data_y[indexes[:length_train]]
    x_test = data_x[indexes[length_train:]]
    y_test = data_y[indexes[length_train:]]
    x_predict = test_x
    
    #정확히는 모르겠지만 keras에서 잘 받아들일수 있게 reshape해주는 과정 
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
    
    
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(x_train[0])
    print(y_train[0])
    
    
    