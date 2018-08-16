import numpy as np
import matplotlib.pyplot as plt
from TimeLaggedAutoencoder.TimeLaggedAutoencoder_Legacy import making_time_lagged_dataset

# LOAD DATA
path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/';
with np.load(path + 'dimredux-challenge-01-data.npz') as fh:
    print(fh.files)
    data_x = fh['data_x']
    validation_x = fh['validation_x']
    validation_y = fh['validation_y']
    
    # CHECK DATA 
    print('data_x.shape : ', data_x.shape)
    print('validation_x.shape : ', validation_x.shape)
    print('validation_y.shape : ', validation_y.shape)

# WHITENING
def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)
    return X_white

# MAKE TIME-LAGGED DATASET
def make_time_lagged_dataset(data, tau):
    # CUT IT DEPENDS ON 'TIME' VARIABLE
    outputX = data[0:len(input)-tau]
    outputY = data[tau : len(input)]
    return outputX, outputY

# SET TO TEST
timeset = [2,3]
modelset = [None, None]
from keras.layers import Dense, Dropout
from keras.models import Sequential

for i in range (0,len(modelset)) : 
    modelset[i] = Sequential()
    # D = 3
    modelset[i].add(Dense(units=3, input_dim=3, activation='relu'))
    modelset[i].add(Dropout(0.5))
    for j in range (0, i) :
        modelset[i].add(Dense(units=3, activation='relu'))
        modelset[i].add(Dropout(0.5))
    # D = 2
    for j in range (0, i+1) :
        modelset[i].add(Dense(units=2, activation='relu'))
        modelset[i].add(Dropout(0.5))
    # D = 1
    modelset[i].add(Dense(units=1, activation='relu'))
    modelset[i].add(Dropout(0.5))
    # D = 2
    for j in range (0, i+1) :
        modelset[i].add(Dense(units=2, activation='relu'))
        modelset[i].add(Dropout(0.5))
    # D = 3
    for j in range (0, i) :
        modelset[i].add(Dense(units=3, activation='relu'))
        modelset[i].add(Dropout(0.5))
    modelset[i].add(Dense(units=3, activation='relu'))
    # SUMMARY
    print('* ', i, '번째 모델')
    print(modelset[i].summary())



# DECLARE HIST ARRAY TO SAVE DATA
# i : timeset, j : modelset
hist = [None]*len(timeset)
for i in range (0, len(hist)) :
    hist[i] = [None]*len(modelset)
    
# TRAIN MODEL WITH DIFFERENT SETS
for i in range(0, len(timeset)) :
    for j in range(0, len(modelset)) : 
        # MAKE TIME-LAGGED DATASET
        outputX, outputY = making_time_lagged_dataset(data_x, timeset[i])
        outputX, outputY = svd_whiten(outputX), svd_whiten(outputY)
        # TRAINING
        modelset[j].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist[i][j] = modelset[j].fit(outputX
                                     , outputY
                                     , nb_epoch=30
                                     , batch_size=250
                                     )
        modelset[j].save('TLATEST',i,j,'.h5')

# MODEL LOAD
from keras.models import load_model
for i in range(0, len(timeset)) :
    for j in range (0, len(modelset)) :
        print('* ', i, ' , ', j, ' 번째 실험 결과')
        modelset[i][j] = load_model('TLATEST',i,j,'.h5')
        
        # MAKE NEW MODEL WHICH WILL INHERIT WEIGHTS FROM TRAINED MODEL
        newmodel = Sequential()
        newmodel.add(Dense(units=3, input_dim=3, activation='relu'))
        for k in range (0, len(modelset)) :
            newmodel.add(Dense(units=3, activation='relu'))
        for k in range (0, len(modelset)+1) :
            newmodel.add(Dense(units=2, activation='relu'))
        newmodel.add(Dense(units=1, activation='softmax'))
        print(newmodel.summary())
        
        prediction = newmodel.predict(svd_whiten(data_x))
        for k in range(0, 10) :
            print(prediction[k])
        
        
        
        
        
