import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

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
    
    # SPLIT VALIDATION DATA AS A TRAINING DATA AND VALIDATION DATA
    (x_train, y_train) = (validation_x[:800], validation_y[:800])
    (x_test, y_test) = (validation_x[800:], validation_y[800:])
    x_train = x_train.reshape(800, 3).astype('float32')
    x_test = x_test.reshape(200,3).astype('float32')
    # CATEGORIZE
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

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
    outputX = data[0:len(data)-tau]
    outputY = data[tau : len(data)]
    return outputX, outputY



# HYPERPARAMETERS THROUGH ALL CODE
timeset = [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
model1 = [None, None, None, None, None]
model3 = [None, None, None]
model3_len = len(model3)

from keras.layers import Dense, Dropout
from keras.models import Sequential

# MAKE MODEL1
for i in range (0,len(model1)) : 
    model1[i] = Sequential()
    # D = 3
    model1[i].add(Dense(units=3, input_dim=3, activation='relu'))

    for j in range (0, i) :
        model1[i].add(Dense(units=3, activation='relu'))
        model1[i].add(Dropout(0.25))
    # D = 2
    for j in range (0, i+1) :
        model1[i].add(Dense(units=2, activation='relu'))
        model1[i].add(Dropout(0.25))
    # D = 1
    model1[i].add(Dense(units=1, activation='relu'))
    model1[i].add(Dropout(0.25))
    # D = 2
    for j in range (0, i+1) :
        model1[i].add(Dense(units=2, activation='relu'))
        model1[i].add(Dropout(0.25))
    # D = 3
    for j in range (0, i) :
        model1[i].add(Dense(units=3, activation='relu'))
        model1[i].add(Dropout(0.25))
    model1[i].add(Dense(units=3, activation='relu'))
    # SUMMARY
    print('* MODEL1 : ', i, '번째 모델')
    print(model1[i].summary())

# MAKE MODEL3
for i in range (0,len(model3)) : 
    model3[i] = Sequential()
    # INPUT
    model3[i].add(Dense(units=1, input_dim=1, activation='relu'))
    
    # DENSE LAYERS
    for j in range (0, i+1) :
        model3[i].add(Dense(units=10, activation='relu'))
        model3[i].add(Dropout(0.25))

    # OUTPUT
    model3[i].add(Dense(units=4, activation='softmax'))
    # SUMMARY
    print('* MODEL3 : ', i, '번째 모델')
    print(model3[i].summary())



# DECLARE HIST ARRAY TO SAVE DATA
# i : timeset, j : model1
hist = [None]*len(timeset)
for i in range (0, len(model1)) :
    hist[i] = [None]*len(model1)
    
    for j in range (0, model3_len) :
        hist[i][j] = [None]*model3_len

    
# TRAIN MODEL WITH DIFFERENT SETS
from keras.models import load_model
# i : time-period
for i in range(0, len(timeset)) :
# j : model1 complexity
    for j in range(0, len(model1)) : 
        
        # TRAINING TIME-LAGGED AUTOENCODER FOR (i, j)
        
        # MAKE TIME-LAGGED DATASET
        outputX, outputY = make_time_lagged_dataset(data_x, timeset[i])
        outputX, outputY = svd_whiten(outputX), svd_whiten(outputY)
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        # TRAINING
        model1[j].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist = model1[j].fit(outputX
                                     , outputY
                                     , nb_epoch=100
                                     , batch_size=250
                                     , callbacks=[early_stopping]
                                     )
        #MODEL SAVE
        filename = 'model1-' + str(i) + str(j) + '.h5'
        model1[j].save(filename)
        
        # SAVE HISTORY
        import pickle
        filename = 'model1-hist-' + str(i) + str(j)+ '.p'
        with open(filename, 'wb') as file:    # hello.txt 파일을 바이너리 쓰기 모드(rb)로 열기
            pickle.dump(hist.history, file)
        
        # MODEL LOAD
        filename = 'model1-' + str(i) + str(j) + '.h5'
        model1 = load_model(filename)
'''
# k : model3 complexity
        for k in range(0, model3_len) :
            
            # TRAINING COMPLETE MODEL
                    
            # MAKE model2 WHICH WILL INHERIT WEIGHTS FROM MODEL1
            model2 = Sequential()
            model2.add(Dense(units=3, input_dim=3, activation='relu'))
            for l in range (0, j) :
                model2.add(Dense(units=3, activation='relu'))
            for l in range (0, j+1) :
                model2.add(Dense(units=2, activation='relu'))
            model2.add(Dense(units=1, activation='relu'))
            
            # DUPLICATING WEIGHTS
            for l in range(0, j+1) :
                model2.layers[l].weight = model1.layers[l].get_weights()
            
            # COMBINING MODEL2 MODEL3 TO MAKE MODEL4
            model2.trainable = False
            model4 = Sequential()
            model4.add(model2)
            model4.add(model3[k])
            print('* MODEL4 : ', i, ' , ', j, ' , ', k, '번째 모델')
            print(model4.summary())
            
            # TRAIN MODEL (i, j, k)
            early_stopping = EarlyStopping(monitor='val_err', patience=0, verbose=1, mode='auto')
            model4.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            history = model4.fit(x_train, y_train
                                    , nb_epoch=30
                                    , batch_size=20
                                    , callbacks=[early_stopping]
                                    , validation_data= (x_test, y_test)                                  
                              )
            
            #hist[i][j][k] = history
            
            # SAVE HISTORY
            import pickle
            filename = 'hist-' + str(i) + str(j) + str(k) + '.p'
            with open(filename, 'wb') as file:    # hello.txt 파일을 바이너리 쓰기 모드(rb)로 열기
                pickle.dump(history.history, file)
            
            # SAVE MODEL4
            filename = 'model4-' + str(i) + str(j) + str(k) + '.h5'
            model4.save(filename)
            
            # LOAD MODEL4
            filename = 'model4-' + str(i) + str(j) + str(k) + '.h5'
            model4 = load_model(filename)
            
            prediction = model4.predict(data_x)
            print('* PREDICTION FROM MODEL4 : ', i, ' , ', j, ' , ', k)
            
            prediction_submit = np.zeros(len(outputX))
            
            # CHANGE FORMAT AS (0,1,0,0)
            for l in range(0, len(outputX)) :
                maxvalue = np.max(prediction[l])
                indexmax = np.where(prediction[l] == maxvalue)
                prediction_submit[l] = indexmax[0]
            
            for l in range(0, 30) :
                print(prediction_submit[l])

'''
        
'''
# EVALUATE THE MODEL
loss_and_metrics = model4.evaluate(x_test, y_test, batch_size=20)
print('\n## evaluation loss and_metrics ##')
print(loss_and_metrics)
'''
            
            





        
        
        
        
        
        
        
