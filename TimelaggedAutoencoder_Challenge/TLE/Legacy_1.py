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

# MAKE MODEL1
# SET TO TEST
timeset = [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000,
           20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000]
modelset = [None]
from keras.layers import Dense, Dropout
from keras.models import Sequential
'''
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
'''



# DECLARE HIST ARRAY TO SAVE DATA
# i : timeset, j : modelset
'''
hist = [None]*len(timeset)
for i in range (0, len(hist)) :
    hist[i] = [None]*len(modelset)
'''
    
# TRAIN MODEL WITH DIFFERENT SETS
for i in range(0, len(timeset)) :
    for j in range(0, len(modelset)) : 
        # MAKE TIME-LAGGED DATASET
        outputX, outputY = make_time_lagged_dataset(data_x, timeset[i])
        outputX, outputY = svd_whiten(outputX), svd_whiten(outputY)
        
        
        model1 = Sequential()
        # MAKE model1 FOLLOWS model1 COMPLEXITY VARIABLE

        # D = 3
        model1.add(Dense(units=3, input_dim=3, activation='relu'))
        model1.add(Dropout(0.25))
        for l in range (0, j) :
            model1.add(Dense(units=3, activation='relu'))
            model1.add(Dropout(0.25))
        # D = 2
        for l in range (0, j+1) :
            model1.add(Dense(units=2, activation='relu'))
            model1.add(Dropout(0.25))
        # D = 1
        model1.add(Dense(units=1, activation='relu'))
        #model1.add(Dropout(0.25))
        # D = 2
        for l in range (0, j+1) :
            model1.add(Dense(units=2, activation='relu'))
            model1.add(Dropout(0.25))
        # D = 3
        for l in range (0, j) :
            model1.add(Dense(units=3, activation='relu'))
            model1.add(Dropout(0.25))
        model1.add(Dense(units=3, activation='relu'))
        # SUMMARY
        print('* ',i, ', ',  j, '번째 모델')
        print(model1.summary())
        
        # TRAINING
        model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist = model1.fit(outputX
                                     , outputY
                                     , nb_epoch=100
                                     , batch_size=250
                                     )
        filename = 'model1-' + str(i) + str(j) + '.h5'
        model1.save(filename)
        
        # SAVE HISTORY
        import pickle
        filename = 'model1-hist-' + str(i) + str(j)+ '.p'
        with open(filename, 'wb') as file:    # hello.txt 파일을 바이너리 쓰기 모드(rb)로 열기
            pickle.dump(hist.history, file)
            
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

        # CHECK OUTPUT OF MODEL2
            
            
        prediction = model2.predict(data_x)
            
        x = np.zeros(len(data_x))
        y = np.zeros(len(data_x))
            
        for l in range(0, len(data_x)) :
            x[l] = l
            y[l] = prediction[l][0]

        title = str(i) + str(j)
        print('* PLT ' + title)    
                
        if np.mean(prediction) != 0  :
            
            title = str(i) + str(j) 
            print('* PLT ' + title + 'passed')
            plt.title(title)
            plt.figure(figsize=(20, 10))
            #plt.plot(prediction[0:200])
                
                
            plt.scatter(x[0:1000], y[0:1000])
            fig = plt.gcf() #변경한 곳
            fig.savefig('graph-'+title+'-1000.png') #변경한 곳
                
            plt.scatter(x, y)
            fig = plt.gcf()
            fig.savefig('graph-'+title+'-all.png')
'''
# MODEL LOAD
from keras.models import load_model
for i in range(0, len(timeset)) :
    for j in range (0, len(modelset)) :
        print('* timeset : ', timeset[i], ' , modelset : ', j, ' 번째 실험 결과')
        filename = 'TLATEST' + str(i) + str(j) + '.h5'
        model1 = load_model(filename)
        
        # MAKE NEW MODEL(MODEL2) WHICH WILL INHERIT WEIGHTS FROM TRAINED MODEL
        model2 = Sequential()
        model2.add(Dense(units=3, input_dim=3, activation='relu'))
        for k in range (0, j) :
            model2.add(Dense(units=3, activation='relu'))
        for k in range (0, j+1) :
            model2.add(Dense(units=2, activation='relu'))
        model2.add(Dense(units=1, activation='relu'))
        #print(model2.summary())
        
        # DUPLICATING WEIGHTS
        for k in range(0, j+1) :
            model2.layers[k].weight = model1.layers[k].get_weights()
        
        
        # MAKE PREDICTION
        model2_prediction = model2.predict(svd_whiten(data_x))
        # CHECK IT ROUGHLY
        #for k in range(0, 10) :
        #    print(prediction_new[k])
        
            
        # MAKE MODEL3
        validation_x_converted = model2.predict(svd_whiten(validation_x))
        model3 = Sequential()
        model3.add(Dense(units=1, input_dim=1, activation='relu'))
        model3.add(Dense(units=10, activation='relu'))
        model3.add(Dropout(0.25))
        model3.add(Dense(units=10, activation='relu'))
        model3.add(Dense(units=4, activation='softmax'))
        
        # SPLIT VALIDATION DATA AS A TRAINING DATA AND VALIDATION DATA
        (x_train, y_train) = (validation_x_converted[:800], validation_y[:800])
        (x_test, y_test) = (validation_x_converted[800:], validation_y[800:])
        x_train = x_train.reshape(800, 1).astype('float32')
        x_test = x_test.reshape(200,1).astype('float32')
        # CATEGORIZE
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        
        # CHECK DATATYPE
        print('* x_train ', x_train.shape)
        print('* y_train', y_train.shape )
        print('* x_test', x_test.shape)
        print('* y_test', y_test.shape)
        print(x_train[0])
        print(y_train[0])
        
        # TRAIN MODEL3
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_err', verbose=1)
        model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist = model3.fit(x_train, y_train
                                     , nb_epoch=30
                                     , batch_size=20
                                     , callbacks=[early_stopping]
                                     , validation_data= (x_test, y_test)
                                     )
        # SAVE MODEL3
        filename = 'model3' + str(i) + str(j) + '.h5'
        model3.save(filename)
        
        # LOAD MODEL3
        filename = 'model3' + str(i) + str(j) + '.h5'
        model3 = load_model(filename)
        
        # EVALUATE MODEL 2
        model3_prediction = model3.predict(model2_prediction)
        for k in range(0, 10) :
            print(model3_prediction[k])
'''     
        
        
        
        
        
        
        
