import numpy as np
import matplotlib.pyplot as plt

# IDEA
# 1. WHITENING?
# 2. TIME-LAGGED AUTOENCODER -> DIMENSION REDUCTION
# 3. APPLYING DEEP BLAH BLAH CLUSTERING


# LOAD DATA
path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/';
with np.load(path + 'dimredux-challenge-01-data.npz') as fh:
    print(fh.files)
    data_x = fh['data_x']
    validation_x = fh['validation_x']
    validation_y = fh['validation_y']
    
    # CHECK DATA 
    print(data_x.shape)
    print(validation_x.shape)
    print(validation_y.shape)
    
    print(data_x[0])
    '''
    data_x = np.reshape(data_x, (3, len(data_x)))
    for i in range (0, 3) :
        print(np.mean(data_x[i]))
        print(np.var(data_x[i]))

    data_x = np.reshape(data_x, (len(data_x), 3))
    '''
# 1. PROPER DATA WHITENING WILL HELP YOU  

# TRIAL 1
def whiten(X,fudge=1E-18):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white
'''
trial1 = whiten(data_x)
print('* TRIAL 1 : WHITENING')
print(trial1)
trial1 = np.reshape(trial1, (3, len(trial1)))
for i in range (0,3):
    print('* MEAN', i, ' : ', np.mean(trial1[i]))
    print('* VAR', i, ' : ', np.var(trial1[i]))
'''

# TRIAL 2
def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)
    return X_white
'''
trial2 = svd_whiten(data_x)
print('* TRIAL 2 : WHITENING')
print(trial2)
trial2 = np.reshape(trial2, (3, len(trial2)))
for i in range (0,3):
    print('* MEAN', i, ' : ', np.mean(trial2[i]))
    print('* VAR', i, ' : ', np.var(trial2[i]))
'''
# TRIAL 3
def another_whiten(X):
    # 입력 자료행렬 X의 크기는 [NxD]로 가정함 
    X -= np.mean(X, axis = 0) # 평균이 0 이도록 평행이동함 (중요함) 
    cov = np.dot(X.T, X) / X.shape[0] # 자료의 공분산 행렬을 얻음
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X, U) # decorrelate the data
    Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
    # 자료를 화이트닝 # 고유값으로 나눔 (특이값에 제곱근을 씌운) 
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    return Xwhite
'''
trial3 = another_whiten(data_x)
print('* TRIAL 3 : WHITENING')
print(trial3)
trial3 = np.reshape(trial3, (3, len(trial3)))
for i in range (0,3):
    print('* MEAN', i, ' : ', np.mean(trial3[i]))
    print('* VAR', i, ' : ', np.var(trial3[i]))
''' 
    
    
# TIME-LAGGED BLAH BLAH 
def making_time_lagged_dataset(input, time):
    
    
    # CUT IT DEPENDS ON 'TIME' VARIABLE
    outputX = input[0:len(input)-time]
    outputY = input[time : len(input)]
    #print(np.shape(outputX))
    #print(np.shape(outputY))
    '''
    # RESHAPE TO CALCULATE MEAN
    tempX = np.reshape(outputX, (3, len(outputX)))
    tempY = np.reshape(outputY, (3, len(outputY)))   
    X_mean, Y_mean = [0, 0, 0], [0, 0, 0]
    for i in range (0, 3) :
        X_mean[i] = np.mean(tempX[i])
        Y_mean[i] = np.mean(tempY[i])
        #print('* X_mean', i, ' :', X_mean[i])
        #print('* Y_mean', i, ' :', Y_mean[i])
    
    # MEAN-FREE
    for i in range (0, len(outputX)) :
        outputX[i][0] -= X_mean[0]
        outputX[i][1] -= X_mean[1]
        outputX[i][2] -= X_mean[2]
    for i in range (0, len(outputY)) :
        outputY[i][0] -= Y_mean[0]
        outputY[i][1] -= Y_mean[1]
        outputY[i][2] -= Y_mean[2]
    '''
    # SEE THE RESULT
    tempX = np.reshape(outputX, (3, len(outputX)))
    tempY = np.reshape(outputY, (3, len(outputY)))
    #for i in range (0, 3) :
        #print('X Mean ', i, ' : ', np.mean(tempX[i]))
        #print('Y Mean ', i, ' : ', np.mean(tempY[i]))
        
    return outputX, outputY
    
   
# THE ORIGINAL DATA WAS ALREADY MEAN-FREE IN THIS CASE
outputX, outputY = making_time_lagged_dataset(data_x, 2)
outputX_whitened = svd_whiten(outputX)
outputY_whitened = svd_whiten(outputY)

# CHECK WHITENED DATA
tempX = np.reshape(outputX_whitened, (3, len(outputX_whitened)))
tempY = np.reshape(outputY_whitened, (3, len(outputY_whitened)))
print('* CHECK WHITENED DATA')
for i in range (0,3):
    print('* X MEAN', i, ' : ', np.mean(tempX[i]))
    print('* X VAR', i, ' : ', np.var(tempX[i]))
    print('* Y MEAN', i, ' : ', np.mean(tempY[i]))
    print('* Y VAR', i, ' : ', np.var(tempY[i]))
print('* X SHAPE : ', np.shape(tempX))
print('* Y SHAPE : ', np.shape(tempY))


# TRAINING DATA
outputX_whitened = np.reshape(outputX_whitened, (len(outputX_whitened), 3))
outputY_whitened = np.reshape(outputY_whitened, (len(outputY_whitened), 3))
# CONV NETWORK
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, GaussianNoise, Flatten
from keras.models import Sequential, Model


# 더 작은모델에서부터 시작해서 늘려나갈것
model = Sequential()

model.add(Dense(units=3, input_dim=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=1, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=3, activation='relu'))
print(model.summary())



from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_acc', verbose=2)
model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit(outputX_whitened
               , outputY_whitened
               , nb_epoch = 100
               , batch_size = 250
               #, callbacks = [early_stopping]
               #, validation_data=(validation_x, validation_y)
               )

# MODEL SAVE - FOR CODING
model.save('TLA.h5')

# MODEL LOAD = FOR CODING
from keras.models import load_model
model = load_model('TLA.h5')

print(outputX_whitened)
print(outputY_whitened)
print(data_x)
print(model.predict(data_x))
print(svd_whiten(data_x))
print(model.predict(svd_whiten(data_x)))
result = model.predict(svd_whiten(data_x))
for i in range (0, len(data_x)) :
    print(result[i])
    

'''
# MAKE NEW MODEL TO ONLY USE ENCODE PART
newmodel = Sequential()
newmodel.add(Dense(units=3, input_dim=3, activation='relu'))
newmodel.add(Dense(units=2, activation='relu'))
newmodel.add(Dense(units=1, activation='relu'))

newmodel.layers[0].weight = model.layers[0].get_weights()
newmodel.layers[1].weight = model.layers[2].get_weights()
newmodel.layers[2].weight = model.layers[2].get_weights()
for i in range (0,3) :
    print(newmodel.layers[i].weight)
print(newmodel.summary())

check = model.predict(svd_whiten(data_x))
prediction = newmodel.predict(svd_whiten(data_x))
for i in range (0, 10) :
    print(prediction[i])
    print(check[i])
'    



# PRINT THE RESULT AS A GRPAH

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
'''











# 2. USE TIME-LAGGED AUTOENCODER TO DO DIMENSION reduction
# IT SHOULD BE ONE-DIMENSIONAL TIME SERIES AT THE END
# THAT 4 DIFFERENT STATES BECOME DISENTANGLED 



# 3. DISCRETIZE YOUR ONE-DIMENSIONAL REPRESENTATION AS {0,1,2,3}
# CLUSTERING