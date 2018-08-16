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

model3 = [None, None, None]

from keras.layers import Dense, Dropout
from keras.models import Sequential


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



    
# TRAIN MODEL WITH DIFFERENT SETS
from keras.models import load_model
# i : time-period
# j : model1 complexity

        
'''
i = 12
j = 0  
# MODEL LOAD
filename = 'LEGACY4/model1-' + str(i) + str(j) + '.h5'
model1 = load_model(filename)
'''
i = 13
j = 0  
# MODEL LOAD
filename = 'LEGACY3/model1-' + str(i) + str(j) + '.h5'
model1 = load_model(filename)


            
# MAKE model2 WHICH WILL INHERIT WEIGHTS FROM MODEL1
model2 = Sequential()
model2.add(Dense(units=3, input_dim=3, activation='relu'))
#model2.add(Dropout(0.25))
for l in range (0, j) :
    model2.add(Dense(units=3, activation='relu'))
    #model2.add(Dropout(0.25))
for l in range (0, j+1) :
    model2.add(Dense(units=2, activation='relu'))
    #model2.add(Dropout(0.25))
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
print('* ' + title)    
  
  
import time 
now = time.gmtime(time.time())  
timenow = str(now.tm_hour) + str(now.tm_min) +  str(now.tm_sec)
                
if np.mean(prediction) != 0  :
            
    title = str(i) + str(j) 
    print('* ' + title + 'passed')
    plt.title(title)
    plt.figure(figsize=(20, 10))
    #plt.plot(prediction[0:200])
                
                
    plt.scatter(x[0:1000], y[0:1000])
    fig = plt.gcf() #변경한 곳
    fig.savefig(timenow + 'graph-'+title+'-1000.png') #변경한 곳
                
    plt.scatter(x, y)
    fig = plt.gcf()
    fig.savefig(timenow + 'graph-'+title+'-all.png')
    

# prediction을 K-Means clustering에 돌림
from sklearn.cluster import KMeans

'''
# Number of clusters
kmeans = KMeans(n_clusters=4)
# Fitting the input data
kmeans = kmeans.fit(prediction)
# Getting the cluster labels
y_classified = kmeans.predict(prediction)
# Centroid values
centroids = kmeans.cluster_centers_
# Comparing with scikit-learn centroids
'''

# Number of clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(prediction)
# Getting the cluster labels
y_classified = kmeans.labels_
# Comparing with scikit-learn centroids


print(y_classified)
print(y_classified.shape)
print(np.mean(y_classified))
plt.clf()
plt.scatter(x,y_classified)
fig = plt.gcf()
fig.savefig(timenow + 'clustered.png')

                
# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert y_classified.ndim == 1
assert y_classified.shape[0] == 100000


# AND SAVE EXACTLY AS SHOWN BELOW
title = timenow + 'prediction.npy'
np.save(title, y_classified)                
                
                
            

                
            
            
            
        
'''
# EVALUATE THE MODEL
loss_and_metrics = model4.evaluate(x_test, y_test, batch_size=20)
print('\n## evaluation loss and_metrics ##')
print(loss_and_metrics)
'''
            
            





        
        
        
        
        
        
        
