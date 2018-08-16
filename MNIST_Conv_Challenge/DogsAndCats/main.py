'''
Created on 2018. 5. 18.

@author: HyeonWoo
'''

# There are two ways to initialising a neural network. Sequence or Graph
from keras.models import Sequential
# Conv2D layer as I know
# For video, you should go 3D
from keras.layers import Conv2D
# MaxPooling2D is missing. Maybe the name is changed
# 영어로는 '통합시키다' 라는 의미
from keras.layers import MaxPooling2D
# Flattening is the process of converting all the resultant 2 dimensional arrays  
# into a "single long continuous linear vector"
from keras.layers import Flatten
# To perform the full connection of the neural network
# For now, I don't get the purpose of this step
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.labeled_tensor import batch

classifier = Sequential()
# number of filters : 32, shape of each filter : (3,3), input shape : width x height x color, activiation : ReLU
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
# to reduce the size  
classifier.add(MaxPooling2D(pool_size=(2,2)))
# taking 2D array and converting them to a one dimensional single vector
classifier.add(Flatten())
# I guess Flatten() itself doesn't contain any connection. So it should be connected by Dense(just a basic, normal thing I learned at the first)
classifier.add(Dense(units = 128, activation = 'relu'))
# go to Output with 'sigmoid'
classifier.add(Dense(units=1, activation = 'sigmoid'))

# Softmax for 'classification' -> 다 더해서 1이 되게 하고, 각각의 확률을 계산함
# Sigmoid -> 각각 0~1을 출력, 하나의 노드를 가지고 이진 분류에 사용되는듯

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Image Agumentations to avoid overfitting
path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/'
train_datagen = ImageDataGenerator(rescale = 1./244,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(path+'training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = train_datagen.flow_from_directory(path+'test_set',
                                             target_size = (64, 64),
                                             batch_size = 32,
                                             class_mode = 'binary')
# 여기서 어떤 이미지가 됐는지 보고싶은데 
#import matplotlib.pyplot as plt
'''
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)
print('wtf1')
from keras.models import load_model
print('wtf2')
classifier.save('dogs_and_cats.h5')
print('wtf3')
'''
'''
#save-train-save-train 
for i in range(25) :
    classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)
    classifier.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps)
    from keras.models import load_model
    classifier.save('dogs_and_cats.h5')
'''
classifier.fit(training_set[0], training_set[1],
          batch_size=100,
          epochs=25,
          verbose=1,
          validation_data=(test_set[0], test_set[1]))
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path + 'test_set/cats/cat.4066.jpg',
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result[0][0])
print(result)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else :
    predcition = 'cat'
    
print('result is : ', prediction)







