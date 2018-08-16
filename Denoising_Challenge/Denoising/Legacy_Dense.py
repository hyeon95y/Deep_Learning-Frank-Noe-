import numpy as np
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28

path = '/Users/HyeonWoo/Library/Mobile Documents/com~apple~CloudDocs/University/2018-1/';
with np.load(path + 'denoising-challenge-01-data.npz') as fh:
    training_images_clean = fh['training_images_clean']
    validation_images_noisy = fh['validation_images_noisy']
    validation_images_clean = fh['validation_images_clean']
    test_images_noisy = fh['test_images_noisy']
    
# REORDER : PUT COLOR CHANNEL AS LAST
    training_images_clean = np.transpose(training_images_clean, (0, 2,3,1))
    validation_images_noisy = np.transpose(validation_images_noisy, (0, 2,3,1))
    validation_images_clean = np.transpose(validation_images_clean, (0, 2,3,1))
    test_images_noisy = np.transpose(test_images_noisy, (0, 2,3,1))
# RESHAPE
    training_images_clean = np.reshape(training_images_clean, (len(training_images_clean), 28, 28))
    validation_images_noisy = np.reshape(validation_images_noisy, (len(validation_images_noisy), 28, 28))
    validation_images_clean = np.reshape(validation_images_clean, (len(validation_images_clean), 28, 28))
    test_images_noisy = np.reshape(test_images_noisy, (len(test_images_noisy), 28, 28))
# RESHAPE 2
    training_images_clean = np.reshape(training_images_clean, (len(training_images_clean), 784))
    validation_images_noisy = np.reshape(validation_images_noisy, (len(validation_images_noisy), 784))
    validation_images_clean = np.reshape(validation_images_clean, (len(validation_images_clean), 784))
    test_images_noisy = np.reshape(test_images_noisy, (len(test_images_noisy), 784))
# PUT NOISE ON ORIGINAL DATA
    noise_factor=0.4
    training_images_noisy = training_images_clean + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_images_clean.shape)
    training_images_noisy = np.clip(training_images_noisy, 0., 1.)







# TRAINING DATA: CLEAN
# 1. INDEX: IMAGE SERIAL NUMBER (20000)
# 2. INDEX: COLOR CHANNEL (1)
# 3/4. INDEX: PIXEL VALUE (28 x 28)
print('* SHAPE : TRAINING_DATA')
print(training_images_clean.shape, training_images_clean.dtype)

print('* SHAPE : VALLIDATION DATA : CLEAN + NOISY')
# VALIDATION DATA: CLEAN + NOISY
print(validation_images_clean.shape, validation_images_clean.dtype)
print(validation_images_noisy.shape, validation_images_noisy.dtype)

print('* SHAPE : TEST DATA : NOISY')
# TEST DATA: NOISY
print(test_images_noisy.shape, test_images_noisy.dtype)

# RANDOM CODE TO CHECK SOMETHING
'''
whatiwanttosee = test_images_noisy
testindex = 3
print(whatiwanttosee.shape)
print(whatiwanttosee[testindex].shape)
plt.imshow(whatiwanttosee[testindex], cmap='gray')
plt.show()
'''
# RANDOM CODE TO CHECK SOMETHING

# TRAIN MODEL ON training_images_clean
# 검색해보고 모델 성능 개선할것 
# 1. DENSE NETWORK
# Early Stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', verbose=1)


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(784, activation='relu'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(training_images_clean, training_images_clean,
          nb_epoch=300,
          batch_size=256,
          #callbacks=[early_stopping],
          verbose=1,
          validation_data = (validation_images_noisy, validation_images_clean))


# CHECK YOUR MODEL USING (validation_images_clean, validation_images_noisy)
score = model.evaluate(validation_images_clean, validation_images_noisy, verbose=0)
print(score)
score2 = model.evaluate(validation_images_noisy, validation_images_clean, verbose=0)
print(score2)

# MODEL SAVE - FOR CODING
model.save('denoising_dense.h5')

# MODEL LOAD - FOR CODING
from keras.models import load_model
model = load_model('denoising_dense.h5')

# DENOISE IMAGES (test_images_clean) USING test_images_noisy
test_images_clean = model.predict(test_images_noisy)
print(test_images_noisy.shape)
print(test_images_clean.shape)
test_images_clean = np.reshape(test_images_clean, (len(test_images_clean), 28, 28))
print(test_images_clean.shape)

plt.imshow(test_images_clean[1], cmap='gray')
plt.show()
plt.imshow(test_images_clean[2], cmap= 'gray')
plt.show()
plt.imshow(test_images_clean[3], cmap= 'gray')
plt.show()
plt.imshow(test_images_clean[4], cmap= 'gray')
plt.show()
plt.imshow(test_images_clean[5], cmap= 'gray')
plt.show()


print('* USING DENOISE, CLEAR TEST SET TO CHECK REMAINING NOISE : WRONG')
test_images_noisy = np.reshape(test_images_noisy, (len(test_images_noisy), 1, 28, 28))
test_images_clean = np.reshape(test_images_clean, (len(test_images_clean), 1, 28, 28))
from numpy import linalg as LA
for i in range (0, 10) :
    print(LA.norm(test_images_noisy[i]-test_images_clean[i]))
print('IN AVERAGE : ', LA.norm(test_images_noisy-test_images_clean))

print('* USING CLEAR, NOISE, DENOISE VALIDATION SET TO CHECK REMAINING NOISE')
validation_images_denoised = model.predict(validation_images_noisy)
validation_images_clean = np.reshape(validation_images_clean, (len(validation_images_clean), 1, 28, 28))
validation_images_denoised = np.reshape(validation_images_denoised, (len(validation_images_denoised), 1, 28, 28))
for i in range (0,10) :
    print(LA.norm(validation_images_denoised[i] - validation_images_clean[i]))



# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert test_images_clean.ndim == 4
assert test_images_clean.shape[0] == 2000
assert test_images_clean.shape[1] == 1
assert test_images_clean.shape[2] == 28
assert test_images_clean.shape[3] == 28

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('test_images_clean.npy', test_images_clean)



#matplotlib inline

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


# Q1. For this case, which one is better? Conv? Dense? 
#    I learned that Conv has less parameters, so it should be past, but in my experience, Dense was always faster

# Q2. What about applying BatchNormalization and Dropout?

# Q3. I'm thinking about reading this. Is it related with this? Otherwise, I have to try all combination of sigma.

# Q4. How can I do this if noise is completely random? For this case, at least there is a information that noise is genearted by Gaussian
