import numpy as np
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

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
    
    # CALCULATE THE NOISE
    total_var=0
    total_mean=0
    for i in range (0, len(test_images_noisy)) :
        total_var += np.var(test_images_noisy[i])
        total_mean += np.mean(test_images_noisy[i])
    print('* TOTAL VAR : ', total_var)
    print('* VAR IN AVERAGE : ', total_var/len(test_images_noisy))
    print('* TOTAL MEAN : ', total_mean)
    print('* MEAN IN AVERAGE : ', total_mean/len(test_images_noisy))
    noise_factor = total_var/len(test_images_noisy)
    
    # RESHAPE
    training_images_clean = np.reshape(training_images_clean, (len(training_images_clean), 28, 28, 1))
    validation_images_noisy = np.reshape(validation_images_noisy, (len(validation_images_noisy), 28, 28, 1))
    validation_images_clean = np.reshape(validation_images_clean, (len(validation_images_clean), 28, 28, 1))
    test_images_noisy = np.reshape(test_images_noisy, (len(test_images_noisy), 28, 28, 1))
    
    print(training_images_clean.shape)
    print(validation_images_noisy.shape)
    print(validation_images_clean.shape)
    print(test_images_noisy.shape)


# TRAINING DATA: CLEAN
# 1. INDEX: IMAGE SERIAL NUMBER (20000)
# 2. INDEX: COLOR CHANNEL (1)
# 3/4. INDEX: PIXEL VALUE (28 x 28)

# VALIDATION DATA: CLEAN + NOISY

# TEST DATA: NOISY

# TRAIN MODEL ON training_images_clean

# CONV NETWORK
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, GaussianNoise, Flatten
from keras.models import Sequential, Model


# 더 작은모델에서부터 시작해서 늘려나갈것

model = Sequential()
model.add(GaussianNoise(noise_factor, input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))



model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Activation('relu'))

print(model.summary())



# EARLY STOPPING
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_acc', verbose=1)

model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['accuracy']) 
hist = model.fit(training_images_clean
          , training_images_clean
          , nb_epoch=10
          , batch_size=250
          , shuffle=True
          , callbacks=[early_stopping]
          , validation_data = (validation_images_noisy, validation_images_clean))


# MODEL SAVE - FOR CODING
model.save('denoising.h5')

# MODEL LOAD - FOR CODING
from keras.models import load_model
model = load_model('denoising.h5')


# TO SEE THE RESULT VISUALY


test_images_denoised = model.predict(test_images_noisy)
test_images_denoised = np.reshape(test_images_denoised, (len(test_images_denoised), 28, 28))
test_images_noisy = np.reshape(test_images_noisy, (len(test_images_noisy), 28, 28))

import random
def show_image(i):       # ❶ 헤더 행
    plt.imshow(test_images_noisy[i], cmap='gray')
    plt.show()
    plt.imshow(test_images_denoised[i], cmap='gray')
    plt.show()
show_image(random.randint(0, 2000))
show_image(random.randint(0, 2000))
show_image(random.randint(0, 2000))





# PRINT THE RESULT ON THE CONSOLE
from numpy import linalg as LA
print('* USING CLEAR, NOISE, DENOISE VALIDATION SET TO CHECK REMAINING NOISE')
validation_images_denoised = model.predict(validation_images_noisy)
validation_images_clean = np.reshape(validation_images_clean, (len(validation_images_clean), 1, 28, 28))
validation_images_denoised = np.reshape(validation_images_denoised, (len(validation_images_denoised), 1, 28, 28))
total_noise = 0
for i in range (0,2000) :
    total_noise += LA.norm(validation_images_denoised[i] - validation_images_clean[i])
print('* TOTAL NOISE : ', total_noise/2000)

# PRINT THE RESULT AS A GRPAH

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


test_images_denoised = np.reshape(test_images_denoised, (len(test_images_denoised), 1, 28, 28))
test_images_clean = test_images_denoised
# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert test_images_clean.ndim == 4
assert test_images_clean.shape[0] == 2000
assert test_images_clean.shape[1] == 1
assert test_images_clean.shape[2] == 28
assert test_images_clean.shape[3] == 28

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('test_images_clean.npy', test_images_clean)



