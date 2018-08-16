import numpy as np
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

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


# TRAIN MODEL ON training_images_clean

# CONV NETWORK
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
input_img = Input(shape=(28, 28, 1))
# INCODE
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2,2), padding='same')(x)
# DECODE 
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='relu', padding='same')(x) # should be linear, proper purpose
# no cmopression,중간에 압축할 필요 없고 같은 크기거나 중간에 더 커져도 
# from 1~2 layers -> bigger
# clean-clean and put noise during train with noise layer
# or noised-clean but slower

# EARLY STOPPING
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', verbose=1)

model = Model(input_img, decoded)
model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['accuracy']) #try meansquare ~
hist = model.fit(training_images_noisy
          , training_images_clean
          , nb_epoch=30
          , batch_size=250
          , shuffle=True
          #, callbacks=[early_stopping]
          , validation_data = (validation_images_noisy, validation_images_clean))


# CHECK YOUR MODEL USING (validation_images_clean, validation_images_noisy)
score = model.evaluate(validation_images_clean, validation_images_noisy, verbose=0)
print(score)
score2 = model.evaluate(validation_images_noisy, validation_images_clean, verbose=0)
print(score2)

# MODEL SAVE - FOR CODING
model.save('denoising_test.h5')

# MODEL LOAD - FOR CODING
from keras.models import load_model
model = load_model('denoising_test.h5')

# DENOISE IMAGES (test_images_clean) USING test_images_noisy
test_images_clean = model.predict(test_images_noisy)
print(test_images_noisy.shape)
print(test_images_clean.shape)


test_images_clean = np.reshape(test_images_clean, (len(test_images_clean), 28, 28))
print(test_images_clean.shape)

plt.imshow(test_images_clean[1], cmap='gray')
plt.show()
plt.imshow(test_images_clean[2], cmap='gray')
plt.show()
plt.imshow(test_images_clean[3], cmap='gray')
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

'''
# MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
assert test_images_clean.ndim == 4
assert test_images_clean.shape[0] == 2000
assert test_images_clean.shape[1] == 1
assert test_images_clean.shape[2] == 28
assert test_images_clean.shape[3] == 28

# AND SAVE EXACTLY AS SHOWN BELOW
np.save('test_images_clean.npy', test_images_clean)
'''