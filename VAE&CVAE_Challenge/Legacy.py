# import modules
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.stats import norm
import os

# for reproducibility
np.random.seed(7)

# parameters
num_classes = 10
batch_size = 16
#latent_dim_vae = 2
latent_dim_cvae = 1
img_shape = (28, 28,1)
label_shape = (1, )

# define loss
def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mu) - 1. - z_log_sigma, axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
    return (0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mu) - 1. - z_log_sigma, axis=1))

def recon_loss(y_true, y_pred):
    return (K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# sampling for VAE & CVAE
def sampling_cvae(args) :
    z_mu, z_log_sigma= args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim_cvae), mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon
'''
def sampling_vae(args) :
    z_mu, z_log_sigma= args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim_vae), mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon
'''
# load data
with np.load(os.path.join(os.path.dirname(__file__), "vae-cvae-challenge.npz")) as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']

    # shuffle indexs, split train and validation set
    number_datapoints = data_x.shape[0]
    indexes = np.arange(number_datapoints)
    np.random.shuffle(indexes)
    length_train = int(number_datapoints*0.8)
    length_vali = number_datapoints - length_train
    x_train = data_x[indexes[:length_train]]
    y_train = data_y[indexes[:length_train]]
    x_test = data_x[indexes[length_train:]]
    y_test = data_y[indexes[length_train:]]

    # reshape
    x_train = x_train.reshape(length_train,784)
    x_test = x_test.reshape(length_vali,784)
    y_train = y_train.reshape(length_train, 1)
    y_test = y_test.reshape(length_vali, 1)

# set input shape
#input_vae = keras.Input(shape=img_shape)
#input_cvae = keras.Input(shape=(784,))

from keras.layers.merge import concatenate
x = keras.Input(shape=(784, ))
cond = keras.Input(shape=(y_train.shape[1], ))
inputs = concatenate([x, cond])





# CVAE encoder
from keras.layers import Conv2D, Dense, Reshape, Flatten, Lambda, Conv2DTranspose
from keras.models import Model
x = Dense(784, activation='relu')(inputs)
x = Reshape((28, 28, 1))(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2,2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x) # (None, 14, 14, 64)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim_cvae)(x)
z_log_sigma = Dense(latent_dim_cvae)(x)
z = Lambda(sampling_cvae)([z_mu, z_log_sigma])

print(z)
z_cond = concatenate([z, cond])
print(z_cond)

# CVAE decoder
decoder_input = keras.Input(K.int_shape(z_cond)[1:])
print('* decoder_input : ', K.int_shape(z_cond))

x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
print('* np.prod(shape_before_flattening[1:]) : ', np.prod(shape_before_flattening[1:]))
print('* x._keras_shape : ', x._keras_shape)
x = Reshape(shape_before_flattening[1:])(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(x)
x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, x)
decoder.summary()
z_decoded = decoder(z_cond)

cvae = Model([x, cond], z_decoded)
