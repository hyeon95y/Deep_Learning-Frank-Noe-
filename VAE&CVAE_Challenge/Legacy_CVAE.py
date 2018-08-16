# import modules
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import backend as K
from scipy.stats import norm
import os

K.clear_session()
np.random.seed(7)

num_classes = 10

# load data
with np.load(os.path.join(os.path.dirname(__file__), "vae-cvae-challenge.npz")) as fh:
    data_x = fh['data_x']
    data_y = fh['data_y']
    print(data_x.shape)
    print(data_y.shape)

    # split train and validation set
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
    #x_train = x_train.reshape(length_train,28,28,1)
    #x_test = x_test.reshape(length_vali,28,28,1)
    y_train = y_train.reshape(length_train, 1)
    print(y_train)
    y_test = y_test.reshape(length_vali, 1)
    # categorize labels
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

#img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 1

# mergle pixel representation and label
x = keras.Input(shape=(x_train.shape[1], ))
cond = keras.Input(shape=(y_train.shape[1], ))
from keras.layers.merge import concatenate
inputs = concatenate([x, cond])
print('*inputs : ', inputs._keras_shape)

from keras.layers import Dense, Lambda
from keras.models import Model

h_q = Dense(512, activation='relu')(inputs)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], latent_dim), mean=0., stddev=1.)
    return mu + K.exp(log_sigma /2) * eps

z = Lambda(sample_z, output_shape=(latent_dim, ))([mu, log_sigma])

z_cond = concatenate([z, cond])

# decoder_input

decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')
h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

cvae = Model([x, cond], outputs)
encoder = Model([x, cond], mu)

d_in = keras.Input(shape=(latent_dim + y_train.shape[1],))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# define loss
def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
    return (0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))

def recon_loss(y_true, y_pred):
    return (K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))

cvae.compile(optimizer='adam', loss=vae_loss, metrics =[KL_loss, recon_loss])
from keras.callbacks import EarlyStopping

cave_hist = cvae.fit([x_train, y_train], x_train, batch_size=16, epochs=30,
                     validation_data =([x_test, y_test], x_test),
                     callbacks = [EarlyStopping(patience=5)])

from keras.models import load_model
filename = 'cvae.h5'
#cvae.save(filename)
#decoder.save('cvaedecoder.h5')
#model = load_model(filename)
decoder = load_model('cvaedecoder.h5')
'''
from scipy.misc import imsave
import time
for i in range(latent_dim+y_train.shape[1]) :
    tmp = np.zeros((1, latent_dim + y_train.shape[1]))
    tmp[0, i] = 1
    generated = decoder.predict(tmp)
    file_name = './img' + str(i) + '.jpg'
    print(generated)
    imsave(file_name, generated.reshape((28,28)))
    time.sleep(0.5)


for i in range (latent_dim + y_train.shape[1]) :
    tmp = np.zeros((1, latent_dim+y_train.shape[1]))
    tmp[0, i] = 1
    generated = decoder.predict(tmp)
    file_name = './img' + str(i) + '.jpg'
    print(generated)
    imsave(file_name, generated.reshape((28, 28)))
    time.sleep(0.5)
'''
# visualize
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
print('* grid_x : ', grid_x)
print('* grid_y : ', grid_y)

for i, yi in enumerate(grid_x) :
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict([z_sample, 1], batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i+1) * digit_size,
               j* digit_size: (j+1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show()
