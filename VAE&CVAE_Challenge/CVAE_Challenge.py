# import modules
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.stats import norm
import os

# for reproducibility
np.random.seed(7)

# sampling
def sampling(args) :
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

# custom varitional layer
class CustomVariationalLayer(keras.layers.Layer) :
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0][:, :784] # because of [x_train, y_train]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# parameters
num_classes = 10
batch_size = 16
latent_dim = 1
img_shape = (784, )
label_shape = (1, )

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
    y_test = y_test.reshape(length_vali, 1)

# encoder
from keras.layers import Conv2D, Flatten, Dense, Lambda, Input, Reshape, Conv2DTranspose
from keras.models import Model
from keras.layers.merge import concatenate

pixels = keras.Input(shape=img_shape)
cond = keras.Input(shape=(y_train.shape[1], ))
input = concatenate([pixels, cond])

x = Dense(784, activation='relu')(input)
x = Reshape((28, 28, 1))(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2,2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mu, z_log_sigma])
z_cond = concatenate([z, cond])
# encoder outputs z, not z_cond
encoder = Model([pixels, cond], z)


decoder_input = keras.Input(K.int_shape(z_cond)[1:])
print('*decoder_input._keras_shape : ', decoder_input._keras_shape)
decoder_dense = Dense(np.prod(shape_before_flattening[1:]), activation='relu')
decoder_reshape = Reshape(shape_before_flattening[1:])
decoder_conv2dtranspose = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))
decoder_conv2d = Conv2D(1, 3, padding='same', activation='sigmoid')

# only for decoding
decoder = Model(decoder_input, decoder_conv2d(decoder_conv2dtranspose(decoder_reshape(decoder_dense(decoder_input)))))
z_decoded = decoder(z_cond)
# cvae
y = CustomVariationalLayer()([input, z_decoded])
cvae = Model([pixels, cond], y)
cvae.compile(optimizer='adam', loss=None, metrics=['accuracy'])

# save every epoch to track suitable epoch
filepath="CVAE-{epoch:02d}.hdf5"
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

'''
hist = cvae.fit(x=[x_train, y_train], y=None,
               shuffle=True,
               epochs=50,
               batch_size=batch_size,
               validation_data=([x_test, y_test], None),
               verbose=1,
               callbacks=[model_checkpoint]
               )
'''
# save and load model for coding
from keras.models import load_model
#cvae.save('CVAE.h5')
#decoder.save('CVAE-decoder.h5')
cvae = load_model('CVAE.h5', custom_objects={'CustomVariationalLayer':CustomVariationalLayer,'latent_dim':latent_dim})
decoder = load_model('CVAE-decoder.h5')

# save/load hist for coding
import pickle
filename = 'CVAE-hist.p'
#with open(filename, 'wb') as file:
#    pickle.dump(hist.history, file)
with open(filename, 'rb') as f:
    hist = pickle.load(f)

# see the history
fig, loss_ax = plt.subplots()
loss_ax.plot(hist['loss'], 'y', label='train loss')
loss_ax.plot(hist['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
plt.title('CVAE-history')
#plt.show()
fig = plt.gcf()
fig.savefig('CVAE-history.png')

# visualize encoding part
plt.clf()
prediction = encoder.predict([x_test, y_test])
print('* prediction : ', prediction)
plt.scatter(prediction[:, :1], y_test)
fig = plt.gcf()
fig.savefig('CVAE-encoding.png')

# visualize
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
grid_x = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]

for i, yi in enumerate(grid_x) :
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i+1) * digit_size,
               j* digit_size: (j+1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.title('CVAE-grid')
plt.imshow(figure, cmap='gnuplot2')
#plt.show()
fig = plt.gcf()
fig.savefig('CVAE-grid.png')
