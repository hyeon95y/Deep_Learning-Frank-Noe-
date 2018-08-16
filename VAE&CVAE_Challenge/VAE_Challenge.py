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
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# parameters
num_classes = 10
batch_size = 16
latent_dim = 2
img_shape = (28, 28,1)
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
    x_train = x_train.reshape(length_train,28,28,1)
    x_test = x_test.reshape(length_vali,28,28,1)

# encoder
from keras.layers import Conv2D, Flatten, Dense, Lambda, Input, Reshape, Conv2DTranspose
from keras.models import Model

input_img = keras.Input(shape=img_shape)
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2,2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mu, z_log_sigma])
# encoder outputs z_mu, not z for scatter
encoder = Model(input_img, z_mu)

decoder_input = keras.Input(K.int_shape(z)[1:])
decoder_dense = Dense(np.prod(shape_before_flattening[1:]), activation='relu')
decoder_reshape = Reshape(shape_before_flattening[1:])
decoder_conv2dtranspose = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))
decoder_conv2d = Conv2D(1, 3, padding='same', activation='sigmoid')

# only for decoding
decoder = Model(decoder_input, decoder_conv2d(decoder_conv2dtranspose(decoder_reshape(decoder_dense(decoder_input)))))
z_decoded = decoder(z)
# vae
y = CustomVariationalLayer()([input_img, z_decoded])
vae = Model(input_img, y)
vae.compile(optimizer='adam', loss=None, metrics=['accuracy'])

# callbacks
filepath="VAE-{epoch:02d}.hdf5"
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='vae_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early_stopping = keras.callbacks.EarlyStopping(patience=3)
'''
hist = vae.fit(x=x_train, y=None,
               shuffle=True,
               epochs=50,
               batch_size=batch_size,
               validation_data=(x_test, None),
               verbose=1,
               callbacks = [early_stopping, model_checkpoint])
'''
# save and load model for coding
from keras.models import load_model
#vae.save('VAE.h5')
#decoder.save('VAE-decoder.h5')
vae = load_model('VAE.h5', custom_objects={'CustomVariationalLayer':CustomVariationalLayer,'latent_dim':latent_dim})
decoder = load_model('VAE-decoder.h5')

# save/load hist for coding
import pickle
filename = 'VAE-hist.p'
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
plt.title('VAE-history')
#plt.show()
fig = plt.gcf()
fig.savefig('VAE-history.png')

# visualize encoding part
plt.clf()
prediction = encoder.predict(x_test)
print('* prediction : ', prediction)

# assign different colours
for i in range(0, len(y_test)) :
    if y_test[i] == 0 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='red')
    elif y_test[i] == 1 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='green')
    elif y_test[i] == 2 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='blue')
    elif y_test[i] == 3 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='yellow')
    elif y_test[i] == 4 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='gray')
    elif y_test[i] == 5 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='orange')
    elif y_test[i] == 6 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='purple')
    elif y_test[i] == 7 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='black')
    elif y_test[i] == 8 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='brown')
    elif y_test[i] == 9 :
        plt.scatter(prediction[i, :1], prediction[i, 1:], color='pink')



#plt.scatter(prediction[0, :1], y_test[0])
fig = plt.gcf()
fig.savefig('VAE-encoding.png')

# visualize
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x) :
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i+1) * digit_size,
               j* digit_size: (j+1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.title('VAE-grid')
plt.imshow(figure, cmap='gnuplot2')
#plt.show()
fig = plt.gcf()
fig.savefig('VAE-grid.png')
