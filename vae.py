#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib as mpl
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import cm
from skimage import io as skio

import keras
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda


def main():
    sns.set_context('talk', font_scale=1.5)
    sns.set_style('white') # for image
    mpl.rc('image', cmap='inferno', interpolation='nearest') # for image
    pass

class VAE():
    def __init__(self):

        self.input_dim = 784
        self.latent_dim = 2
        nch = 256

        # encoder
        self.x_input = Input(shape=(self.input_dim,))
        x = Dense(nch, activation='relu')(self.x_input)
        x = Dense(nch, activation='relu')(x)

        self.z_mean = Dense(self.latent_dim)(x)
        self.z_log_var = Dense(self.latent_dim)(x)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # decoder
        self.dec1 = Dense(nch, activation='relu')
        self.dec2 = Dense(nch, activation='relu')
        self.dec_out = Dense(self.input_dim, activation='sigmoid')

        x = self.dec1(self.z)
        x = self.dec2(x)
        self.x_out = self.dec_out(x)

    def sampling(self, args):
        z_mean, z_log_var = args
        nd = K.shape(z_mean)[0]
        nc = self.latent_dim
        eps = K.random_normal(shape=(nd, nc), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * eps

    def vae(self):
        return Model(self.x_input, self.x_out)

    def encoder(self):
        return Model(self.x_input, self.z_mean)

    def decoder(self):
        z = Input(shape=(self.latent_dim,))
        x = self.dec1(z)
        x = self.dec2(x)
        x_out = self.dec_out(x)
        return Model(z, x_out)

    def loss(self):
        bce = metrics.binary_crossentropy(self.x_input, self.x_out)
        xent_loss = self.input_dim * bce
        kl = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = - 0.5 * K.sum(kl, axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss


if __name__ == '__main__':
    main()

    seed = 0 # for network init

    batch_size = 32
    epochs = 2000
    nd = 200 # number of images for training

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train[:nd]
    n, h, w = x_train.shape
    x_train = x_train.reshape(n, -1)

    # make vae model and train
    np.random.seed(seed)
    vae = VAE()
    model = vae.vae()
    model.summary()
    model.add_loss(vae.loss())
    np.random.seed(seed)
    model.compile(optimizer='adam', loss=None)
    model.fit(x_train, epochs=epochs, batch_size=batch_size)

    # decode images from 2-dim latent z
    ndiv = 15 # number of images for axis
    dec = vae.decoder()
    gx = np.linspace(-3.0, 3.0, ndiv)
    gxx, gyy = np.meshgrid(gx, gx[::-1])
    z = np.append(gxx.reshape(-1,1), gyy.reshape(-1,1), axis=1)
    x = dec.predict(z)
    dst = x.reshape(ndiv,ndiv,28,28)
    dst = dst.transpose(0,2,1,3)
    dst = dst.reshape(ndiv*28, ndiv*28)

    # plot images
    plt.imshow(dst)
    plt.tight_layout();plt.show()

    # output images
    out = (cm.inferno(dst)[:,:,:3]*255).astype(np.uint8)
    #skio.imsave('vae_out_%03d.png' % seed, out)
