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
from keras.layers.merge import concatenate

def main():
    sns.set_context('talk', font_scale=1.5)
    sns.set_style('white') # for image
    mpl.rc('image', cmap='inferno', interpolation='nearest') # for image
    pass


class CVAE():
    def __init__(self):
        self.input_dim = 784
        self.latent_dim = 128
        nch = 256

        # input
        self.x_input = Input(shape=(self.input_dim,))
        self.label_input_enc = Input(shape=(10,))
        self.label_input_dec = Input(shape=(10,))

        # encoder
        x = Dense(nch, activation='relu')(self.x_input)
        lx = Dense(nch, activation='relu')(self.label_input_enc)
        x = concatenate([x, lx])
        x = Dense(nch, activation='relu')(x)

        self.z_mean = Dense(self.latent_dim)(x)
        self.z_log_var = Dense(self.latent_dim)(x)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # decoder
        self.dec1 = Dense(nch, activation='relu')
        self.dec1_label = Dense(nch, activation='relu')
        self.dec2 = Dense(nch, activation='relu')
        self.dec_out = Dense(self.input_dim, activation='sigmoid')

        x = self.dec1(self.z)
        lx = self.dec1_label(self.label_input_dec)
        x = concatenate([x, lx])
        x = self.dec2(x)
        self.x_out = self.dec_out(x)

    def sampling(self, args):
        z_mean, z_log_var = args
        nd = K.shape(z_mean)[0]
        nc = self.latent_dim
        eps = K.random_normal(shape=(nd, nc), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * eps

    def cvae(self):
        return Model([self.x_input, self.label_input_enc, self.label_input_dec], self.x_out)

    def encoder(self):
        return Model([self.x_input, self.label_input_enc], self.z_mean)

    def decoder(self):
        z = Input(shape=(self.latent_dim,))
        l = Input(shape=(10,))
        x = self.dec1(z)
        lx = self.dec1_label(l)
        x = concatenate([x, lx])
        x = self.dec2(x)
        x_out = self.dec_out(x)
        return Model([z,l], x_out)

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
    nd = 1000 # number of images for training
    ndo = 100 # number of images for testing

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train[:nd]
    x_train = x_train.reshape(nd, -1)
    y_train = y_train[:nd]
    y_train = np.eye(10)[y_train] # to one-hot

    x_test = x_test.astype('float32') / 255.
    x_test = x_test[:ndo]
    x_test = x_test.reshape(ndo, -1)
    y_test = y_test[:ndo]
    y_test = np.eye(10)[y_test] # to one-hot

    # make vae model and train
    np.random.seed(seed)
    cvae = CVAE()
    model = cvae.cvae()
    model.summary()
    model.add_loss(cvae.loss())
    np.random.seed(seed)
    model.compile(optimizer='adam', loss=None)
    model.fit([x_train, y_train, y_train], epochs=epochs, batch_size=batch_size)

    # extract latent z
    pos = [] # select nu
    for i in range(10):
        n = x_train.shape[0]
        mask = (np.argmax(y_train, axis=-1)==i)
        pos.append(np.arange(n)[mask][0])

    x = x_train[pos]
    y = y_train[pos]
    enc = cvae.encoder()
    z = enc.predict([x,y])

    # style gen
    zz = z.repeat(10, axis=0)
    yy = np.tile(np.eye(10), 10).T
    dec = cvae.decoder()
    xx = dec.predict([zz, yy])
    xin = x.reshape(280,28)
    xin = np.append(xin, np.zeros((280,5))+1.0, axis=1)
    xout = xx.reshape(10,10,28,28).transpose(0,2,1,3).reshape(280,280)
    dst = np.append(xin, xout, axis=1)

    # plot images
    plt.imshow(dst)
    plt.tight_layout();plt.show()

    # output images
    out = (cm.inferno(dst)[:,:,:3]*255).astype(np.uint8)
    out[:,28:33] = 255
    skio.imsave('png/cvae_out_%03d.png' % seed, out)
