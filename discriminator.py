'''
    File name: discriminator.py
    Author: maayan wislizky
    Date created: 8/8/2021
    Date last modified: 8/8/2021
    Python Version: 3.7
'''

import tensorflow_addons as tfa
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import LeakyReLU


class Discriminator:

    def __init__(self, image_shape, **kwargs):
        # initialize all necessary parameters for discriminator model
        self.image_shape = image_shape
        self.loss = kwargs.get('loss', 'mse')
        self.lr = kwargs.get('lr', 2e-4)
        self.loss_initial_weights = kwargs.get('loss_initial_weights', [.5])
        self.kernel_size = kwargs.get('kernel_size', (4, 4))
        self.alpha = kwargs.get('alpha', .2)
        self.padding = kwargs.get('padding', 'same')
        self.strides = kwargs.get('strides', (2, 2))
        self.beta_1 = kwargs.get('beta_1', .5)

    def build(self) -> Model:
        """
        build Discriminator model according to the pre defined parameters
        :return: tf Model
        """
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # source image input
        in_image = Input(shape=self.image_shape)

        # C64
        d = Conv2D(64, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=self.alpha)(d)

        # C128
        d = Conv2D(128, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=init)(d)
        d = tfa.layers.InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=self.alpha)(d)

        # C256
        d = Conv2D(256, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=init)(d)
        d = tfa.layers.InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=self.alpha)(d)

        # C512
        d = Conv2D(512, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=init)(d)
        d = tfa.layers.InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=self.alpha)(d)

        # second last output layer
        d = Conv2D(512, self.kernel_size, padding=self.padding, kernel_initializer=init)(d)
        d = tfa.layers.InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=self.alpha)(d)

        # patch output
        patch_out = Conv2D(1, self.kernel_size, padding=self.padding, kernel_initializer=init)(d)

        # creates the model
        model = Model(in_image, patch_out)

        # compile model
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, beta_1=self.beta_1),
                      loss_weights=self.loss_initial_weights)

        return model
