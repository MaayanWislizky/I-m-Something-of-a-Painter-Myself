'''
    File name: generator.py
    Author: maayan wislizky
    Date created: 1/8/2021
    Date last modified: 2/8/2021
    Python Version: 3.7
'''

import tensorflow_addons as tfa
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate


class Generator:

    def __init__(self, image_shape, **kwargs):
        # initialize all necessary parameters for generator model
        self.image_shape = image_shape
        self.kernel_size = kwargs.get('kernel_size', [(7, 7), (3, 3)])
        self.padding = kwargs.get('padding', 'same')
        self.strides = kwargs.get('strides', (2, 2))
        self.n_resnet_layers = kwargs.get('n_resnet_layers', 9)

    def _resnet(self, n_filters, input_layer):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # first layer convolutional layer
        g = Conv2D(n_filters, self.kernel_size[1], padding=self.padding, kernel_initializer=init)(input_layer)
        g = tfa.layers.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        # second convolutional layer
        g = Conv2D(n_filters, self.kernel_size[1], padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(g)

        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g

    def build(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # image input
        in_image = Input(shape=self.image_shape)

        # c7s1-64
        d = Conv2D(64, self.kernel_size[0], padding=self.padding, kernel_initializer=init)(in_image)
        g = tfa.layers.InstanceNormalization(axis=-1)(d)
        g = Activation('relu')(g)

        # d128
        d = Conv2D(128, self.kernel_size[1], strides=self.strides, padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(d)
        g = Activation('relu')(g)

        # d256
        d = Conv2D(256, self.kernel_size[1], strides=self.strides, padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(d)
        g = Activation('relu')(g)

        # R256
        for _ in range(self.n_resnet_layers):
            g = self._resnet(256, g)

        # u128
        g = Conv2DTranspose(128, self.kernel_size[1], strides=self.strides, padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        # u64
        g = Conv2DTranspose(64, self.kernel_size[1], strides=self.strides, padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        # c7s1-3
        g = Conv2D(3, self.kernel_size[0], padding=self.padding, kernel_initializer=init)(g)
        g = tfa.layers.InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)

        # creates the model
        model = Model(in_image, out_image)
        return model