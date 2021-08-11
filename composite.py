'''
    File name: composite.py
    Author: maayan wislizky
    Date created: 1/8/2021
    Date last modified: 1/8/2021
    Python Version: 3.7
'''

from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizer_v2.adam import Adam

class Composite:

    def __init__(self, image_shape, generator_1, generator_2, discriminator, **kwargs):
        # initialize all necessary parameters for composite model
        self.image_shape = image_shape
        self.generator_1 = generator_1
        self.generator_2 = generator_2
        self.discriminator = discriminator
        self.lr = kwargs.get('lr', 2e-4)
        self.beta_1 = kwargs.get('beta_1', .5)

    def build(self):
        # ensure the model we're updating is trainable
        self.generator_1.trainable = True
        
        # mark discriminator as not trainable
        self.discriminator.trainable = False
        
        # mark other generator model as not trainable
        self.generator_2.trainable = False
        
        # discriminator element
        input_gen = Input(shape=self.image_shape)
        gen1_out = self.generator_1(input_gen)
        output_d = self.discriminator(gen1_out)
        
        # identity element
        input_id = Input(shape=self.image_shape)
        output_id = self.generator_1(input_id)
        
        # forward cycle
        output_f = self.generator_2(gen1_out)
        
        # backward cycle
        gen2_out = self.generator_2(input_id)
        output_b = self.generator_1(gen2_out)
        
        # define model graph
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        
        # define optimization algorithm configuration
        opt = Adam(lr=self.lr, beta_1=self.beta_1)
        
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        return model
