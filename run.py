'''
    File name: common.py
    Author: maayan wislizky
    Date created: 8/8/2021
    Date last modified: 8/8/2021
    Python Version: 3.7
'''


from utils import load_real_samples
from discriminator import Discriminator
from composite import Composite
from generator import Generator
from train import GanTrainer


if __name__ == '__main__':
    # load image data
    PATH = r".\real2monet.npz"
    dataset = load_real_samples(PATH)
    print('Loaded', dataset[0].shape, dataset[1].shape)

    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]

    # generator: A -> B
    generatorA2B = Generator(image_shape)
    gModel_A2B = generatorA2B.build()

    # generator: B -> A
    generatorB2A = Generator(image_shape)
    gModel_B2A = generatorB2A.build()

    # discriminator: A -> [real/fake]
    discriminator_A = Discriminator(image_shape)
    d_model_A = discriminator_A.build()

    # discriminator: B -> [real/fake]
    discriminator_B = Discriminator(image_shape)
    d_model_B = discriminator_B.build()

    # composite: A -> B -> [real/fake, A]
    compositeA2B = Composite(image_shape=image_shape, generator_1=gModel_A2B, generator_2=gModel_B2A, discriminator=d_model_B)
    c_model_AtoB = compositeA2B.build()

    # composite: B -> A -> [real/fake, B]
    compositeB2A = Composite(image_shape=image_shape, generator_1=gModel_B2A, generator_2=gModel_A2B, discriminator=d_model_A)
    c_model_BtoA = compositeB2A.build()

    # train models
    trainer = GanTrainer()
    trainer.train(d_model_A, d_model_B, gModel_A2B, gModel_B2A, c_model_AtoB, c_model_BtoA, dataset)