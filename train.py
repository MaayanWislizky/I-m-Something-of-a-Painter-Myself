'''
    File name: common.py
    Author: maayan wislizky
    Date created: 8/8/2021
    Date last modified: 8/8/2021
    Python Version: 3.7
'''

from utils import *

class GanTrainer:

    def __init__(self, **kwarg):
        self.n_epochs = kwarg.get('n_epochs', 10)
        self.n_batch = kwarg.get('n_batch', 10)

    def train(self, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
        # determine the output square shape of the discriminator
        n_patch = d_model_A.output_shape[1]

        # unpack dataset
        trainA, trainB = dataset

        # prepare image pool for fakes
        poolA, poolB = list(), list()

        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / self.n_batch)

        # calculate the number of training iterations
        n_steps = bat_per_epo * self.n_epochs

        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = generate_real_samples(trainA, self.n_batch, n_patch)
            X_realB, y_realB = generate_real_samples(trainB, self.n_batch, n_patch)

            # generate a batch of fake samples
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

            # update fakes from pool
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)

            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
                i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

            # evaluate the model performance every so often
            if (i + 1) % (bat_per_epo * 1) == 0:
                # plot A->B translation
                summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
                # plot B->A translation
                summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
            if (i + 1) % (bat_per_epo * 5) == 0:
                # save the models
                save_models(i, g_model_AtoB, g_model_BtoA)

        save_models(i, g_model_AtoB, g_model_BtoA)
