
# Warm-up + cosine Learning Rate scheduler. Based on the formulas from:
#
# "Bag of Tricks for Image Classification with Convolutional ...." paper

import math
import matplotlib.pyplot as plt
import torch.optim as optim

from callbacks.cb import Callbacks    # base


class LR_SchedCB(Callbacks):
    def __init__(self):
        # Some default values
        self.warmup = 5
        self.base_lr = 1e-3
        self.epochs = 50
        self.next_lr = 0
        self.res = []

    def begin_train_val(self, epochs, train_dataloader, val_dataloader, bs_size, optimizer):
        self.warmup = epochs//20 if epochs//20 > 5 else self.warmup     # 5 or 5%
        self.epochs = epochs

    def update_LR(self, epoch, model, optimizer, optim_args): #, stages, **kwargs):
        """Prepare optimizer according to epoch. """

        if epoch < 1: epoch = 1

        if epoch == 1:
            for param_group in optimizer.param_groups:
                self.base_lr = param_group['lr']        # get original LR from MAIN - one group

        if epoch <= self.warmup:
            self.next_lr = epoch * self.base_lr/self.warmup
        else:
            self.next_lr = 1/2*(1+math.cos((epoch-self.warmup)*math.pi/(self.epochs-self.warmup)))*self.base_lr

        optimizer = optim.Adam(model.parameters(), lr=self.next_lr)
        self.res.append(self.next_lr)

        # Check # of parameters to be updated
        cont = 0
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                cont += 1
        print(f'Updating {cont:3d} parameters ', end='')

        for param_group in optimizer.param_groups:
            print("current learning rate is: {}".format(param_group['lr']))

        return optimizer

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        pass

    def after_train_val(self):
        x_axis = range(1, self.epochs+1)
        plt.plot(x_axis, self.res)
        plt.legend(['Warm+Cosine LR'], loc="upper right")
        plt.show()
