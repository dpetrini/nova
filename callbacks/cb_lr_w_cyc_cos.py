
# Warm-up + CYCLIC cosine Learning Rate scheduler. Based on the formulas from:
#
# "Bag of Tricks for Image Classification with Convolutional ...." paper and
#
#  SGDR:   STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS" 
#
# Stores LR to plot in the end

import math
import matplotlib.pyplot as plt
import torch.optim as optim

from callbacks.cb import Callbacks    # base


class LR_SchedCB_W_Cyc_Cos(Callbacks):
    def __init__(self):
        print('Warm-up + Cyclic Cosine Learning Rate scheduler')
        # Some default values
        self.warmup = 5
        self.base_lr = 1e-3
        self.epochs = 50
        self.next_lr = 0
        self.res = []

    def __repr__(self):
        return 'Warm-up_+_cosine_LR'

    def begin_train_val(self, epochs, model, train_dataloader, val_dataloader, bs_size, optimizer):
        self.warmup = epochs//20 if epochs//20 > 5 else self.warmup     # 5 or 5%
        self.epochs = epochs

    def update_LR(self, epoch, model, optimizer, optim_args): #, stages, **kwargs):
        """Prepare optimizer according to epoch. """

        self.base_lr = optim_args['base_lr'] if 'base_lr' in optim_args else 1e-3
        self.delta = optim_args['delta'] if 'delta' in optim_args else 1e-4
        self.T = optim_args['period'] if 'period' in optim_args else 20
        self.warmup = optim_args['warmup'] if 'warmup' in optim_args else 5

# def lr_cos(epoch, T, delta, lr_base):
#     n = 1/2*(delta)*(1+math.cos(epoch*math.pi/T)) + lr_base
#     return n


        if epoch < 1: epoch = 1

        # if epoch == 1:
        #     for param_group in optimizer.param_groups:
        #         self.base_lr = param_group['lr']        # get original LR from MAIN - one group

        if epoch <= self.warmup:
            self.next_lr = epoch * self.base_lr/self.warmup
        else:
            # self.next_lr = 1/2*(1+math.cos((epoch-self.warmup)*math.pi/(self.epochs-self.warmup)))*self.base_lr
            self.next_lr = 1/2*(self.delta)*(1+math.cos((epoch-self.warmup)*math.pi/self.T)) + self.base_lr - self.delta/2

        optimizer = optim.Adam(model.parameters(), lr=self.next_lr)
        self.res.append(self.next_lr)

        # Check # of parameters to be updated
        cont = 0
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                cont += 1
        print(f'Updating {cont:3d} parameters ', end='')

        for param_group in optimizer.param_groups:
            print(f"current learning rate is: {param_group['lr']:1.2e}")

        return optimizer

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        pass

    def after_train_val(self):
        pass
        # x_axis = range(1, self.epochs+1)
        # plt.plot(x_axis, self.res)
        # plt.legend(['Warm+Cyclic Cosine LR'], loc="upper right")
        # plt.show()
