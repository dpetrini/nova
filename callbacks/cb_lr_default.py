
# Using standard pytorch lr scheduler
# create an instance in begin_train_val

import math
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from callbacks.cb import Callbacks    # base


class LR_SchedCB(Callbacks):
    def __init__(self):
	self.base_lr = 1e-3

    def __repr__(self):
        return 'Default_LR_Sched'

    def begin_train_val(self, epochs, model, train_dataloader, val_dataloader, bs_size, optimizer):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    def update_LR(self, epoch, model, optimizer, optim_args): #, stages, **kwargs):
        """Prepare optimizer according to epoch. """
     
        for param_group in optimizer.param_groups:
            self.base_lr = param_group['lr']        # get original LR from MAIN - one group

        self.res.append(self.lr)

        # Check # of parameters to be updated
        cont = 0
        for name, param in model.named_parameters():
            # print(name, p.size().item())
            if param.requires_grad is True:
                cont += 1
        print(f'Updating {cont:3d} parameters ', end='')

        for param_group in optimizer.param_groups:
            print("current learning rate is: {}".format(param_group['lr']))

        return False  # no changing LR here

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        pass
        self.scheduler.step()

    def after_train_val(self):
        a = range(1, self.epochs+1)
        plt.plot(a, self.res)
        plt.legend(['STD LR sched'], loc="upper right")
        plt.show()
