
# Create LR stages for a FULL IMAGE Classifier
# Using slices of Resnet and scheduled according to stages

import torch.optim as optim

from callbacks.cb import Callbacks    # base


class LR_SchedCB_full(Callbacks):
    def __init__(self):
        #print("init Learning Rate sched patch clf.")
        pass

    def update_LR(self, epoch, model, optimizer, optim_args): #, stages, **kwargs):
        """Prepare optimizer according to epoch. """

        ep_stage1 = optim_args['stages']
        use_wd = optim_args['use_wd'] if optim_args['use_wd'] else False
        #use_wd = kwargs.get('use_wd') if kwargs.get('use_wd') else False

        # set differnt LR for different trainable layers and epoch
        if epoch < ep_stage1:

            print('Fase 1: ', end='')
            for n, param in enumerate(model.parameters()):
                if n < 261:   # 161:Resnet50,  261:ResNest50
                    param.requires_grad = False
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.003)

        if epoch >= ep_stage1:  # and epoch < ep_stage2:
            print('Fase 2: ', end='')
            for n, param in enumerate(model.parameters()):
                #     if n < 141:  # 151:
                #         param.requires_grad = False
                #     else:
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

        # Check # of parameters to be updated
        cont = 0
        for name, param in model.named_parameters():
            #print(cont, name)
            if param.requires_grad is True:
                cont += 1
        print(f'Updating {cont:3d} parameters.')
