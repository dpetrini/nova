
# Create LR stages for a Patch Classifier
# Using slices of Resnet and scheduled according to stages

import torch.optim as optim

from callbacks.cb import Callbacks    # base


class LR_SchedCB_2views(Callbacks):
    def __init__(self):
        #print("init Learning Rate sched patch clf.")
        pass

    def update_LR(self, epoch, model, optimizer, optim_args):
        """Prepare optimizer according to epoch. """

        ep_stage1, ep_stage2, ep_stage3 = optim_args['stages']
        layer_1, layer_2, layer_3 = optim_args['layers']
        use_wd = optim_args['use_wd'] if optim_args['use_wd'] else False

        # set differnt LR for different trainable layers and epoch
        if epoch < ep_stage1:
            print('Fase1: ', end='')
            for n, param in enumerate(model.parameters()):
                if n < layer_1: ## 173?? #185:     # APENAS FCs MID:209 (old:211)  MIDThin:185 TOP:185 (187-OLD)
                    param.requires_grad = False
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

        if epoch >= ep_stage1 and epoch < ep_stage2:
            print('Fase2: ', end='')
            for n, param in enumerate(model.parameters()):
                if n < layer_2: #185: # 185:  # (novo Resblock) MID: 185 (old:187)  MIDThin: 161  TOP: pula este
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4, # MID:lr=1e-4 MIDThin:1e-3, Transfer:lr=1e-4 
                                   weight_decay=0.005 if use_wd else 0)

        if epoch >= ep_stage2 and epoch < ep_stage3:
            print('Fase 3: ', end='')


            for n, param in enumerate(model.parameters()):
                if n < layer_3:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # for param in model.parameters():
            #     param.requires_grad = True


            optimizer = optim.Adam(model.parameters(), lr=1e-5,
                                   weight_decay=0.004 if use_wd else 0)

        if epoch >= ep_stage3:
            print('Fase 4: ', end='')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5,
                                   weight_decay=0.001 if use_wd else 0)

        # Check # of parameters to be updated
        cont = 0
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                # print("\t", name) #, '\t', param)
                cont += 1
        print('Updating {:3d} parameters.'.format(cont))

        return optimizer
