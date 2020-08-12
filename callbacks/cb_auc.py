
# Calculate AUC for classification

# TODO: bring auc calculation to here (reading label and output)

import torch
import copy
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


from callbacks.cb import Callbacks    # base


class AUC_CB(Callbacks):
    def __init__(self, name):
        self.name = name
        self.models_dir = f'models_{name}'
        self.plots_dir = f'plots_{name}'
        pass
        # self.has_auc = False        # during tests
        # self.y_hat_auc = []
        # self.label_auc = []
        # self.y_hat_val_auc = []
        # self.label_val_auc = []

    def begin_epoch(self, current_epoch):
        self.y_hat_auc, self.label_auc = [], []
        self.y_hat_val_auc, self.label_val_auc = [], []

    def begin_train_val(self, epochs, train_dataloader, val_dataloader, bs_size, optimizer):
        super().begin_train_val(epochs)
        self.history = []
        self.best_auc = 0.3
        return True

    def after_step(self, n_samples, labels, outputs):
        self.label_auc = np.append(self.label_auc, labels.cpu().detach().numpy())
        self.y_hat_auc = np.append(self.y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
        # pegando soh malignant posicao [1] para comparar com label. ok.
        #print(outputs, torch.softmax(outputs, dim=1), torch.softmax(outputs, dim=1)[0][1])

        return True

    def after_step_val(self, n_samples, labels, outputs):
        # guarda Y-hat/label para AUC - VAL
        self.label_val_auc = np.append(self.label_val_auc, labels.cpu().detach().numpy())
        self.y_hat_val_auc = np.append(self.y_hat_val_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
        # arrumar essa linha de cima...

        return True

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        # auc_train = kwargs.get('auc_train')
        # auc_val = kwargs.get('auc_val')

        #print(auc_train, auc_val)

        #if (auc_train and auc_val):

        # calculate AUC Train
        auc_malign_train = roc_auc_score(self.label_auc.ravel(),
                                         self.y_hat_auc.ravel())
        # calculate AUC VAL
        #print(self.label_val_auc.shape, self.y_hat_val_auc.shape, self.label_val_auc.sum(), self.y_hat_val_auc.sum())
        #print(self.label_val_auc)
        #print(self.y_hat_val_auc)
        auc_malign_val = roc_auc_score(self.label_val_auc.ravel(),
                                       self.y_hat_val_auc.ravel())

        print('[Train] AUC Malignant: %.3f' % auc_malign_train, end='')
        print(' [Val] AUC Val Malignant: %.3f' % auc_malign_val, end='')

        self.history.append([auc_malign_train, auc_malign_val])
        if (auc_malign_val > self.best_auc):
            print(f' |------>  Best Val Auc model now {auc_malign_val:1.4f}')
            self.best_model = copy.deepcopy(model)  # Will work                
            self.best_auc = auc_malign_val
        else: print()   # noop

        return True

    def after_train_val(self):

        ts = time.time()
        # n_samples = self.total_train_samples + self.total_val_samples
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')
        summary = str(st)+'_'+str(self.epochs)+'ep'#+str(n_samples)+'n'

        result_auc = f"Best AUC: {self.best_auc:1.4f}"
        print(result_auc)

        history = np.array(self.history)
        plt.plot(history[:, 0:2])
        plt.title(f'AUC - FullClassifier {self.name} {result_auc}')
        plt.legend(['Train AUC', 'Val AUC'], loc="lower right")
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
        plt.text(0, 0.9, result_auc, bbox=dict(facecolor='red', alpha=0.3))
        plt.savefig(f'{self.plots_dir}/{st}_AUC_curve.png')
        plt.show()

        # save the model
        #torch.save(self.model.state_dict(), 'models_train/'+summary+'_model_'+'.pt')
        torch.save(self.best_model.state_dict(),
                   f'{self.models_dir}/{summary}_best_model_AUC_0{result_auc[-3:]}.pt')
        print(f'cb_auc: Best auc model saved in {self.models_dir}/')

        return True

    @property
    def best_auc_model(self):
        return self.best_model
