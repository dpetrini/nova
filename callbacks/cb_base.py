#
# Clean code from main file
#
# It must keep generic to be used many times. Improvements goes
# here then become available for all trials. No stats, graphs,
# etc repeated.
#
# TODO:
#  - mprove Object sharing with CBs..

import torch
import time
import copy
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

from callbacks.cb import Callbacks    # base 


class BaseCB(Callbacks):
    def __init__(self, name):
        self.models_dir = f'models_{name}'
        if os.path.isdir(f'{self.models_dir}') is False:
            os.makedirs(f'{self.models_dir}')
            print(f'Creating dir {self.models_dir}')
        pass

    def __repr__(self):
        return 'BASE'

    def begin_train_val(self, epochs, train_dataloader, val_dataloader, bs_size):
        super().begin_train_val(epochs)
        self.train_step = len(train_dataloader)
        self.val_step = len(val_dataloader)
        self.bar_step = self.train_step // 50 if self.train_step >= 50 else 1
        self.bar_step_val = self.val_step // 10 if self.val_step >= 10 else 1  #-- CONFIRM
        #print(self.bar_step, self.bar_step_val)
        self.total_train_samples, self.total_val_samples = 0, 0
        self.bs_size = bs_size
        self.best_val_acc = 0.3
        self.n_epoch = 0
        self.history = []
        self.start = time.time()
        self.second_best = 0.3  # GUARDA AUC - DEPOIS DEVE IR PARA OUTRO CB
        return True

    def begin_epoch(self, current_epoch):
        self.train_loss, self.train_acc = 0., 0.
        self.val_loss, self.val_acc = 0., 0.
        self.n_train_samples, self.n_val_samples = 0, 0
        #self.n_step, self.n_step_val = 0., 0.
        self.n_iter = 0
        self.n_epoch = current_epoch
        self.epoch_start = time.time()
        print('\nEpoch: {}/{}'.format(self.n_epoch, self.epochs))
        return True

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        # Epoch accumulators
        # self.train_acc += train_acc
        # self.train_loss += train_loss
        # self.val_acc += val_acc
        # self.val_loss += val_loss
        self.model = model
        #self.n_iter = 0
        #self.n_epoch += 1

        # fing average training loss and accuracy
        #print(self.n_train_samples, self.n_val_samples)
        # avg_train_loss = self.train_loss/self.n_train_samples
        # avg_train_acc = self.train_acc/self.n_train_samples
        avg_train_loss = train_loss/self.n_train_samples
        avg_train_acc = train_acc/self.n_train_samples

        # find average validation and loss
        # avg_val_loss = self.val_loss/self.n_val_samples
        # avg_val_acc = self.val_acc/self.n_val_samples

        #print(">> ", val_acc, self.n_val_samples)

        avg_val_loss = val_loss/self.n_val_samples
        avg_val_acc = val_acc/self.n_val_samples

        self.total_train_samples += self.n_train_samples
        self.total_val_samples += self.n_val_samples

        self.history.append([avg_train_loss, avg_val_loss,
                             avg_train_acc, avg_val_acc])

        if (avg_val_acc > self.best_val_acc):
            print(f' |------>  Best Val Acc model now {avg_val_acc:1.4f}')
            self.best_model = copy.deepcopy(model)  # Will work
            self.best_val_acc = avg_val_acc
        else: print()   # noop

        epoch_end = time.time()
        print('Epoch: {:03d}, Train: Loss: {:.4f}, Acc: {:.2f}%,' \
              ' Val: Loss: {:0.4f}, Acc: {:.2f}%, Time: {:.2f}s'
              .format(self.n_epoch, avg_train_loss, avg_train_acc*100,
                      avg_val_loss, avg_val_acc*100,
                      epoch_end-self.epoch_start))

        return True

    def after_step(self, n_samples, *args):
        self.n_train_samples += n_samples
        self.n_iter += 1
        #print(self.n_iter, self.bar_step, self.n_iter % self.bar_step)
        #if (self.n_step % self.bar_step) == 0:
        if (self.n_iter % self.bar_step) == 0:
            print('▒', end='', flush=True)
        return True

    def after_step_val(self, n_samples, *args):
        self.n_val_samples += n_samples
        self.n_iter += 1
        #if (self.n_step_val % self.bar_step_val) == 0:
        #print(self.n_iter, self.bar_step_val, self.n_iter % self.bar_step_val)
        if (self.n_iter % self.bar_step_val) == 0:
            print('░', end='', flush=True)
        return True

    def after_train_val(self):
        elapsed_mins = (time.time()-self.start)/60
        print('Total training time:  {:.2f} mins.'.format(elapsed_mins))

        #return self.history, self.best_model, self.best_val_acc, self.second_best

        ts = time.time()
        n_samples = int((self.total_train_samples + self.total_val_samples)/self.epochs)
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')
        summary = str(st)+'_'+str(self.epochs)+'ep_'+str(n_samples)+'n'

        # save the model
        torch.save(self.model.state_dict(),
                   f'{self.models_dir}/{summary}_model.pt')
        torch.save(self.best_model.state_dict(),
                   f'{self.models_dir}/{summary}_best_model_ACC.pt')
        print(f'cb_base: Last and best acc models saved in {self.models_dir}/')

        result_text = f"Best ACC: {self.best_val_acc:1.2f}"

        # plots
        history = np.array(self.history)
        plt.plot(history[:, 0:2])
        plt.title("Loss - Patch Classifier Resnet50")
        plt.legend(['Tr Loss', 'Val Loss'], loc="upper right")
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0, 5)
        plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
        plt.savefig('plot_train/'+str(st)+'_loss_curve.png')
        plt.show()

        plt.plot(history[:, 2:4])
        plt.title("ACC - Patch Classifier Resnet50 " + result_text)
        plt.legend(['Tr Accuracy', 'Val Accuracy'], loc="lower right")
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
        plt.savefig('plot_train/'+str(st)+'_acc_curve.png')
        plt.text(0, 0.9, result_text, bbox=dict(facecolor='red', alpha=0.3))
        plt.show()

        return True

    # Workaround para passar modelo
    def get_model(self):
        return self.model

    def get_best_model(self):
        return self.best_model
