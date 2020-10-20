#
# Clean code from main file
#
# It must keep generic to be used many times. Improvements goes
# here then become available for all trials. No stats, graphs,
# etc repeated.
#
# TODO:
#  - mprove Object sharing with CBs..


import time
import copy
import datetime
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from callbacks.cb import Callbacks    # base


class BaseCB(Callbacks):
    def __init__(self, name, title, config):

        # Podemos chamar init da classe mae (Callbacks) e passar config
        # e entao usar algumas variaves comuns a todos callbacks
        # como variaveis de classe. E faz sentido fazer isso no cb_base
        # pois ele sempre é chamado.
        # Assim todos cbs terao essas variaves ex. title pela classe.
        # a classe mae pode expor tudo com @properties, @classmethod ou super() -> ver como melhor
        #
        self.title = title
        if 'save_path' in config:
            self.save_path = config['save_path']
            if not self.save_path.endswith('/'):
                self.save_path += '/'
        else:
            self.save_path = ''
        self.models_dir = f'{self.save_path}models_{name}'
        self.plots_dir = f'{self.save_path}plots_{name}'
        if os.path.isdir(f'{self.models_dir}') is False:
            os.makedirs(f'{self.models_dir}')
            print(f'Creating dir {self.models_dir}')
        if os.path.isdir(f'{self.plots_dir}') is False:
            os.makedirs(f'{self.plots_dir}')
            print(f'Creating dir {self.plots_dir}')

        self.save_last = config['save_last'] if 'save_last' in config else False
        self.save_best = config['save_best'] if 'save_best' in config else True
        self.show_plots = config['show_plots'] if 'show_plots' in config else True
        self.make_plots = config['make_plots'] if 'make_plots' in config else True

        self.best_val_acc = 0.05
        self.best_val_acc_ep = 0

    def __repr__(self):
        return 'BASE_Train'

    def begin_train_val(self, epochs, model, train_dataloader, val_dataloader, bs_size, optimizer):
        super().begin_train_val(epochs)
        train_step = len(train_dataloader)
        val_step = len(val_dataloader)
        # of break line, fix here
        #self.bar_step = train_step // 50 if train_step >= 50 else 1
        if train_step < 50:
            self.bar_step = 1
        elif train_step >= 50 and train_step < 100:
            self.bar_step = 2
        else:
            self.bar_step = train_step // 50
        self.bar_step_val = val_step // 10 if val_step >= 10 else 1
        #self.bar_step_val = self.bar_step #// 4 if val_step >= 1 else 1
        print(f'Fix progress: train_step: {train_step} val_step: {val_step}, bsize: {bs_size}, bar_step:{self.bar_step} bar_step_val:{self.bar_step_val}')
        self.total_train_samples, self.total_val_samples = 0, 0
        self.n_epoch = 0
        self.history = []
        self.start = time.time()
        return True

    def begin_epoch(self, current_epoch):
        self.train_loss, self.train_acc = 0., 0.
        self.val_loss, self.val_acc = 0., 0.
        self.n_train_samples, self.n_val_samples = 0, 0
        self.n_iter = 0
        self.n_epoch = current_epoch
        self.epoch_start = time.time()
        print(f'\nEpoch: {self.n_epoch}/{self.epochs}')
        return True

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        self._model = model

        # fing average training loss and accuracy
        avg_train_loss = train_loss/self.n_train_samples
        avg_train_acc = train_acc/self.n_train_samples

        # find average validation and loss
        avg_val_loss = val_loss/self.n_val_samples
        avg_val_acc = val_acc/self.n_val_samples

        self.total_train_samples += self.n_train_samples
        self.total_val_samples += self.n_val_samples

        self.history.append([avg_train_loss, avg_val_loss,
                             avg_train_acc, avg_val_acc])

        if avg_val_acc > self.best_val_acc:
            print(f' |------>  Best Val Acc model now {avg_val_acc:1.4f}')
            if self.save_best:
                self._best_model = copy.deepcopy(model)  # Will work
            self.best_val_acc = avg_val_acc
            self.best_val_acc_ep = self.n_epoch
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
        if (self.n_iter % self.bar_step) == 0:
            print('▒', end='', flush=True)
        return True

    def after_step_val(self, n_samples, *args):
        self.n_val_samples += n_samples
        self.n_iter += 1
        if (self.n_iter % self.bar_step_val) == 0:
            print('░', end='', flush=True)
        return True

    def after_train_val(self):
        elapsed_mins = (time.time()-self.start)/60
        print('Total training time:  {:.2f} mins.'.format(elapsed_mins))
        ts = time.time()
        n_samples = int((self.total_train_samples + self.total_val_samples)/self.epochs)
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')
        summary = str(st)+'_'+str(self.epochs)+'ep_'+str(n_samples)+'n'

        result_text = f"Best ACC: {self.best_val_acc:1.4f} (@ep {self.best_val_acc_ep})"
        acc_value = f'{self.best_val_acc:1.4f}'[-4:]
        print(result_text)

        # save the model
        if self.save_last:
            torch.save(self._model.state_dict(),
                       f'{self.models_dir}/{summary}_model.pt')
            print(f'cb_base: Last model saved in {self.models_dir}/')
        if self.save_best:
            torch.save(self._best_model.state_dict(),
                       f'{self.models_dir}/{summary}_best_model_ACC_0{acc_value[-4:]}.pt')
            print(f'cb_base: Best acc model saved in {self.models_dir}/')


        # plots
        if self.make_plots:
            history = np.array(self.history)
            plt.plot(history[:, 0:2])
            plt.title(self.title+" - Loss")
            plt.legend(['Tr Loss', 'Val Loss'], loc="upper right")
            plt.xlabel('Epoch Number')
            plt.ylabel('Loss')
            plt.ylim(0, 3)
            plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
            plt.savefig(f'{self.plots_dir}/{st}_loss_curve_ACC_0{acc_value}.png')
            if self.show_plots:
                plt.show()
            plt.clf()

            plt.plot(history[:, 2:4])
            plt.title(self.title+' - '+result_text)
            plt.legend(['Tr Accuracy', 'Val Accuracy'], loc="lower right")
            plt.xlabel('Epoch Number')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
            plt.savefig(f'{self.plots_dir}/{st}_acc_curve_ACC_0{acc_value}.png')
            plt.text(0, 0.9, result_text, bbox=dict(facecolor='red', alpha=0.3))
            if self.show_plots:
                plt.show()
            plt.clf()

        return True

    # Workaround para passar modelo
    @property
    def last_model(self):
        return self._model

    @property
    def best_model(self):
        return self._best_model
