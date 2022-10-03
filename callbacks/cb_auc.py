
# Calculate AUC for classification

# TODO: bring auc calculation to here (reading label and output)


import copy
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import wandb


from callbacks.cb import Callbacks    # base


class AUC_CB(Callbacks):
    def __init__(self, name, config):
        self.name = name
        self.show_plots = config['show_plots'] if 'show_plots' in config else True
        self.make_plots = config['make_plots'] if 'make_plots' in config else True
        self.save_best = config['save_best'] if 'save_best' in config else True
        self.cv_k = config['cv_k'] if 'cv_k' in config else False
        self.cv_support = True if 'cv_k' in config else False
        self.use_wandb = config['use_wandb'] if 'use_wandb' in config else False

        if 'save_path' in config:
            self.save_path = config['save_path']
            if not self.save_path.endswith('/'):
                self.save_path += '/'
        else:
            self.save_path = ''
        self.models_dir = f'{self.save_path}models_{name}'
        self.plots_dir = f'{self.save_path}plots_{name}'
        self.best_auc_ep = 0
        self.n_epoch = 0
        self._metric_name = 'AUC'

    def __repr__(self):
        return 'AUC_Stats'

    def begin_epoch(self, current_epoch):
        self.n_epoch = current_epoch
        self.y_hat_auc, self.label_auc = [], []
        self.y_hat_val_auc, self.label_val_auc = [], []

    def begin_train_val(self, epochs, model, train_dataloader, val_dataloader, bs_size, optimizer):
        super().begin_train_val(epochs)
        self.history = []
        self.best_auc = 0.3
        return True

    def after_step(self, n_samples, labels, outputs):
        self.label_auc = np.append(self.label_auc, labels.cpu().detach().numpy())
        self.y_hat_auc = np.append(self.y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
        # pegando soh malignant posicao [1] para comparar com label. ok.
        # print(outputs, torch.softmax(outputs, dim=1), torch.softmax(outputs, dim=1)[0][1])

        return True

    def after_step_val(self, n_samples, labels, outputs):
        # guarda Y-hat/label para AUC - VAL
        self.label_val_auc = np.append(self.label_val_auc, labels.cpu().detach().numpy())
        self.y_hat_val_auc = np.append(self.y_hat_val_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
        # arrumar essa linha de cima...

        return True

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):

        auc_malign_train, auc_malign_val = 0, 0

        # calculate AUC Train
        try:
            auc_malign_train = roc_auc_score(self.label_auc.ravel(),
                                            self.y_hat_auc.ravel())
        except:
            print('AUC Train calc error (possibly bad results): ')
            print('Labels: ', self.label_auc.ravel())
            with np.printoptions(threshold=np.inf, precision=2):
                print('Outputs: ', self.y_hat_auc.ravel())

        # calculate AUC VAL
        try:
            auc_malign_val = roc_auc_score(self.label_val_auc.ravel(),
                                        self.y_hat_val_auc.ravel())
        except:
            print('AUC Val calc error (possibly bad results): ')
            print('Labels: ', self.label_auc.ravel())
            with np.printoptions(threshold=np.inf, precision=2):
                print('Outputs: ', self.y_hat_auc.ravel())

        print('[Train] AUC Malignant: %.3f' % auc_malign_train, end='')
        print(' [Val] AUC Val Malignant: %.3f' % auc_malign_val, end='')

        self.history.append([auc_malign_train, auc_malign_val])
        if (auc_malign_val > self.best_auc):
            print(f' |------>  Best Val Auc model now {auc_malign_val:1.4f}')
            self.best_model = copy.deepcopy(model)  # Will work
            self.best_auc = auc_malign_val
            self.best_auc_ep = self.n_epoch
            # self._best_metric_epoch = self.n_epoch
            # self._best_metric = auc_malign_val
        else: print()   # noop

        if self.use_wandb:
            wandb.log({"auc_train": auc_malign_train, "auc_val": auc_malign_val})

        return True

    def after_train_val(self):

        ts = time.time()
        # n_samples = self.total_train_samples + self.total_val_samples
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')
        summary = str(st)+'_'+str(self.epochs)+'ep'#+str(n_samples)+'n'

        # Cross validation sufix, if configured
        if self.cv_support:
            cv_sufix = '_cv_'+str(self.cv_k)
        else:
            cv_sufix = ''

        result_auc = f"Best AUC: {self.best_auc:1.4f}  (@ep {self.best_auc_ep}) {cv_sufix}"
        auc_value = f'{self.best_auc:1.4f}'[-4:]
        print(result_auc)

        if self.use_wandb:
            wandb.summary['best_auc_val'] = self.best_auc
            wandb.summary['best_auc_val_epoch'] = self.best_auc_ep

        # save the model
        if self.save_best and hasattr(self, 'best_model'):
            self._best_model_file = f'{self.models_dir}/{summary}_best_model_AUC_0{auc_value}{cv_sufix}.pt'
            torch.save(self.best_model.state_dict(),
                    self._best_model_file)
            print(f'cb_auc: Best auc model saved in {self.models_dir}/')

        if self.make_plots:
            history = np.array(self.history)
            plt.plot(history[:, 0:2])
            plt.title(f'AUC - Classifier {self.name} {result_auc}')
            plt.legend(['Train AUC', 'Val AUC'], loc="lower right")
            plt.xlabel('Epoch Number')
            plt.ylabel('Area Under Curve')
            plt.ylim(0, 1)
            plt.grid(True, ls=':', lw=.5, c='k', alpha=.3)
            plt.text(0, 0.95, result_auc, bbox=dict(facecolor='red', alpha=0.3))
            self._auc_plot = f'{self.plots_dir}/{st}_AUC_curve_AUC_0{auc_value}.png'
            plt.savefig(self._auc_plot)
            if self.use_wandb:
                wandb.log({'img': [wandb.Image(plt)]})
            if self.show_plots and not self.use_wandb:
                plt.show()
            plt.clf()

        return True

    @property
    def best_metric_epoch(self):
        # print('AUC in ', self._best_metric_epoch)
        return self.best_auc_ep #_best_metric_epoch

    @property
    def best_metric(self):
        return self.best_auc #_best_metric

    @property
    def best_auc_model(self):
        return self.best_model

    @property
    def loss_plot(self):
        return 'NA'

    @property
    def metric_plot(self):
        return self._auc_plot

    @property
    def best_model_file(self):
        return self._best_model_file

    @property
    def metric_name(self):
        return self._metric_name

    @property
    def elapsed_mins(self):
        return 0