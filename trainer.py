from matplotlib.pyplot import show
import torch
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score

from callbacks.cb_handler import CallbackHandler
from callbacks.cb_base import BaseCB
from callbacks.cb_lr_patch_clf import LR_SchedCB_patch
from callbacks.cb_lr_full_clf import LR_SchedCB_full
from callbacks.cb_lr_2views_clf import LR_SchedCB_2views
from callbacks.cb_lr_w_cyc_cos import LR_SchedCB_W_Cyc_Cos
from callbacks.cb_lr_w_cos import LR_SchedCB_W_Cos
from callbacks.cb_auc import AUC_CB

# from parallel import DataParallelModel, DataParallelCriterion
from util.util import show_auc, calc_auc_desv

parallel = False

#APAGAR
import cv2

# Accuracy
def acc(y_hat, labels):
    """ Default accuracy """

    # para parallel
    if len(y_hat) > 1 and parallel:
        y_hat = torch.cat(y_hat)

    return (torch.argmax(y_hat, dim=1) == labels).float().sum()


class Trainer():
    """
    Many possible configurations for Trainer
    config = {
        'num_epochs': NUM_EPOCHS,
        'batch_size': MINI_BATCH,
        'name': 'example',
        'title': 'Cats & Dogs Classifier',
        'save_last': True,          # optional: Save last model (default=False)
        'save_best': True,          # optional: Save best models (ACC, {AUC}) (default=True)
        'stable_metric: N           # optional: extend epochs number to wait N epochs with no metric change (ex.AUC)
        'save_checkpoints': N,      # Save checkpoint each N epochs
        'features': ['auc'],        # optional: features like auc stats or some scheduler (if none default:optim)
        'save_path': folder,        # if want to save artifacts in other place (eg.cloud)
        'show_plots': False,        # if want to show plots
        'make_plots': False,        # if want to disable plots
        'cv_k': (number),           # interactio number if using Cross Validation
    }
    """

    def __init__(self, model, train_dataloader, val_dataloader,
                 loss_criterion, optimizer, optimizer_args,
                 device, config):
        self.model = model
        self.device = device
        self.loss_criterion = loss_criterion

        # parts of config are only retrieved in callbacks
        self.epochs = int(config['num_epochs']) if 'num_epochs' in config else 10
        self.mini_batch = int(config['batch_size']) if 'batch_size' in config else 1
        self.first_epoch = int(config['start_epoch']) if 'start_epoch' in config else 1
        self.stable_metric = int(config['stable_metric']) if 'stable_metric' in config else False
        self.name = config['name'] if 'name' in config else 'default'
        self.title = config['title'] if 'title' in config else 'Classifier'
        self.features = config['features'] if 'features' in config else []
        self.make_plots = config['make_plots'] if 'make_plots' in config else True

        if train_dataloader:
            self.train_dataloader = train_dataloader
        else:
            return

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        print(self.title)

        # Load Callbacks for this session
        callbacks = [BaseCB(self.name, self.title, config)]
        for feat in self.features:
            if feat == 'auc':
                callbacks.append(AUC_CB(self.name, config))
            if feat == 'lr_step_full':
                callbacks.append(LR_SchedCB_full())
            if feat == 'lr_step_patch':
                callbacks.append(LR_SchedCB_patch())
            if feat == 'lr_step_2views':
                callbacks.append(LR_SchedCB_2views())
            if feat == 'lr_warmup_cos':
                callbacks.append(LR_SchedCB_W_Cos())
            if feat == 'lr_warmup_cyc_cos':
                callbacks.append(LR_SchedCB_W_Cyc_Cos())
            if feat == 'LR_SchedCB_W_Cos':
                callbacks.append(LR_SchedCB_W_Cos())
        self.cb = CallbackHandler(callbacks)


    def train_and_validate(self, **kwargs):
        """
        Main train and validate function that runs main loop (fit).
        Receives all parameters and feed callback system.
        Loop through epochs and executes pytorch forward, loss,
        backpropagation and optimization (grads calc).
        Returns the model trained.
        """

        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        input_dict = kwargs.get('input_dict') if kwargs.get('input_dict') else []

        if not self.cb.begin_train_val(self.epochs, self.model, self.train_dataloader,
                                       self.val_dataloader, self.mini_batch, self.optimizer):
            return

        self.cb.update_loss(self.loss_criterion, calc_acc)

        device = self.device

        for epoch in range(self.first_epoch, self.epochs+1):
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            val_loss, val_acc = 0.0, 0.0

            if not self.cb.begin_epoch(epoch): return  # noqa: E701

            optim = self.cb.update_LR(epoch, self.model, self.optimizer, self.optimizer_args)
            if optim: self.optimizer = optim

            # Train loop
            for _, (inputs, labels) in enumerate(self.train_dataloader):

                if isinstance(inputs, dict):
                    for key in input_dict:
                        inputs[key] = inputs[key].to(device)
                else:
                    inputs = Variable(inputs.to(device))

                labels = Variable(labels.to(device))

                # inserting MIXUP handling
                res = self.cb.begin_batch(inputs, labels)
                if res: inputs, labels, self.loss_criterion, calc_acc = res

                self.optimizer.zero_grad()                   # clean existing gradients
                outputs = self.model(inputs)                 # forward pass
                loss = self.loss_criterion(outputs, labels)  # compute loss
                if parallel:
                    loss = loss.mean()                       # list in this case
                loss.backward()                              # backprop the gradients
                self.optimizer.step()                        # update parameters
                train_loss += loss.item() * labels.size(0)   # inputs.size(0) == mini_batch size
                train_acc += calc_acc(outputs, labels).item()

                self.cb.after_step(labels.size(0), labels, outputs)

            # validation - no gradient tracking needed
            with torch.no_grad():
                self.model.eval()
                self.cb.begin_val()

                # validation loop
                for _, (inputs, labels) in enumerate(self.val_dataloader):

                    if isinstance(inputs, dict):
                        for key in input_dict:
                            inputs[key] = inputs[key].to(device)
                    else:
                        inputs = Variable(inputs.to(device))

                    labels = Variable(labels.to(device))

                    outputs = self.model(inputs)                 # forward pass
                    loss = self.loss_criterion(outputs, labels)  # compute loss
                    if parallel:
                        loss = loss.mean()
                    val_loss += loss.item() * labels.size(0) # inputs.size(0) == mini_batch size
                    val_acc += calc_acc(outputs, labels).item()

                    self.cb.after_step_val(labels.size(0), labels, outputs)

            self.cb.after_epoch(self.model, train_acc, train_loss, val_acc, val_loss)

        self.cb.after_train_val()

        return self.model

    def train_and_validate_amp(self, **kwargs):
        """
        Mixed precision (automatic) version for train_and_validate.
        Uses FP16 and FP32 in main loop with pytorch Automatic Mixed Precision.
        In simple tests: use 75% of memory in 66% of time. Less memory and faster.
        Sometimes it just don't work and get worse, like for resnest...
        """

        assert torch.__version__ >= '1.6.0', "[Mixed precision] Please use PyTorch 1.6.0+"

        print('Using AMP')

        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        input_dict = kwargs.get('input_dict') if kwargs.get('input_dict') else []

        if not self.cb.begin_train_val(self.epochs, self.model, self.train_dataloader,
                                       self.val_dataloader, self.mini_batch, self.optimizer):
            return

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        device = self.device

        # for epoch in range(self.first_epoch, self.epochs+1):
        epoch = self.first_epoch        # suport for "wait N epochs after best metric"
        last_epoch = self.epochs
        while epoch <= last_epoch:
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            val_loss, val_acc = 0.0, 0.0

            if not self.cb.begin_epoch(epoch): return  # noqa: E701

            optim = self.cb.update_LR(epoch, self.model, self.optimizer, self.optimizer_args)
            if optim: self.optimizer = optim

            # Train loop
            for _, (inputs, labels) in enumerate(self.train_dataloader):

                if isinstance(inputs, dict):
                    for key in input_dict:
                        inputs[key] = inputs[key].to(device)
                else:
                    inputs = Variable(inputs.to(device))

                labels = Variable(labels.to(device))

                self.optimizer.zero_grad()                       # clean existing gradients
                # Runs the forward pass with autocasting.
                with autocast():
                    outputs = self.model(inputs)                 # forward pass
                    loss = self.loss_criterion(outputs, labels)  # compute loss
                if parallel:
                    loss = loss.mean()          # list in this case
                scaler.scale(loss).backward()   # backward() on scaled loss for scaled gradients.        
                scaler.step(self.optimizer)     # update parameters
                scaler.update()                 # Updates the scale for next iteration.

                train_loss += loss.item() * labels.size(0)           # == mini_batch size
                train_acc += calc_acc(outputs, labels).item()

                self.cb.after_step(labels.size(0), labels, outputs)

            # validation - no gradient tracking needed
            with torch.no_grad():
                self.model.eval()

                # validation loop
                for _, (inputs, labels) in enumerate(self.val_dataloader):

                    if isinstance(inputs, dict):
                        for key in input_dict:
                            inputs[key] = inputs[key].to(device)
                    else:
                        inputs = Variable(inputs.to(device))

                    labels = Variable(labels.to(device))

                    outputs = self.model(inputs)                 # forward pass
                    loss = self.loss_criterion(outputs, labels)  # compute loss
                    if parallel:
                        loss = loss.mean()
                    val_loss += loss.item() * labels.size(0)     # == mini_batch size
                    val_acc += calc_acc(outputs, labels).item()

                    self.cb.after_step_val(labels.size(0), labels, outputs)

            self.cb.after_epoch(self.model, train_acc, train_loss, val_acc, val_loss)

            epoch += 1
            last_epoch = self.epochs if not self.stable_metric else max(self.epochs, self.cb.best_metric_epoch + self.stable_metric)
            # print('-', self.cb.best_metric_epoch, last_epoch)

        self.cb.after_train_val()

        values = [self.cb.best_metric, self.cb.best_metric_epoch, self.cb.elapsed_mins, 
                  self.cb.metric_name, self.cb.loss_plot, self.cb.metric_plot, 
                  self.cb.best_model_file]

        return values


    def run_test(self, test_dataloader, model_type, **kwargs):
        """ Run test from test_dataloader according to model_type.
            if model_type = 'normal' : use last saved model
            if model_type = 'best' : use best model
            Uses: loss function from Trainer
            Input: test_dataloader
        """
        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        quiet = kwargs.get('quiet') if kwargs.get('quiet') else False

        if model_type == 'normal':
            model = self.cb.last_model
        elif model_type == 'best':
            model = self.cb.best_model
        elif model_type == 'bootstrap':
            model = self.model

        test_acc, test_loss = 0., 0.
        batch_val_counter = 0
        device = self.device

        with torch.no_grad():
            model.eval()

            # validation loop
            for _, (inputs, labels) in enumerate(test_dataloader):

                if isinstance(inputs, dict):
                    for key in ['CC', 'MLO']:
                        inputs[key] = inputs[key].to(device)
                    labels = Variable(labels.to(device))
                else:
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))

                outputs = model(inputs)                         # forward pass
                loss = self.loss_criterion(outputs, labels)     # compute loss
                if parallel:
                    loss = loss.mean()
                test_loss += loss.item() * labels.size(0)
                test_acc += calc_acc(outputs, labels).item()

                batch_val_counter += labels.size(0)

        # Find average test loss and test accuracy
        avg_test_loss = test_loss/batch_val_counter
        avg_test_acc = test_acc/batch_val_counter

        if not quiet:
            print(f'Model: {model_type} - Test accuracy : {avg_test_acc:.5f}' +
                  f' Test loss : {avg_test_loss:.5f}')

        return avg_test_acc 


    def run_test_auc(self, test_dataloader, model_type, **kwargs):
        """ Run test from test_dataloader, calculating AUC and ROC curve
            According to model_type:
            if model_type = 'normal' : use last saved model
            if model_type = 'best' : use best model
            If we are running test iunference only can pass model through kwargs.
            Uses: loss function from Trainer
            Input: test_dataloader
        """
        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        model = kwargs.get('model') if kwargs.get('model') else None
        show_results = kwargs.get('show_results') if kwargs.get('show_results') else False
        m_positive = kwargs.get('m') if kwargs.get('m') else False
        n_negative = kwargs.get('n') if kwargs.get('n') else False

        if model is None:
            if model_type == 'normal':
                model = self.cb.last_model
            elif model_type == 'best':
                model = self.cb.best_model
            elif model_type == 'test':
                model = self.model
            elif model_type == 'bootstrap':
                model = self.model

        test_acc, test_loss = 0., 0.
        batch_val_counter = 0
        y_hat_auc, label_auc = [], []
        device = self.device

        with torch.no_grad():
            model.eval()

            # validation loop
            for _, (inputs, labels) in enumerate(test_dataloader):
                if isinstance(inputs, dict):
                    for key in ['CC', 'MLO']:
                        inputs[key] = inputs[key].to(device)
                    labels = Variable(labels.to(device))
                else:
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))
                outputs = model(inputs)                         # forward pass
                loss = self.loss_criterion(outputs, labels)     # compute loss
                test_loss += loss.item() * labels.size(0)

                # calculate acc
                test_acc += calc_acc(outputs, labels).item()
                batch_val_counter += labels.size(0)

                # Store auc for malignant
                label_auc = np.append(label_auc, labels.cpu().detach().numpy())
                y_hat_auc = np.append(y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

                # enter show result mode
                if self.mini_batch == 1 and show_results:
                    print(f'{labels.item()}  {torch.softmax(outputs, dim=1)[:, 1].item():.3f}')

        # Find average test loss and test accuracy
        avg_test_loss = test_loss/batch_val_counter
        avg_test_acc = test_acc/batch_val_counter

        print(f"Model: {model_type} - Test accuracy : {avg_test_acc:.3f}" +
              f" Test loss : {avg_test_loss:.4f}", end='')

        # calculate AUC TEST
        auc_mal_val = roc_auc_score(label_auc.ravel(), y_hat_auc.ravel())
        print(f' AUC Malignant: {auc_mal_val:.4f}', end='')
        if m_positive and n_negative:
            print(f'±{calc_auc_desv(m_positive, n_negative, auc_mal_val):.4f}')
        else:
            print()


        if self.make_plots:
            show_auc(label_auc, y_hat_auc, self.title, show_plt=False)
        
        return auc_mal_val


    # Not fully tested yet (2021-05)
    # it seems to be working - maybe integrate in single function as above
    #  and use kwargs to indicate that it is test-data- aug?
    def run_test_data_aug_auc(self, test_dataloader, model_type, **kwargs):
        """ Run test from test_dataloader, calculating AUC and ROC curve
            --> Using test-data augmentation: rotation 0°, 90°, 180°, 270°
            --> All rotated sample will be infered and AUC will consider all.
            According to model_type:
            if model_type = 'normal' : use last saved model
            if model_type = 'best' : use best model
            If we are running test iunference only can pass model through kwargs.
            Uses: loss function from Trainer
            Input: test_dataloader
        """
        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        model = kwargs.get('model') if kwargs.get('model') else None

        if model is None:
            if model_type == 'normal':
                model = self.cb.last_model
            elif model_type == 'best':
                model = self.cb.best_model
            elif model_type == 'test':
                model = self.model

        test_acc, test_loss = 0., 0.
        batch_val_counter = 0
        y_hat_auc, label_auc = [], []
        device = self.device

        with torch.no_grad():
            model.eval()

            # validation loop
            for _, (inputs, labels) in enumerate(test_dataloader):
                for rot in range(0,4):
                    
                    # print(rot, inputs.shape)
                    inputs = torch.rot90(inputs, rot, [2, 3])
                    # inputs = Variable(inputs.to(device))
                    # labels = Variable(labels.to(device))
                    # print(counter, rot, inputs.shape)

                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))

                    # img = inputs.cpu().detach().numpy()
                    # img = img.transpose(0,2,3,1)
                    # print(img[0, :, :, 0:3].shape)
                    # cv2.imwrite('thrash/test-aug_'+str(rot)+'.png', img[0, :, :, 0:3]*65535)

                    outputs = model(inputs)                         # forward pass
                    loss = self.loss_criterion(outputs, labels)     # compute loss
                    test_loss += loss.item() * labels.size(0)

                    # calculate acc
                    test_acc += calc_acc(outputs, labels).item()
                    batch_val_counter += labels.size(0)

                    # Store auc for malignant
                    label_auc = np.append(label_auc, labels.cpu().detach().numpy())
                    y_hat_auc = np.append(y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

                    # enter show result mode
                    if self.mini_batch == 1:
                        print(f'{labels.item()}  {torch.softmax(outputs, dim=1)[:, 1].item():.3f}')

        print('batch_val_counter ', batch_val_counter)

        # Find average test loss and test accuracy
        avg_test_loss = test_loss/batch_val_counter
        avg_test_acc = test_acc/batch_val_counter

        print(f"Model: {model_type} - Test accuracy : {avg_test_acc:.3f}" +
              f" Test loss : {avg_test_loss:.4f}", end='')

        # calculate AUC TEST
        auc_mal_val = roc_auc_score(label_auc.ravel(), y_hat_auc.ravel())
        print(f' AUC Malignant: {auc_mal_val:.4f}')

        if self.make_plots:
            show_auc(label_auc, y_hat_auc, self.title, show_plt=False)
        
        return auc_mal_val
