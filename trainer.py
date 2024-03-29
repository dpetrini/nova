import numpy as np
import torch
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb

from callbacks.cb_handler import CallbackHandler
from callbacks.cb_base import BaseCB
from callbacks.cb_lr_patch_clf import LR_SchedCB_patch
from callbacks.cb_lr_full_clf import LR_SchedCB_full
from callbacks.cb_lr_2views_clf import LR_SchedCB_2views
from callbacks.cb_lr_w_cyc_cos import LR_SchedCB_W_Cyc_Cos
from callbacks.cb_lr_w_cos import LR_SchedCB_W_Cos
from callbacks.cb_auc import AUC_CB

# from parallel import DataParallelModel, DataParallelCriterion
from util.util import show_auc, calc_auc_desv, plot_confusion_matrix2

parallel = False

# Accuracy
def acc(y_hat, labels):
    """ Default accuracy """

    # # para parallel
    # if len(y_hat) > 1 and parallel:
    #     y_hat = torch.cat(y_hat)

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
        'save_best': True,          # optional: Save best models (ACC, {AUC}) (default=True) - need for AUC  & other
        'save_best_acc': False,     # optional: Save best models Acc (default=False) -
        'stable_metric: N           # optional: extend epochs number to wait N epochs with no metric change (ex.AUC)
        'save_checkpoints': N,      # Save checkpoint each N epochs
        'features': ['auc'],        # optional: features like auc stats or some scheduler (if none default:optim)
        'save_path': folder,        # if want to save artifacts in other place (eg.cloud)
        'show_plots': False,        # if want to show plots
        'make_plots': False,        # if want to disable plots
        'cv_k': (number),           # interactio number if using Cross Validation
        #'resume': False,            # resume training from saved checkpoint
        'use_wandb': False,         # Save metrics & models in Wand DB
        'name_sufix': '',           # Append sufix name in best model file name
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
        self.use_wandb = config['use_wandb'] if 'use_wandb' in config else False
        # self.resume = config['resume'] if 'resume' in config else False

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
        Function optimized for performances in many ways.
        """

        assert torch.__version__ >= '1.6.0', "[Mixed precision] Please use PyTorch 1.6.0+"

        print('Using AMP')

        calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
        #input_dict = kwargs.get('input_dict') if kwargs.get('input_dict') else []
        return_dict = kwargs.get('return_dict') if kwargs.get('return_dict') else False
        aug_func = kwargs.get('aug_func') if kwargs.get('aug_func') else None

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
            # train_loss, train_acc = 0.0, 0.0
            train_loss = torch.tensor(0.0, device=torch.device('cuda:0'), requires_grad=False)
            train_acc = torch.tensor(0.0, device=torch.device('cuda:0'), requires_grad=False)
            bs_size = torch.tensor(self.mini_batch, device=torch.device('cuda:0'), requires_grad=False)
            # val_loss, val_acc = 0.0, 0.0
            val_loss = torch.tensor(0.0, device=torch.device('cuda:0'), requires_grad=False)
            val_acc = torch.tensor(0.0, device=torch.device('cuda:0'), requires_grad=False)

            if not self.cb.begin_epoch(epoch): return  # noqa: E701

            optim = self.cb.update_LR(epoch, self.model, self.optimizer, self.optimizer_args)
            if optim: self.optimizer = optim

            # Train loop
            for _, (inputs, labels) in enumerate(self.train_dataloader):

                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # check performance - 1
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)


                # print(inputs.mean().item(), inputs.std().item())
                # print('A ', inputs.shape, inputs[0].mean().item(), inputs[0].std().item(), labels[0].item(), inputs[0])

                if aug_func:
                    # inputs, labels = aug_func(inputs, labels, 'train')
                    inputs = aug_func(inputs)  # .to(device)
                
                # print(inputs.mean().item(), inputs.std().item())
                # for i in range(len(inputs)):
                #     print(i, inputs[i].mean().item(), inputs[i].std().item())
                #print('B ', inputs.shape, inputs[0].mean().item(), inputs[0].std().item(), labels[0].item(),  inputs[0])

                # self.optimizer.zero_grad() 

                # check performance - 2
                self.optimizer.zero_grad(set_to_none=True)                       # clean existing gradients

                # Runs the forward pass with autocasting.
                with autocast():
                    outputs = self.model(inputs)                 # forward pass
                    loss = self.loss_criterion(outputs, labels)  # compute loss
                if parallel:
                    loss = loss.mean()          # list in this case
                scaler.scale(loss).backward()   # backward() on scaled loss for scaled gradients.        
                scaler.step(self.optimizer)     # update parameters
                scaler.update()                 # Updates the scale for next iteration.


                # evitar uso de item(), que força sincronismo com CPU: TODO
                # check performance - 3
                # train_loss += loss.item() * labels.size(0)           # == mini_batch size
                train_loss += loss.data * bs_size           # == mini_batch size

                # print(train_loss, loss.item(), loss, labels.size(0), bs_size)

                # check performance - 4
                # train_acc += calc_acc(outputs, labels).item()
                train_acc += calc_acc(outputs, labels).data

                # check performance - 5
                # self.cb.after_step(labels.size(0), labels, outputs)
                self.cb.after_step(bs_size, labels, outputs)

            # validation - no gradient tracking needed
            with torch.no_grad():
                self.model.eval()

                # validation loop
                for _, (inputs, labels) in enumerate(self.val_dataloader):

                    # inputs = inputs.to(device)
                    # labels = labels.to(device)

                    # check performance - 3
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)               

                    if aug_func:
                        # inputs, labels = aug_func(inputs, labels, 'val')
                        inputs = aug_func(inputs)

                    outputs = self.model(inputs)                 # forward pass
                    loss = self.loss_criterion(outputs, labels)  # compute loss
                    if parallel:
                        loss = loss.mean()
                    # val_loss += loss.item() * labels.size(0)     # == mini_batch size
                    val_loss += loss.data * labels.size(0)     # == mini_batch size may be different from train
                    # val_acc += calc_acc(outputs, labels).item()
                    val_acc += calc_acc(outputs, labels).data

                    self.cb.after_step_val(labels.size(0), labels, outputs)

            self.cb.after_epoch(self.model, train_acc, train_loss, val_acc, val_loss)

            epoch += 1
            # print('-', self.cb.best_metric_epoch[self.cb.metric_name[-1]], last_epoch)
            # Is use stable metric - will stop training earlier, after 
            #  stable_metric epochs without validation metric (to be selected) improve
            # last_epoch = self.epochs if not self.stable_metric else max(self.epochs, self.cb.best_metric_epoch[self.cb.metric_name[-1]] + self.stable_metric)
            # for metric in self.cb.metric_name:
            #     print(metric)
            last_epoch = self.epochs if not self.stable_metric else min(self.epochs, self.cb.best_metric_epoch[self.cb.metric_name[-1]] + self.stable_metric)

        self.cb.after_train_val()

        if return_dict:
            values_dic = {'best_metric': self.cb.best_metric, 'best_metric_epoch': self.cb.best_metric_epoch,
                          'elapsed_mins': self.cb.elapsed_mins, 'metric_name': self.cb.metric_name, 
                          'loss_plot': self.cb.loss_plot, 'metric_plot': self.cb.metric_plot,
                          'best_model_file': self.cb.best_model_file}
            return values_dic

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
        aug_func = kwargs.get('aug_func') if kwargs.get('aug_func') else None

        if model_type == 'normal':
            model = self.cb.last_model
        elif model_type == 'best':
            model = self.cb.best_model
        elif model_type == 'bootstrap':
            model = self.model
        elif model_type == 'test':
            model = self.model
        else:
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
                if aug_func:
                    # inputs, labels = aug_func(inputs, labels, 'test')
                    inputs = aug_func(inputs)

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
        quiet = kwargs.get('quiet') if kwargs.get('quiet') else False
        aug_func = kwargs.get('aug_func') if kwargs.get('aug_func') else None

        if self.cb.best_metric['AUC'] == 0.3:     # Default Auc, no update
            print('Wrong values for AUC test returning zero...')
            return 0                    # possibly only errors return 0

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

            # test loop
            for _, (inputs, labels) in enumerate(test_dataloader):
                if isinstance(inputs, dict):
                    for key in ['CC', 'MLO']:
                        inputs[key] = inputs[key].to(device)
                    labels = Variable(labels.to(device))
                else:
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device))
                if aug_func:
                    # inputs, labels = aug_func(inputs, labels, 'test')
                    inputs = aug_func(inputs)

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

        if not quiet:
            print(f"Model: {model_type} - Test accuracy : {avg_test_acc:.3f}" +
                f" Test loss : {avg_test_loss:.4f}", end='')

        # calculate AUC TEST
        try:
            auc_mal_val = roc_auc_score(label_auc.ravel(), y_hat_auc.ravel())
        except:
            auc_mal_val = 0
            print('Bad AUC, probably bad auc values during training... skip it!')

        if m_positive and n_negative:
            auc_final = f'{auc_mal_val:.4f}±{calc_auc_desv(m_positive, n_negative, auc_mal_val):.4f}'
            # print(f'±{calc_auc_desv(m_positive, n_negative, auc_mal_val):.4f}')
            print(f' AUC Malignant: {auc_final}')
        else:
            auc_final = f'{auc_mal_val:.4f}'
            if not quiet:
                print(f' AUC Malignant: {auc_final}')
            # print()

        if self.use_wandb and model_type != 'bootstrap':
            wandb.summary['auc_test'] = auc_final

        if self.make_plots:
            show_auc(label_auc, y_hat_auc, self.title, show_plt=False)

        return auc_final

    def run_test_auc_fast(self, test_dataloader, aug_func):
        """ Run test from test_dataloader, calculating AUC
            FASTER to run multiple times in bootstrap
            If we are running test iunference only
            Uses: test data loader
            Input: test_dataloader
        """
        model = self.model
        y_hat_auc, label_auc = [], []
        device = self.device

        with torch.no_grad():
            model.eval()
            # Test loop
            for _, (inputs, labels) in enumerate(test_dataloader):
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))
                if aug_func:
                    inputs, labels = aug_func(inputs, labels, 'test')
                outputs = model(inputs)
                # Store prediction for malignant (second class)
                label_auc = np.append(label_auc, labels.cpu().detach().numpy())
                y_hat_auc = np.append(y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

        # calculate AUC TEST
        return roc_auc_score(label_auc.ravel(), y_hat_auc.ravel())


    def run_test_cm(self, test_dataloader, model_type, **kwargs):
        """ Run test from test_dataloader, calculating CONFUSION MATRIX
            labels: labels to plot
            show_results: True if show plot (good for notebooks)
            title: Used for file name
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
        labels_text = kwargs.get('labels') if kwargs.get('labels') else None
        title = kwargs.get('title') if kwargs.get('title') else ''
        aug_func = kwargs.get('aug_func') if kwargs.get('aug_func') else None

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
                if aug_func:
                    # inputs, labels = aug_func(inputs, labels, 'test')
                    inputs = aug_func(inputs)
                    
                outputs = model(inputs)                         # forward pass
                loss = self.loss_criterion(outputs, labels)     # compute loss
                test_loss += loss.item() * labels.size(0)

                # calculate acc
                test_acc += calc_acc(outputs, labels).item()
                batch_val_counter += labels.size(0)

                # Store auc for malignant
                label_auc = np.append(label_auc, labels.cpu().detach().numpy())
                y_hat_auc = np.append(y_hat_auc, torch.argmax(outputs, dim=1).cpu().detach().numpy())

        if labels is None:
            print('Please provide labels for Confusion Matrix.')
            return
        else:
            labels = labels_text
        cm = confusion_matrix(label_auc.ravel(), y_hat_auc.ravel())
        # print(label_auc.ravel().shape, y_hat_auc.ravel().shape)
        # print((label_auc.ravel() == y_hat_auc.ravel()).sum())
        # print(cm)
        if self.make_plots:
            plot_confusion_matrix2(cm, labels_text, title=title, normalize=True, show_plt=show_results)

        if self.use_wandb:
            cm = wandb.plot.confusion_matrix(
                y_true=label_auc,
                preds=y_hat_auc,
                class_names=labels_text)
            wandb.log({"conf_mat": cm})

        return

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
