import torch
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score

# from parallel import DataParallelModel, DataParallelCriterion
from util.util import show_auc

parallel = False


# Accuracy
def acc(y_hat, labels):
    """ Default accuracy """

    # para parallel
    if len(y_hat) > 1 and parallel:
        y_hat = torch.cat(y_hat)

    return (torch.argmax(y_hat, dim=1) == labels).float().sum()


def train_and_validate(model, train_dataloader, val_dataloader,
                       loss_criterion, optimizer, optimizer_args,
                       mini_batch, epochs, first_epoch, device,
                       cb, **kwargs):
    """
    Main train and validate function that runs main loop (fit).
    Receives all parameters and feed callback system.
    Loop through epochs and executes pytorch forward, loss,
    backpropagation and optimization (grads calc).
    Returns the model trained.
    """

    calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc
    input_dict = kwargs.get('input_dict') if kwargs.get('input_dict') else []

    if not cb.begin_train_val(epochs, model, train_dataloader,
                              val_dataloader, mini_batch, optimizer):
        return

    cb.update_loss(loss_criterion, calc_acc)

    for epoch in range(first_epoch, epochs+1):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0

        if not cb.begin_epoch(epoch): return  # noqa: E701

        optim = cb.update_LR(epoch, model, optimizer, optimizer_args)
        if optim: optimizer = optim

        # Train loop
        for _, (inputs, labels) in enumerate(train_dataloader):

            if isinstance(inputs, dict):
                for key in input_dict:
                    inputs[key] = inputs[key].to(device)
            else:
                inputs = Variable(inputs.to(device))

            labels = Variable(labels.to(device))

            # inserting MIXUP handling
            res = cb.begin_batch(inputs, labels)
            if res: inputs, labels, loss_criterion, calc_acc = res

            # print('LE. ', inputs.shape, inputs.type())

            optimizer.zero_grad()                   # clean existing gradients
            outputs = model(inputs)                 # forward pass
            loss = loss_criterion(outputs, labels)  # compute loss
            if parallel:
                loss = loss.mean()                  # list in this case
            loss.backward()                         # backprop the gradients
            optimizer.step()                        # update parameters
            train_loss += loss.item() * labels.size(0) # inputs.size(0) mini_batch
            train_acc += calc_acc(outputs, labels).item()

            cb.after_step(labels.size(0), labels, outputs) # inputs.size(0)

        # validation - no gradient tracking needed
        with torch.no_grad():
            model.eval()
            cb.begin_val()

            # validation loop
            for _, (inputs, labels) in enumerate(val_dataloader):

                if isinstance(inputs, dict):
                    for key in input_dict:
                        inputs[key] = inputs[key].to(device)
                else:
                    inputs = Variable(inputs.to(device))

                labels = Variable(labels.to(device))

                outputs = model(inputs)                 # forward pass
                loss = loss_criterion(outputs, labels)  # compute loss
                if parallel:
                    loss = loss.mean()
                val_loss += loss.item() * labels.size(0) # inputs.size(0) mini_batch
                val_acc += calc_acc(outputs, labels).item()

                cb.after_step_val(labels.size(0), labels, outputs) # inputs.size(0) mini_batch

        cb.after_epoch(model, train_acc, train_loss, val_acc, val_loss)

    cb.after_train_val()

    return model

def train_and_validate_amp(model, train_dataloader, val_dataloader,
                           loss_criterion, optimizer, optimizer_args,
                           mini_batch, epochs, first_epoch, device,
                           cb, **kwargs):
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

    if not cb.begin_train_val(epochs, model, train_dataloader,
                              val_dataloader, mini_batch, optimizer):
        return

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in range(first_epoch, epochs+1):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0

        if not cb.begin_epoch(epoch): return  # noqa: E701

        optim = cb.update_LR(epoch, model, optimizer, optimizer_args)
        if optim: optimizer = optim

        # Train loop
        for _, (inputs, labels) in enumerate(train_dataloader):

            if isinstance(inputs, dict):
                for key in input_dict:
                    inputs[key] = inputs[key].to(device)
            else:
                inputs = Variable(inputs.to(device))

            labels = Variable(labels.to(device))

            optimizer.zero_grad()                   # clean existing gradients
            # Runs the forward pass with autocasting.
            with autocast():
                outputs = model(inputs)                 # forward pass
                loss = loss_criterion(outputs, labels)  # compute loss
            if parallel:
                loss = loss.mean()                  # list in this case
            scaler.scale(loss).backward()   # Calls backward() on scaled loss for scaled gradients.             
            scaler.step(optimizer)                   # update parameters
            scaler.update()                 # Updates the scale for next iteration.

            train_loss += loss.item() * labels.size(0) # inputs.size(0) mini_batch
            train_acc += calc_acc(outputs, labels).item()

            cb.after_step(labels.size(0), labels, outputs)  # inputs.size(0)

        # validation - no gradient tracking needed
        with torch.no_grad():
            model.eval()

            # validation loop
            for _, (inputs, labels) in enumerate(val_dataloader):

                if isinstance(inputs, dict):
                    for key in input_dict:
                        inputs[key] = inputs[key].to(device)
                else:
                    inputs = Variable(inputs.to(device))

                labels = Variable(labels.to(device))

                outputs = model(inputs)                 # forward pass
                loss = loss_criterion(outputs, labels)  # compute loss
                if parallel:
                    loss = loss.mean()
                val_loss += loss.item() * labels.size(0) # inputs.size(0) mini_batch
                val_acc += calc_acc(outputs, labels).item()

                cb.after_step_val(labels.size(0), labels, outputs)  # inputs.size(0) mini_batch

        cb.after_epoch(model, train_acc, train_loss, val_acc, val_loss)

    cb.after_train_val()

    return model


def run_test_auc(model, loss_criterion, test_dataloader, device,
                 summary, batch_size, model_type, title, **kwargs):
    """ Run test from test_dataloader, calculating AUC and ROC curve"""

    calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc

    test_acc, test_loss = 0., 0.
    batch_val_counter = 0
    y_hat_auc, label_auc = [], []

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
            outputs = model(inputs)                     # forward pass
            loss = loss_criterion(outputs, labels)      # compute loss
            test_loss += loss.item() * labels.size(0) # inputs.size(0)   # batch total loss

            # calculate acc
            test_acc += calc_acc(outputs, labels).item()
            batch_val_counter += labels.size(0) #inputs.size(0)

            # Store auc for malignant
            label_auc = np.append(label_auc, labels.cpu().detach().numpy())
            y_hat_auc = np.append(y_hat_auc, torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

            # enter show result mode
            if batch_size == 1:
                print(f'{labels.item()}  {torch.softmax(outputs, dim=1)[:, 1].item():.3f}')

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/batch_val_counter
    avg_test_acc = test_acc/batch_val_counter

    print(f"Model: {model_type} - Test accuracy : {avg_test_acc:.3f}" +
          f" Test loss : {avg_test_loss:.3f}", end='')

    # calculate AUC TEST
    auc_mal_val = roc_auc_score(label_auc.ravel(), y_hat_auc.ravel())
    print(f' AUC Malignant: {auc_mal_val:.3f}')

    show_auc(label_auc, y_hat_auc, title, show_plt=False)


def run_test(model, loss_criterion, test_dataloader, device,
             summary, batch_size, model_type, **kwargs):
    """ Run test from test_dataloader """

    calc_acc = kwargs.get('accuracy') if kwargs.get('accuracy') else acc

    test_acc, test_loss = 0., 0.
    batch_val_counter = 0

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

            outputs = model(inputs)                     # forward pass
            loss = loss_criterion(outputs, labels)      # compute loss
            if parallel:
                loss = loss.mean()
            test_loss += loss.item() * labels.size(0)# inputs.size(0)   # batch total loss
            test_acc += calc_acc(outputs, labels).item()

            batch_val_counter += labels.size(0) #inputs.size(0)

        # # find average training loss and validation acc
        # avg_test_loss = test_loss/batch_val_counter
        # avg_test_acc = test_acc/batch_val_counter

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/batch_val_counter
    avg_test_acc = test_acc/batch_val_counter

    print("Model: " + model_type + " - Test accuracy : " + str(avg_test_acc) +
          " Test loss : " + str(avg_test_loss))
