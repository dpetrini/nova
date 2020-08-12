#
#   Example for nova simple pytorch-based library
#

import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet50 import MyResnet50
from dataset_example import MyDataset   # load our dataset class

from callbacks.cb_handler import CallbackHandler
from callbacks.cb_base import BaseCB
from callbacks.cb_lr_patch_clf import LR_SchedCB

from learner import train_and_validate, run_test
from util.util import load_checkpoint


prefix = 'data_cats_dogs'

device = 'gpu'      # 'cpu' #    # valid if parallel false
gpu_number = 0

num_epochs = 50
mini_batch = 64

LR = 3e-3

PRE_TRAINED = False

NUM_WORKERS = 2


def main():

    if PRE_TRAINED:
        model = MyResnet50(pretrained=True)
    else:
        model = MyResnet50()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)   # change for our 2 categories

    if not PRE_TRAINED:
        # init para not pre-trained
        print("Init weights with kaiming")
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    device = 'gpu'

    if (device == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(gpu_number))
    else:
        device = torch.device("cpu")

    model = model.to(device)

    train_image_paths = prefix+'/train'
    val_image_paths = prefix+'/val'
    test_image_paths = prefix+'/test'

    # classe dataset que carregar arquivos e faz transformacoes
    dataset_train = MyDataset(train_image_paths, train=True)
    dataset_val = MyDataset(val_image_paths, train=False)
    dataset_test = MyDataset(test_image_paths, train=False)

    print('Size train:', len(dataset_train), ' Size val: ', len(dataset_val))

    n_samples = len(dataset_train) + len(dataset_val)

    train_dataloader = DataLoader(dataset_train, batch_size=mini_batch,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=mini_batch,
                                shuffle=False, num_workers=NUM_WORKERS+1)
    test_dataloader = DataLoader(dataset_test, batch_size=mini_batch,
                                 shuffle=False, num_workers=1)

    loss_func = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1

    # Load Callbacks for this session
    cb = CallbackHandler([BaseCB('example')])  # AUC_CB() , LR_SchedCB()

    optim_args = {} 

    # train the model
    train_and_validate(model, train_dataloader, val_dataloader, loss_func,
                       optimizer, optim_args, mini_batch, num_epochs,
                       start_epoch, device, cb)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H_%M')

    print("\nRunning model in test set...")
    model = cb.last_model
    best_model = cb.best_model
    run_test(model, loss_func, test_dataloader, device, st, mini_batch, "normal")
    run_test(best_model, loss_func, test_dataloader, device, st, mini_batch, "best")


if __name__ == '__main__':
    main()
