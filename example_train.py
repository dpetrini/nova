#
# Example for nova simple pytorch-based library
#
# Size train: 2000  Size val:  1000  Size Test: 1000
# Loading from Ram
# Using AMP (pytorch 1.6+)
#
# Benchmarks:
# September/2020
#   Num epochs: 50
#   Batch size: 64
#   Acc Train: 83.45  Acc Val: 81.10   Acc Test: 79.30
#   Time @RTX2060:


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

from learner import train_and_validate_amp, run_test

device = 'gpu'      # 'cpu' #    # valid if parallel false
gpu_number = 0

PREFIX = 'data_cats_dogs'
NUM_EPOCHS = 50
MINI_BATCH = 64
LR = 3e-3
PRE_TRAINED = True
NUM_WORKERS = 2


def main():
    """ Just main """

    torch.backends.cudnn.benchmark = True

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

    train_image_paths = PREFIX+'/train'
    val_image_paths = PREFIX+'/val'
    test_image_paths = PREFIX+'/test'

    # classe dataset que carregar arquivos e faz transformacoes
    dataset_train = MyDataset(train_image_paths, train=True)
    dataset_val = MyDataset(val_image_paths, train=False)
    dataset_test = MyDataset(test_image_paths, train=False)

    print('Size train:', len(dataset_train), ' Size val: ', len(dataset_val))

    n_samples = len(dataset_train) + len(dataset_val)

    train_dataloader = DataLoader(dataset_train, batch_size=MINI_BATCH,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=MINI_BATCH,
                                shuffle=False, num_workers=NUM_WORKERS+1)
    test_dataloader = DataLoader(dataset_test, batch_size=MINI_BATCH,
                                 shuffle=False, num_workers=1)

    loss_func = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1

    # Load Callbacks for this session
    cb = CallbackHandler([BaseCB('example')])  # AUC_CB() , LR_SchedCB()

    optim_args = {} 

    # train the model
    train_and_validate_amp(model, train_dataloader, val_dataloader, loss_func,
                           optimizer, optim_args, MINI_BATCH, NUM_EPOCHS,
                           start_epoch, device, cb)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H_%M')

    print("\nRunning model in test set...")
    model = cb.last_model
    best_model = cb.best_model
    run_test(model, loss_func, test_dataloader, device, st, MINI_BATCH, "normal")
    run_test(best_model, loss_func, test_dataloader, device, st, MINI_BATCH, "best")


if __name__ == '__main__':
    main()
