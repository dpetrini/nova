import torch
import datetime
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, roc_curve


# [UTIL] Checkpoints
def save_checkpoint(optimizer, model, acc, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        # 'best_model': best_model.state_dict(),
        'acc': acc,
        'epoch': epoch
    }
    # print('Saving checkpoint: ', filename)
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    # best_model.load_state_dict(checkpoint_dict['best_model'])
    acc = checkpoint_dict['acc']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch, acc  # , best_model


def show_auc(label_auc, y_hat_auc, title, pr=False, show_plt=True):
    """Plots AUC and Precision-Recall Curves
    Input: labels and inference outputs as np arrays
    Output: plots on screen and saved in plot_test_auc folder """

    # get time for file naming
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')

    if os.path.isdir('plot_test_auc/') is False:
        os.makedirs('plot_test_auc/', exist_ok=False)

    # #### Compute ROC curve and ROC area for each class MALIGN
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thr = roc_curve(label_auc.ravel(),
                                                y_hat_auc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    #plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='ROC curve (area = {0:0.4f})'
             ''.format(np.round(roc_auc["micro"], 4)),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title+' - ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('plot_test_auc/'+str(st)+'_Category_1_ROC.png')
    if show_plt:
        plt.show()

    if not pr:
        return

    # Precision/recal curve - Malign
    precision, recall, _ = precision_recall_curve(label_auc.ravel(),
                                                  y_hat_auc.ravel())
    avg_precision = average_precision_score(label_auc.ravel(),
                                            y_hat_auc.ravel())

    # Plot PR curves
    #plt.figure()
    plt.plot(precision, recall,
             label='Precision-Recall curve (area = {0:0.2f})'
             ''.format(avg_precision),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('plot_test_auc/'+str(st)+'_Maligno_PR.png')
    if show_plt:
        plt.show()
