import torch
import datetime
import time
import os
import itertools
import numpy as np
import pandas as pd
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


def save_results(label_auc, y_hat_auc, df, title, save_path):

    if not save_path.endswith('/'):
        save_path += '/'

    if os.path.isdir(save_path) is False:
        os.makedirs(save_path, exist_ok=False)

    # get time for file naming
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')

    #gravar resultados numpy
    if not os.path.isfile(save_path+'labels.npy'):
        np.save(save_path+'labels', label_auc)   # labels: save if not already
    np.save(save_path+'predictions_'+str(st), y_hat_auc)
    df.to_csv(save_path+'results_'+str(st)+'.csv', sep='\t')


def show_auc(label_auc, y_hat_auc, title, save_path, pr=False, show_plt=True):
    """Plots AUC and Precision-Recall Curves
    Input: labels and inference outputs as np arrays
    Output: plots on screen and saved in plot_test_auc folder """

    # get time for file naming
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')

    if not save_path.endswith('/'):
        save_path += '/'

    plots_dir = save_path

    if os.path.isdir(plots_dir) is False:
        os.makedirs(plots_dir, exist_ok=False)

    # #### Compute ROC curve and ROC area for each class MALIGN
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thr = roc_curve(label_auc.ravel(),
                                                y_hat_auc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
             label='ROC curve (area = {0:0.4f})'
             ''.format(np.round(roc_auc["micro"], 4)),
             color='deeppink', linestyle=':', linewidth=2)

    auc_value = f'{np.round(roc_auc["micro"], 4):1.4f}'[-4:]
    auc_file = plots_dir+str(st)+'_AUC_ROC_test_0' + auc_value + '.png'

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('[ROC Curve] '+ title)
    plt.legend(loc="lower right")
    plt.savefig(auc_file)
    if show_plt:
        plt.show()

    if not pr:
        return auc_file

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
    plt.savefig(plots_dir+str(st)+'_Maligno_PR.png')
    if show_plt:
        plt.show()


# Hanley and McNeil AUC std dev
# m: positive samples, n: negative samples
# auc: calculated value
# returns : standard deviation
def calc_auc_desv(m, n, auc):

    if auc == 0:
        return 0

    Pxxy = auc / (2-auc)
    Pxyy = 2*auc**2 / ((1+auc))

    sigma_2 = (auc*(1-auc) + (m-1)*(Pxxy- auc**2) + (n-1)*(Pxyy - auc**2)) / (m*n)
    sigma = np.sqrt(sigma_2)

    return sigma


def plot_confusion_matrix2(cm, classes, normalize=False, title='Confusion matrix',
                           cmap=plt.cm.Blues, dir_to_save='plot_cm/', show_plt=True, 
                           lang='English'):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    if os.path.isdir(dir_to_save) is False:
        os.makedirs(dir_to_save, exist_ok=False)

    # get time for file naming
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%Hh%Mm')


    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        #plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color= "black")

    plt.tight_layout()
    plt.ylabel('True label') if lang=='English' else plt.ylabel('Classe real')
    if lang=='English':
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    else:
        plt.xlabel('Classe predita\nAcurácia={:0.4f}; Erro={:0.4f}'.format(accuracy, misclass))
    plt.savefig(dir_to_save+str(st)+'_'+title+'_cm.png', bbox_inches='tight')
    if show_plt:
        plt.show()
