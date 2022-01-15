import numpy as np
from collections import Counter
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import umap
import umap.plot


def plot_weigths_norm(y, w,  outdir = '.'):
    
    plt.figure(figsize = (10,7))
    sorted_cls = [k for k,v in dict(Counter(y)).items()]
    plt.plot(np.linalg.norm(w, axis= 0)[sorted_cls])
    
    plt.savefig(os.path.join(outdir, 'w_norm.png'))
    
def get_umap(feats, y,  outdir = '.'):

    feats = np.squeeze(feats)
    embedding = umap.UMAP(n_neighbors=5).fit_transform(feats)

    classes = np.unique(y)
    _, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*embedding.T, s=20.0, c=y, cmap='jet_r', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP Embedding', fontsize=14)
    cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5)
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'umap.png'))

def plot_cm(y_test, y_pred, outdir = '.'):
    
    """ Plots the confusion matrix and returns the normalized accuracy
        Inputs:
            y_test: groundtruth labels (n,1)
            y_pred: predicted labels after argmax (n,1)
            outdir: folder to save the Confusion Matrix
        Returns:
            norm_acc: normalized accuracy
    """
    
    cm = confusion_matrix(y_test, y_pred)
    cm = cm/np.sum(cm, axis=1)
    cm = np.round(cm, 2)

    df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
              columns = [i for i in "0123456789"])

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

    plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))

    norm_acc = 100*accuracy_score(y_test, y_pred, normalize= True)	
    print('Norm. Acc. %.2f' % (norm_acc))
    
    return norm_acc

def get_probs(y, strategy):
    """ returns the probability accordingly with the sampling strategy
        Inputs:
           
            y_train_lt: labels (n,1)
            sqrt_on: squared root sampling
            
        Returns:
            probs (1,c) with the probabilities for each class
    """
    
    num_cls = len(np.unique(y))
    if strategy == 'sqrt':    
        n= []
        for i in np.arange(num_cls):
            pos = np.where(y == i)[0]
            n.append(len(pos))

        probs = np.sqrt(n)/np.sum(np.sqrt(n))
    elif strategy == 'cls_balanced':
        probs = np.ones(num_cls)/num_cls
    else: # instance-balanced
        n= []
        for i in np.arange(num_cls):
            pos = np.where(y == i)[0]
            n.append(len(pos))

        probs = n/np.sum(n)
    
    return probs

def balanced_generator(x, y, batch_size, strategy = 'cls_balanced'):
    
    """ data generator based on the probabilities of each class
        Inputs:
            x: input data (n,h,w,c)
            y: labels (n,1)
            batch_size: the number of samples in a batch
            sqrt_on: squared root sampling
        Returns:
            batches (x,y) at each iteration
    """
    
    cls_ = np.unique(y)
    
    probs = get_probs(y, strategy)
    
    idx_cls = {}
    for i in cls_:
        idx_cls[i] = np.where(y == i)[0]

    while True:
        
        samples = np.random.choice(len(cls_), batch_size, p=probs) 
        
        x_batch, y_batch  = [], []

        for i in cls_:
            
            idx = np.random.choice(idx_cls[i], np.sum(samples == i))

            x_batch.extend(x[idx])
            y_batch.extend(y[idx])

            if len(x_batch) == batch_size:

                x_batch, y_batch = np.array(x_batch), np.array(y_batch)

                yield(x_batch,y_batch)

                x_batch, y_batch = [], []