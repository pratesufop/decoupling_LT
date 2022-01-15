from tensorflow.keras.datasets import mnist
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import pareto
import numpy as np
import os

def load_mnist_lt(outdir, b = 6):
    
    """ loading the mnist dataset and using the pareto distribution to transform it in a long-tail dataset
        Inputs:
            outdir: output dir to save the dataset distributions
            b: parameter that controls the long tail distribution shape
        Returns:
            x_train, y_train, x_test, y_test (usual mnist dataset)
            x_train_lt, y_train_lt (long tail version - pareto sampled version of the mnist)
    """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    x_test = (x_test.astype(np.float32) - 127.5)/127.5

    open_test = []

    num_cls = 10 # 

    # generating a distribution of samples/class with Pareto dist.
    x = np.linspace(pareto.ppf(0.01, b),
            pareto.ppf(0.99, b), num_cls)
    rv = pareto(b)
    probs = rv.pdf(x)/max(rv.pdf(x))

    counts = dict(Counter(y_train))

    nums = list(counts.keys())
    values = list(counts.values())

    idx = np.argsort(values)[::-1]
    values = np.array(values)[idx]
    nums = np.array(nums)[idx]

    x_train_lt, y_train_lt = [], []
    final_idx = []

    np.random.shuffle(probs)

    for v,n,p in zip(values, nums, probs):
        final_idx.extend(np.random.choice(np.where(y_train == n)[0], int(v*p)))


    np.random.shuffle(final_idx)
    x_train_lt = x_train[final_idx]
    y_train_lt = y_train[final_idx]

    plt.figure(figsize=(20,10))

    plt.subplot(121)
    w = Counter(y_train_lt)
    plt.bar(w.keys(), w.values())
    plt.title('Long Tail')

    plt.subplot(122)
    w = Counter(y_train)
    plt.bar(w.keys(), w.values())
    plt.title('Normal')

    plt.savefig(os.path.join(outdir,'MNIST-LT.png'))

    return (x_train, y_train, x_test, y_test, x_train_lt, y_train_lt)