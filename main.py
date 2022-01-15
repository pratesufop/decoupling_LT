import os
import numpy as np
import tensorflow
from scipy.spatial import distance
from utils import plot_cm, balanced_generator, get_umap, plot_weigths_norm
from model import get_model, get_cls_retrain_model, get_lws_model
from read_data import load_mnist_lt
import argparse
# evitar erro de alocação de memória
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

parser = argparse.ArgumentParser()
   
parser.add_argument('-m','--mode', required=True,  choices=['balanced', 'cls_retrain', 'nmc', 't-normalized', 'lws'], help="one of the modes from the paper")
parser.add_argument('-s','--strategy',  choices=['sqrt', 'cls_balanced','instance_balanced'], default = 'instance_balanced', help="sqrt or cls_balanced, only valid for the balanced setup")


args = parser.parse_args()


bs = 128
num_epochs = 30

mode = args.mode

np.random.seed(2389)

# loading the data
x_train, y_train, x_test, y_test, x_train_lt, y_train_lt =  load_mnist_lt(outdir = '.', b = 6)


model = get_model(num_outputs = 10)

model.summary()

model.compile(loss= tensorflow.keras.losses.SparseCategoricalCrossentropy(),
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights=1,
            metrics='acc')

if mode == 'balanced':
    
    gen_ = balanced_generator(x_train_lt, y_train_lt, bs, args.strategy)

    
    training_output = model.fit(gen_,
                                batch_size = bs, 
                                epochs =  num_epochs, steps_per_epoch =  1000, 
                                shuffle= True)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)

elif mode == 'cls_retrain':
    
    gen_ = balanced_generator(x_train_lt, y_train_lt, bs, 'instance_balanced') # following the paper

    training_output = model.fit(
                                gen_,
                                batch_size = bs, 
                                epochs =  num_epochs, steps_per_epoch =  1000, 
                                shuffle= True)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
    
    retrain_model = get_cls_retrain_model(model)

    retrain_model.compile(loss= tensorflow.keras.losses.SparseCategoricalCrossentropy(),
                optimizer= tensorflow.keras.optimizers.Adam(),
                loss_weights=1,
                metrics='acc')
    
    
    cls_gen = balanced_generator(x_train_lt, y_train_lt, bs, 'cls_balanced')

    training_output = retrain_model.fit(cls_gen, 
                                batch_size = bs , 
                                epochs =  num_epochs, steps_per_epoch =  1000 , 
                                shuffle= True)

    y_pred = retrain_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
    
elif mode == 'nmc':
    
    gen_ = balanced_generator(x_train_lt, y_train_lt, bs, 'instance_balanced') # following the paper
    
    training_output = model.fit(#x_train_lt, y_train_lt,
                                gen_,
                                batch_size = bs, 
                                epochs =  num_epochs, steps_per_epoch =  1000)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    plot_cm(y_test, y_pred)
    
    feat_model = tensorflow.keras.Model(inputs= model.inputs, outputs= model.get_layer('gap').output)
    feat_lt = feat_model.predict(x_train_lt)

    centers = []
    for u in np.unique(y_train_lt):
        centers.append(np.mean(feat_lt[y_train_lt == u,:], axis=0))

    centers = np.array(centers)

    # cosine distance
    test_feat = feat_model.predict(x_test)

    y_pred = []
    for t in test_feat:
        dists = []
        for c in centers:
            dists.append(distance.cosine(t, c))
        y_pred.append(np.argsort(dists)[0]) # closest neighbor

    plot_cm(y_test, np.array(y_pred))
    get_umap(feat_lt, y_train_lt)

elif mode == 't-normalized':
    
    gen_ = balanced_generator(x_train_lt, y_train_lt, bs, 'instance_balanced') # following the paper
    
    training_output = model.fit(#x_train_lt, y_train_lt,
                                gen_,
                                batch_size = bs, 
                                epochs =  num_epochs, steps_per_epoch =  1000)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
    
    w = model.get_layer('cls').weights[0]

    # checking the hypothesis : norm of the classifiers are correlated with the cardinality of the classes (more samples, bigger norms)
    plot_weigths_norm(y_train_lt, w)

    w = np.array(w)
    p = 0.7
    w_ = np.array([nw/np.power(np.sum(np.power(np.abs(nw),p)),1/p) for nw in w.T]).T
    model.get_layer('cls').set_weights([w_])

    y_pred =  model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
    
elif mode == 'lws':
    
    gen_ = balanced_generator(x_train_lt, y_train_lt, bs, 'instance_balanced') # following the paper
    
    training_output = model.fit(#x_train_lt, y_train_lt,
                                gen_,
                                batch_size = bs, 
                                epochs =  num_epochs, steps_per_epoch =  1000,
                                shuffle= True)
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
    
    lws_model = get_lws_model(model)
    
    lws_model.compile(loss= tensorflow.keras.losses.SparseCategoricalCrossentropy(),
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights=1,
            metrics='acc')
    
    cls_gen = balanced_generator(x_train_lt, y_train_lt, bs, 'cls_balanced')

    training_output = lws_model.fit(cls_gen, 
                                batch_size = bs , 
                                epochs =  num_epochs, steps_per_epoch =  1000,
                                shuffle= True)

    y_pred = lws_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    plot_cm(y_test, y_pred)
