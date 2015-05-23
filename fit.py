from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from sklearn.cross_validation import KFold
import numpy as np
from numpy import *
import myObjective
import skimage
from multiprocessing import Pool
from load_data import hog_filter
BATCH_SIZE = 5
LEARNING_RATE = 0.001
MOMENTUM = 0.9

def float32(x):
    return np.cast['float32'](x)







def load_data(X, X_hog, y, eval_size=0.1):
    """Get data with labels, split into training, validation and test set."""
    kf = KFold(y.shape[0], round(1. / eval_size))
    train_indices, valid_indices = next(iter(kf))
    X_train, X_hog_train, y_train = X[train_indices], X_hog[train_indices], y[train_indices]
    X_valid, X_hog_valid, y_valid = X[valid_indices], X_hog[valid_indices], y[valid_indices]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        X_hog_train=lasagne.utils.floatX(X_hog_train),
        y_train=y_train,
        X_valid=lasagne.utils.floatX(X_valid),
        X_hog_valid=lasagne.utils.floatX(X_hog_valid),
        y_valid=y_valid,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0]
    )



def create_iter_functions(inp1, inp2, dataset, output_layer,
                          batch_size, l2_strength):

    X_batch = T.cast(theano.shared(lasagne.utils.floatX(dataset['X_train'][0:1])), 'float32')
    X_hog_batch = T.matrix('x1')
    y_batch = T.cast(theano.shared(dataset['y_train'][0:1]), 'float32')

    learning_rate = T.fscalar('rate')

    objective = myObjective.Objective(input_layer=output_layer,
        loss_function=lasagne.objectives.mse, l2_strength = l2_strength)



    inp1.input_var = X_batch
    inp2.input_var = X_hog_batch

    loss_train = objective.get_loss(target=y_batch)
    loss_eval = objective.get_loss(target=y_batch,
                                   deterministic=True)

    pred = output_layer.get_output(deterministic=True)

    acs = T.cast([T.eq(pred[i].argmax(), y_batch[i].argmax()) for i in xrange(batch_size)], 'float32')
    accuracy = T.mean(acs)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.adagrad(
        loss_train, all_params, learning_rate)

    iter_train = theano.function(
        [X_batch, X_hog_batch, y_batch, learning_rate], loss_train,
        updates=updates,
        allow_input_downcast=True,

        on_unused_input='ignore'
    )

    iter_valid = theano.function(
        [X_batch, X_hog_batch, y_batch], [loss_eval, accuracy],
        on_unused_input='ignore'
    )


    return dict(
        train=iter_train,
        valid=iter_valid
    )





def train(iter_funcs, dataset, ls, batch_size, Flip, p):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """

    num_batches_train = np.ceil(float32(dataset['num_examples_train']) / float32(batch_size))
    num_batches_valid = np.ceil(float32(dataset['num_examples_valid']) / float32(batch_size))

    for epoch in itertools.count(1):
        batch_train_losses = []
        for X_batch, X_hog_batch, y_batch, batch_index in BatchIterator(dataset=dataset, batch_size=batch_size, valid=False, Flip=Flip, p=p):
            batch_train_loss = iter_funcs['train'](X_batch, X_hog_batch, y_batch, ls[epoch-1])
            batch_train_losses.append(batch_train_loss)
            if batch_index>=num_batches_train:
                break;
        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for X_batch, X_hog_batch, y_batch, batch_index in BatchIterator(dataset=dataset, batch_size=batch_size, valid=True, Flip=False, p=p):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](X_batch, X_hog_batch, y_batch)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)
            if batch_index>=num_batches_valid:
                break;

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }



def Fliphog(image):
    bimage = np.empty((32,32,3))
    bimage[:,:,0] = image[0,:,:]
    bimage[:,:,1] = image[1,:,:]
    return hog_filter(np.round(bimage*255))



def BatchIterator(dataset, batch_size, valid, Flip, p):
    for batch_index in itertools.count(0):
        batch_slice = slice(batch_index * batch_size,
                            (batch_index + 1) * batch_size)
        if valid:
            X_batch, X_hog_batch, y_batch = dataset['X_valid'][batch_slice], dataset['X_hog_valid'][batch_slice], dataset['y_valid'][batch_slice]
        else:
            X_batch = dataset['X_train'][batch_slice]
            X_hog_batch=dataset['X_hog_train'][batch_slice]
            y_batch=dataset['y_train'][batch_slice]
            if Flip:
                bs = X_batch.shape[0]
                indices = np.random.choice(bs, bs / 2, replace=False)
                X_batch[indices] = X_batch[indices, :, :, ::-1]
                X_hog_batch[indices] = map(Fliphog, X_batch[indices])
        yield X_batch, X_hog_batch, y_batch, batch_index+1





def fit(lin, lhog, output_layer, X, X_hog, y, eval_size=0.1, num_epochs=100,
        l_rate_start = 0.1, l_rate_stop = 0.00001, batch_size=100, l2_strength=0, Flip=True, p=None):
    dataset = load_data(X, X_hog, y, eval_size)
    iter_funcs = create_iter_functions(lin, lhog, dataset, output_layer, batch_size, l2_strength)
    ls = np.linspace(l_rate_start, l_rate_stop, num_epochs)


    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset, ls, batch_size, Flip, p):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.5f} %".format(
                epoch['valid_accuracy'] * 100))

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass
    print('\n')
    return output_layer



def pred (TEST, TEST_hog, lin, lhog, output_layer):
    lin.input_var = theano.shared(TEST)
    lhog.input_var = theano.shared(TEST_hog)
  ##  print ('start')
    pred = theano.function([], lasagne.layers.get_output(output_layer, deterministic=True))
    return  pred()

