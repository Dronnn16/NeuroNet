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


BATCH_SIZE = 100
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
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        X_hog_train=theano.shared(lasagne.utils.floatX(X_hog_train)),
        y_train=T.cast(theano.shared(y_train),  'float32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        X_hog_valid=theano.shared(lasagne.utils.floatX(X_hog_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'float32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0]
    )










def create_iter_functions(inp1, inp2, dataset, output_layer,
                          batch_size=BATCH_SIZE):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index * batch_size,
                    (batch_index + 1) * batch_size)
    X_batch = dataset['X_train'][batch_slice]
    X_hog_batch = T.matrix('x1')
    y_batch = T.matrix('y')

    learning_rate = T.fscalar('rate')

    objective = myObjective.Objective(output_layer,
        loss_function=lasagne.objectives.mse)


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
        [batch_index, learning_rate], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            X_hog_batch: dataset['X_hog_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
        allow_input_downcast=True,
        on_unused_input='warn'
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            X_hog_batch: dataset['X_hog_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
        on_unused_input='warn'
    )


    return dict(
        train=iter_train,
        valid=iter_valid
    )


def train(iter_funcs, dataset, ls, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """

    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b, ls[epoch-1])
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


def fit(lin, lhog, output_layer, X1, X_hog, y, eval_size=0.1, num_epochs=100,
        l_rate_start = 0.1, l_rate_stop = 0.00001,):
    dataset = load_data(X1 ,X_hog, y, eval_size)
    iter_funcs = create_iter_functions(lin, lhog, dataset, output_layer)
    ls = np.linspace(l_rate_start, l_rate_stop, num_epochs)


    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset, ls):
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

    return output_layer



def pred (TEST, TEST_hog, lin, lhog, output_layer):
    lin.input_var = TEST
    lhog.input_var = TEST_hog
    print ('start')
    pred = theano.function([], output_layer.get_output(deterministic=True))
    return  pred()

