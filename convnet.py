# -*- coding: utf-8 -*-
import os
#from twisted.protocols.amp import _SwitchBox
from skimage import *
import glob
import numpy as np
from numpy import *
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import *
from lasagne.updates import adagrad
from nolearn.lasagne import *
import skimage
from skimage.viewer import ImageViewer
import  theano
from theano import *
import theano.tensor as T
import time
import matplotlib.pyplot as plt
plt.ion()
from fit import fit, pred
import itertools
from load_data import data_loader
import sys
sys.stdout = open('log.txt', 'w')
from sklearn.cross_validation import KFold

def float32(x):
    return np.cast['float32'](x)
	
loader = data_loader()

NTEST = 300000
EPOCHS = 100

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]

X, X_hog, y= loader.load_data(pathes)




lin = layers.InputLayer((None, 3, 32, 32))
lhog = layers.InputLayer((None, X_hog.shape[1]))

h1 = layers.Conv2DLayer(incoming=lin, num_filters=100, filter_size=(3,3), name = 'afterinputconv')
addconv = layers.Conv2DLayer(incoming=h1, num_filters=150, filter_size=(2,2), name = 'conv')
h2 = layers.MaxPool2DLayer(incoming=addconv, pool_size=(2,2), name = 'pool')
h4 = layers.MaxPool2DLayer(incoming=h3, pool_size=(2,2), name = 'pool')
h5 = layers.Conv2DLayer(incoming=h4, num_filters=100, filter_size=(2,2), name = 'conv')
h6 = layers.MaxPool2DLayer(incoming=h5, pool_size=(2,2), name = 'pool')
h7 = layers.Conv2DLayer(incoming=h6, num_filters=100, filter_size=(2,2), name = 'conv')
h8 = layers.MaxPool2DLayer(incoming=h7, pool_size=(2,2), name = 'pool')
h9 = layers.ReshapeLayer(h4, ([0], -1))
p = layers.DenseLayer(h4, 100)
merge = layers.ConcatLayer([h9, lhog])
out = layers.DenseLayer(h10, 10, nonlinearity=nonlinearities.softmax)

_layers = [h1, h2, h3, h4, h5, h6, h7]

'''
shape = lin.get_output_shape()
Xi =  np.asarray([(t.ravel()) for t in X])
for l in _layers:
    if (l.name != 'merge' and l.name != 'pool'):
        inp = layers.InputLayer(shape)

        if ('conv' in l.name):
            tlayer = layers.Conv2DLayer(incoming=inp, num_filters=l.num_filters, filter_size=l.filter_size, W=l.W, b=l.b)
        else:
            tlayer = layers.DenseLayer(incoming=inp, num_units=l.num_units, W=l.W, b=l.b)

        out = layers.DenseLayer(incoming=tlayer, num_units=Xi.shape[1])
        if ('afterinput' in l.name):
            fit(lin=inp, lhog = lhog, output_layer=out, X=X, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=1,
            l_rate_start = 0.01, l_rate_stop = 0.00001, batch_size = 100, l2_strength = 0, Flip=False)
        else:
            fit(lin=inp, lhog = lhog, output_layer=out, X=Xi, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=1,
            l_rate_start = 0.01, l_rate_stop = 0.00001, batch_size = 100, l2_strength = 0, Flip=False)

    shape = l.output_shape
    kf = KFold(NTRAIN, 100)
    Xi = np.empty(tuple(np.append([1], shape[1:])), 'float32')
    for indices in iter(kf):
        lin.input_var = theano.shared(X[indices[1]])
        lhog.input_var = theano.shared(X_hog[indices[1]])
        out = theano.function([], layers.get_output(l, deterministic=True), on_unused_input='ignore')
        t=out()
        Xi = np.concatenate((Xi, out()))

    Xi = Xi[1:]
'''



fit(lin=lin, lhog=lhog, output_layer=out, X=X, X_hog=X_hog, y=y, eval_size=0.1, num_epochs=EPOCHS,


loader.print_prediction(count=NTEST, numiters=1000, pred=pred, lin=lin, lhog=lhog, output_layer=out)