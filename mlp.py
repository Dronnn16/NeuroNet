# -*- coding: utf-8 -*-
import os
#from twisted.protocols.amp import _SwitchBox
from skimage import *
import glob
import numpy as np
from numpy import *
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
from load_data import load_data, print_prediction
import sys
sys.stdout = open('log.txt', 'w')
from sklearn.cross_validation import KFold

def float32(x):
    return np.cast['float32'](x)





NTRAIN = 50000
NTEST = 200
EPOCHS = 100

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]
X, X_hog, y = load_data(pathes)








lin = layers.InputLayer((None, 3, 32, 32))
lhog = layers.InputLayer((None, 324))

h1 = layers.DenseLayer(lin, 50, name = 'afterinput', nonlinearity=nonlinearities.sigmoid)
#merge = layers.ConcatLayer([h1, lhog], name = 'merge')
h2 = layers.DenseLayer(h1, 40)#, nonlinearity=nonlinearities.sigmoid)
h3 = layers.DenseLayer(h2, 20)#, nonlinearity=nonlinearities.sigmoid)
h4 = layers.DenseLayer(h3, 20)#, nonlinearity=nonlinearities.sigmoid)
h5 = layers.DenseLayer(h4, 10, nonlinearity=nonlinearities.softmax)

_layers = [h1, h2, h3, h4]


shape = lin.get_output_shape()
Xi =  np.asarray([(t.ravel()) for t in X])
for l in _layers:
    if (l.name != 'merge'):
        inp = layers.InputLayer(shape)
        tlayer = layers.DenseLayer(incoming=inp, num_units=l.num_units, W=l.W, b=l.b, nonlinearity=l.nonlinearity)
        out = layers.DenseLayer(incoming=tlayer, num_units=Xi.shape[1])#, nonlinearity=nonlinearities.sigmoid)
        if (l.name == 'afterinput'):
            fit(lin=inp, lhog = lhog, output_layer=out, X=X, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=100,
            l_rate_start = 0.01, l_rate_stop = 0.00001, batch_size = 100, l2_strength = 0, Flip=False)
        else:
            fit(lin=inp, lhog = lhog, output_layer = out, X=Xi, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=50,
            l_rate_start = 0.001, l_rate_stop = 0.0001, batch_size = 100, l2_strength = 0, Flip=False)



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





fit(lin=lin, lhog=lhog, output_layer=h5, X=X, X_hog=X_hog, y=y, eval_size=0.1, num_epochs=EPOCHS,
    l_rate_start = 0.000001, l_rate_stop = 0.0000001, batch_size = 100, l2_strength = 0, Flip=False)


print_prediction (count=NTEST, numiters=1000, pred=pred, lin=lin, lhog=lhog, output_layer=h5)


