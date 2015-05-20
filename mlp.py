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
from load_data import load_data
import sys
sys.stdout = open('log.txt', 'w')

def float32(x):
    return np.cast['float32'](x)

def tonum(s):
    t = {
        'airplane' : 1,
        'automobile' : 2,
        'bird' : 3,
        'cat' : 4,
        'deer' : 5,
        'dog' : 6,
        'frog' : 7,
        'horse' : 8,
        'ship' : 9,
        'truck' : 10
    }
    return t[s]

def tostr(s):
    t = {
        1 : 'airplane',
        2 : 'automobile',
        3 : 'bird',
        4 : 'cat',
        5 : 'deer',
        6 : 'dog',
        7 : 'frog',
        8 : 'horse',
        9 : 'ship',
        10 : 'truck'
    }
    return t[s+1]



NTRAIN = 1000
NTEST = 300000
EPOCHS = 10

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]
X, H_hog = load_data(pathes)

_y = read_csv('trainLabels.csv', ',').label.apply(tonum).values
y = np.zeros((len(X), 10))
for i in xrange(len(X)):
    y[i, _y[i]-1] = float32(1)

y = float32(y)






lin = layers.InputLayer((None, 3, 32, 32))
lhog = layers.InputLayer((None, 324))

h1 = layers.DenseLayer(lin, 50, name = 'afterinput')
#merge = layers.ConcatLayer([h1, lhog])
h2 = layers.DenseLayer(h1, 40)
h3 = layers.DenseLayer(h2, 20)
h4 = layers.DenseLayer(h3, 20)
h5 = layers.DenseLayer(h4, 10, nonlinearity=nonlinearities.softmax)

_layers = [h1, h2, h3, h4]


shape = lin.get_output_shape()
Xi =  np.asarray([(t.ravel()) for t in X])
for l in _layers:
    if (l.name != 'merge'):
        inp = layers.InputLayer(shape)
        tlayer = layers.DenseLayer(incoming=inp, num_units=l.num_units, W=l.W, b=l.b)
        out = layers.DenseLayer(incoming=tlayer, num_units=Xi.shape[1])
        if (l.name == 'afterinput'):
            fit(lin=inp, lhog = lhog, output_layer=out, X1=X, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=100,
            l_rate_start = 0.01, l_rate_stop = 0.00001)
        else:
            fit(lin=inp, lhog = lhog, output_layer = out, X1=Xi, X_hog=X_hog, y=Xi, eval_size=0.1, num_epochs=100,
            l_rate_start = 0.01, l_rate_stop = 0.00001)

    lin.input_var = X
    lhog.input_var = X_hog
    out = theano.function([], l.get_output())
    Xi = out()
    shape = l.get_output_shape()




fit(lin, lhog, h5, X, X_hog, y, eval_size=0.1, num_epochs=EPOCHS, l_rate_start = 0.01, l_rate_stop = 0.00001)





f = open('ans.txt', 'w')
f.write('id,label\n')

pathes = ["test/%s.png" %  (i) for i in range(1, 100001)]
TEST, TEST_hog = load_data(pathes)
ANSES = pred (TEST, TEST_hog, lin, lhog, h5)
for ans, i in zip (ANSES, itertools.count(1)):
    s = tostr(ans.argmax())
    f.write('%d,%s\n' % (i, s))
   # print ('%d,%s\n' % (i, s))


pathes = ["test/%s.png" %  (i) for i in range(100001, 200001)]
TEST, TEST_hog = load_data(pathes)
ANSES = pred (TEST, TEST_hog, lin, lhog, h5)
for ans, i in zip (ANSES, itertools.count(100001)):
    s = tostr(ans.argmax())
    f.write('%d,%s\n' % (i, s))
   # print ('%d,%s\n' % (i, s))

pathes = ["test/%s.png" %  (i) for i in range(200001, 300001)]
TEST, TEST_hog = load_data(pathes)
ANSES = pred (TEST, TEST_hog, lin, lhog, h5)
for ans, i in zip (ANSES, itertools.count(200001)):
    s = tostr(ans.argmax())
    f.write('%d,%s\n' % (i, s))
  #  print ('%d,%s\n' % (i, s))