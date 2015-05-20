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
#from utils import generate_data, get_context
from AV import AdjustVariable
from Flip import FlipBatchIterator
from lasagne.layers import  *
from lasagne.regularization import l2
import myObjective
import myAutoencoder
from myAutoencoder import AutoEncoder
from fit import fit


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
NTEST = 50
EPOCHS = 2

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]
X = []
X_hog = []
for i in xrange(len(pathes)):
    image = skimage.io.imread(pathes[i])
    hog = skimage.feature.hog(skimage.color.rgb2grey(image))
    X.append(float32(image/float32(255)))
    X_hog.append(float32(hog))


X = np.asarray(X).reshape((-1, 3, 32, 32))
X_hog = np.asarray(X_hog)

_y = read_csv('trainLabels.csv', ',').label.apply(tonum).values
y = np.zeros((len(X), 10))
for i in xrange(len(X)):
    y[i, _y[i]-1] = float32(1)

y = float32(y)






lin = layers.InputLayer((None, 3, 32, 32))
lhog = layers.InputLayer((None, 324))

h1 = layers.DenseLayer(lin, 50)
merge = layers.ConcatLayer([h1, lhog])
h2 = layers.DenseLayer(merge, 40)
h3 = layers.DenseLayer(h2, 20)
h4 = layers.DenseLayer(h3, 20)
h5 = layers.DenseLayer(h4, 10, nonlinearity=nonlinearities.softmax)

_layers = [h1, h2, h3, h4, h5]



'''
Xi = X
shape = lin.get_output_shape()
for l in _layers:
    ennet = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer)
            ],
        input_shape = shape,
        hidden_num_units = l.num_units, hidden_W = l.W, hidden_b = l.b,
        output_nonlinearity=None,
        output_num_units=np.product(shape[1:]),
        update=adagrad,
        update_learning_rate=theano.shared(float32(0.01)),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            ],
        objective=myObjective.Objective,
        eval_size=float32(0.1),
        regression=True,
        max_epochs=100,
        verbose=1,
    )

    ennet.fit(Xi, Xi.reshape((-1, np.product(shape[1:]))))

    x = T.vector()
    out = theano.function([x], l.get_output_for(x))
    Xi = np.asarray([out(t.ravel())[0] for t in Xi])
    shape = l.get_output_shape()

'''


fit(lin, lhog, h5, X, X_hog, y, eval_size=0.1, num_epochs=EPOCHS, l_rate_start = 0.1, l_rate_stop = 0.00001)





f = open('ans.txt', 'w')

pathes = ["test/%s.png" %  (i) for i in range(1, NTEST+1)]
TEST = []
TEST_hog = []

for i in xrange(len(pathes)):
    image = skimage.io.imread(pathes[i])
    hog = skimage.feature.hog(skimage.color.rgb2grey(image))
    TEST.append(float32(image/float32(255)))
    TEST_hog.append(float32(hog))

TEST = np.asarray(TEST).reshape((-1, 3, 32, 32))
TEST_hog = np.asarray(TEST_hog)

lin.input_var = TEST
lhog.input_var = TEST_hog
pred = theano.function([], h5.get_output(deterministic=True))
ANSES = pred()

f.write('id,label\n')
for ans, i in zip (ANSES, xrange (len(ANSES))):
    s = tostr(ans.argmax())
    f.write('%d,%s\n' % (i+1, s))
    print ('%d,%s\n' % (i+1, s))
