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
import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters
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


#data = generate_data(200)

NTRAIN = 50000
NTEST = 50
EPOCHS = 10

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]
X = []
X_hog = []
for i in xrange(len(pathes)):
    image = skimage.io.imread(pathes[i])
   # hog = skimage.feature.hog(skimage.color.rgb2grey(image))
    X.append(float32(image/float32(255)))
   # X_hog.append(float32(hog))


X = np.asarray(X).reshape(-1, 3, 32, 32)
#X_hog = np.asarray(X_hog)

_y = read_csv('trainLabels.csv', ',').label.apply(tonum).values
y = np.zeros((len(X), 10))
for i in xrange(len(X)):
    y[i, _y[i]-1] = float32(1)

y = float32(y)

X2 = np.asarray(X).reshape(-1, 3*32*32)
















































lin = layers.InputLayer((None, 3, 32, 32))
l = layers.DenseLayer(incoming=lin, num_units=40)

ennet = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 32, 32),
    hidden_num_units = 40, hidden_W = l.W, hidden_b = l.b,
    output_nonlinearity=None,
    output_num_units=3*32*32,
    update=adagrad,
    update_learning_rate=theano.shared(float32(0.01)),
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0000001),
        ],
    objective=myObjective.Objective,
    eval_size=float32(0.1),
    regression=True,
    max_epochs=2000,
    verbose=1,
)

ennet.fit(X,X.reshape(-1,3*32*32))


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        #('hog', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
       # ('merge', layers.ConcatLayer),
        ('hidden3', layers.DenseLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 3, 32, 32),  # 96x96 input pixels per batch
  #  hog_shape=(None, 324),
    hidden1_num_units=40, #hidden1_W = rbm1.W.var.get_value(),
    hidden2_num_units=50,  # number of units in hidden layer
    #merge_incomings=['hog', 'hidden2'],
    hidden3_num_units=20,  # number of units in hidden layer
    hidden4_num_units=10,  # number of units in hidden layer
    hidden5_num_units=10,  # number of units in hidden layer
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=10,  # 30 target values

    # optimization method:
    update=adagrad,
    update_learning_rate=theano.shared(float32(0.01)),
    #update_momentum=theano.shared(float32(0.9)),
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
       # AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    objective=myObjective.Objective,
    eval_size=float32(0.1),
    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=EPOCHS,  # we want to train this many epochs
    verbose=1,
)

net1.fit(X, y)



f = open('ans.txt', 'w')

pathes = ["test/%s.png" %  (i) for i in range(1, NTEST+1)]
TEST = []
#X_hog = []
f.write('id,label\n')
for i in xrange(len(pathes)):
    image = float32(skimage.io.imread(pathes[i])/float32(255))
    #hog = np.squeeze(np.asarray(skimage.feature.hog(skimage.color.rgb2grey(image)).ravel())).reshape(-1)
    #TEST.append(float32(image.ravel())/float32(255))
  #  X_hog.append(float32(hog))
    image = np.asarray(image.reshape(3, 32, 32))
    s = tostr(net1.predict(np.asarray([image])).argmax())
    f.write('%d,%s\n' % (i+1, s))
    print ('%d,%s\n' % (i+1, s))
