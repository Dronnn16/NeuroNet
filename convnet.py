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
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import *
import skimage
from skimage.viewer import ImageViewer
import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters
import theano
from theano import *
import theano.tensor as T
import time
import matplotlib.pyplot as plt
plt.ion()
#from utils import generate_data, get_context
from AV import AdjustVariable
from theano import config
from Flip import FlipBatchIterator
from lasagne.updates import adagrad
import myObjective
theano.config.exception_verbosity='high'


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


def float32(x):
    return np.cast['float32'](x)



#data = generate_data(200)

NTRAIN = 50000
NTEST = 300000
EPOCHS = 100

pathes = ["train/%s.png" %  (i) for i in range(1, NTRAIN+1)]
X = []
for i in xrange(len(pathes)):
    image = skimage.io.imread(pathes[i])
   # hog = np.squeeze(np.asarray(skimage.feature.hog(skimage.color.rgb2grey(image)).ravel())).reshape(-1)
    X.append(image)
 #   X.append(image.ravel())
X = np.asarray(X)
X = float32(X)/float32(255)
X = X.reshape(-1, 3, 32, 32)

_y = read_csv('trainLabels.csv', ',').label.apply(tonum).values
y = np.zeros((len(X), 10))
for i in xrange(len(X)):
    y[i, _y[i]-1] = float32(1)

y = float32(y)









net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 32, 32),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # optimization method:
    update=adagrad,
    update_learning_rate=theano.shared(float32(0.03)),
  #  update_momentum=theano.shared(float32(0.9)),
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
      #  AdjustVariable('update_momentum', start=0.9, stop=0.999),
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
