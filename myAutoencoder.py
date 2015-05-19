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

def float32(x):
    return np.cast['float32'](x)


class AutoEncoder(NeuralNet):
    def __init__(self, input_layer, layer):
         self.input_layer = input_layer
         self.layer = layer

         super(AutoEncoder, self,
             layers=[
                 self.input_layer,
                 self.output_layer,
                ('output', layers.DenseLayer),
                ],

             output_nonlinearity=None,
             output_shape = self.input_layer.shape,


             update=adagrad,
             update_learning_rate=theano.shared(float32(0.01)),

             on_epoch_finished=[
                 AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
               # AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 ],
             objective=myObjective.Objective,
             eval_size=float32(0.1),
             regression=True,
             max_epochs=10,
             verbose=1,
        )
    def train(self, X):
         self.fit(X,X)