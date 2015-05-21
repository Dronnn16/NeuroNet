import skimage
import numpy as np
from sklearn.cross_validation import KFold
import itertools
from numpy import *

def float32(x):
    return np.cast['float32'](x)


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

def load_data (pathes):
    TEST = []
    TEST_hog = []
    for i in xrange(len(pathes)):
        #print ('loading %s\n' % pathes[i])
        image = skimage.io.imread(pathes[i])
        hog = skimage.feature.hog(skimage.color.rgb2grey(image), orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), normalise=True)
        normimage = float32(image/float32(255))
        R =  normimage[:, :, 0]
        G =  normimage[:, :, 1]
        B =  normimage[:, :, 2]
        TEST.append(np.asarray([R, G, B]))
        TEST_hog.append(float32(hog))

    TEST = np.asarray(TEST)
    TEST_hog = np.asarray(TEST_hog)

    return TEST, TEST_hog

def print_prediction (count, numiters, pred, lin, lhog, h5):
    kf = KFold(count, numiters)
    f = open('ans.txt', 'w')
    f.write('id,label\n')
    for indices in iter(kf):
        pathes = ["test/%s.png" %  (i+1) for i in indices[1]]
        TEST, TEST_hog = load_data(pathes)
        ANSES = pred (TEST, TEST_hog, lin, lhog, h5)
        for ans, i in zip (ANSES, itertools.count(1)):
            s = tostr(ans.argmax())
            f.write('%d,%s\n' % (i, s))
