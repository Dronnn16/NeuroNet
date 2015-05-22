import skimage
import numpy as np
from sklearn.cross_validation import KFold
import itertools
from numpy import *
from pandas.io.parsers import read_csv
from multiprocessing import Pool, cpu_count

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




def take_image(path):
    image = skimage.io.imread(path)
    hog = skimage.feature.hog(skimage.color.rgb2grey(image))
    normimage = float32(image/float32(255))
    R =  normimage[:, :, 0]
    G =  normimage[:, :, 1]
    B =  normimage[:, :, 2]
    return np.asarray([R, G, B])





def load_data (pathes, ontest=False):
    X = []
    X_hog = []
    p = cpu_count()
    p = Pool(4)
    X = p.map(take_image, pathes)
    for i in xrange(len(pathes)):
        #print ('loading %s\n' % pathes[i])
        image = skimage.io.imread(pathes[i])
        hog = skimage.feature.hog(skimage.color.rgb2grey(image), orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), normalise=True)
        normimage = float32(image/float32(255))
        R =  normimage[:, :, 0]
        G =  normimage[:, :, 1]
        B =  normimage[:, :, 2]
        X.append(np.asarray([R, G, B]))
        X_hog.append(float32(hog))

    X = np.asarray(X)
    X_hog = np.asarray(X_hog)

    y = np.zeros((len(X), 10))
    if not ontest:
        _y = read_csv('trainLabels.csv', ',').label.apply(tonum).values
        for i in xrange(len(X)):
            y[i, _y[i]-1] = float32(1)

        y = float32(y)

    return X, X_hog, y

def print_prediction (count, numiters, pred, lin, lhog, output_layer):
    kf = KFold(count, numiters)
    f = open('ans.txt', 'w')
    f.write('id,label\n')
    for indices in iter(kf):
        pathes = ["test/%s.png" %  (i+1) for i in indices[1]]
        TEST, TEST_hog, _ = load_data(pathes, True)
        ANSES = pred (TEST, TEST_hog, lin, lhog, output_layer)
        for ans, i in zip (ANSES, indices[1]):
            s = tostr(ans.argmax())
            f.write('%d,%s\n' % (i+1, s))
