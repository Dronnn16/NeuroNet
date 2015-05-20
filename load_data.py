import skimage
import numpy as np

def float32(x):
    return np.cast['float32'](x)

def load_data (pathes):
    TEST = []
    TEST_hog = []
    for i in xrange(len(pathes)):
        #print ('loading %s\n' % pathes[i])
        image = skimage.io.imread(pathes[i])
        hog = skimage.feature.hog(skimage.color.rgb2grey(image))
        normimge = float32(image/float32(255))
        R =  np.asarray([[pixel[0] for pixel in string] for string in normimge])
        G =  np.asarray([[pixel[1] for pixel in string] for string in normimge])
        B =  np.asarray([[pixel[2] for pixel in string] for string in normimge])
        TEST.append(np.asarray([R, G, B]))
        TEST_hog.append(float32(hog))

    TEST = np.asarray(TEST)
    TEST_hog = np.asarray(TEST_hog)

    return TEST, TEST_hog