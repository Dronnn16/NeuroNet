import skimage
import numpy as np

def float32(x):
    return np.cast['float32'](x)

def load_test (pathes):
    TEST = []
    TEST_hog = []
    for i in xrange(len(pathes)):
        print ('loading %s\n' % pathes[i])
        image = skimage.io.imread(pathes[i])
        hog = skimage.feature.hog(skimage.color.rgb2grey(image))
        TEST.append(float32(image/float32(255)))
        TEST_hog.append(float32(hog))

    TEST = np.asarray(TEST).reshape((-1, 3, 32, 32))
    TEST_hog = np.asarray(TEST_hog)
    return TEST, TEST_hog