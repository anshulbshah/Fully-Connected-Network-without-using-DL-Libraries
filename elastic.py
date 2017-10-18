#Elastic Distortion : https://github.com/KyotoSunshine/CNN-for-handwritten-kanji/blob/master/create_augmented_dataset.py
import cPickle, gzip
from numpy import *
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import pylab
import matplotlib.pyplot as plt
import math
import shutil
from scipy.signal import convolve2d

def openmnist(mnist):
	f = gzip.open(mnist, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	#print(train_set[1][255])
	f.close()
	return train_set

def show_image(data_vec):
    # A function to show the image
    #print data_vec.shape
    test_image = 255*reshape(data_vec,(28,28))
    img = Image.fromarray(uint8(test_image.T))
    img.show()
def elastic_distortion(image, kernel_dim=31, sigma=4, alpha=40):

    # Returns gaussian kernel in two dimensions
    # d is the square kernel edge size, it must be an odd number.
    # i.e. kernel is of the size (d,d)
    def gaussian_kernel(d, sigma):
        if d % 2 == 0:
            raise ValueError("Kernel edge size must be an odd number")

        cols_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)))
        rows_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)).reshape(d, 1))

        kernel = np.exp(-1. * ((rows_identifier - d/2)**2 +
            (cols_identifier - d/2)**2) / (2. * sigma**2))
        kernel *= 1. / (2. * math.pi * sigma**2)  # normalize
        return kernel

    field_x = np.random.uniform(low=-1, high=1, size=image.shape) * alpha
    field_y = np.random.uniform(low=-1, high=1, size=image.shape) * alpha

    kernel = gaussian_kernel(kernel_dim, sigma)

    # Distortion fields convolved with the gaussian kernel
    # This smoothes the field out.
    field_x = convolve2d(field_x, kernel, mode="same")
    field_y = convolve2d(field_y, kernel, mode="same")

    d = image.shape[0]
    cols_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)))
    rows_identifier = np.int32(
        np.ones((d, d))*np.array(np.arange(d)).reshape(d, 1))

    down_row = np.int32(np.floor(field_x)) + rows_identifier
    top_row = np.int32(np.ceil(field_x)) + rows_identifier
    down_col = np.int32(np.floor(field_y)) + cols_identifier
    top_col = np.int32(np.ceil(field_y)) + cols_identifier
#    plt.imshow(field_x, cmap=pylab.cm.gray, interpolation="none")
#    plt.show()

    padded_image = np.pad(
        image, pad_width=d, mode="constant", constant_values=0)

    x1 = down_row.flatten()
    y1 = down_col.flatten()
    x2 = top_row.flatten()
    y2 = top_col.flatten()

    Q11 = padded_image[d+x1, d+y1]
    Q12 = padded_image[d+x1, d+y2]
    Q21 = padded_image[d+x2, d+y1]
    Q22 = padded_image[d+x2, d+y2]
    x = (rows_identifier + field_x).flatten()
    y = (cols_identifier + field_y).flatten()

    # Bilinear interpolation algorithm is as described here:
    # https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
    distorted_image = (1. / ((x2 - x1) * (y2 - y1)))*(
        Q11 * (x2 - x) * (y2 - y) +
        Q21 * (x - x1) * (y2 - y) +
        Q12 * (x2 - x) * (y - y1) +
        Q22 * (x - x1) * (y - y1))

    distorted_image = distorted_image.reshape((d, d))
    return distorted_image

# distorted_image = elastic_distortion(reshape(train_set[0][300],(28,28)))
# show_image(distorted_image)
# show_image(reshape(train_set[0][300],(28,28)))
# # show_image(distorted_image)
# show_image(train_set[0][300])
def mainfun(mnist):
	train_set=openmnist(mnist)
	elastic_set = train_set[0].copy()
	elastic_label = train_set[1].copy()
	f = open('test.pl','w')
	for i in range(50000):
	    distorted_image = elastic_distortion(reshape(train_set[0][i],(28,28)))
	    elastic_set[i,:] = reshape(distorted_image,(784,))

	cPickle.dump([elastic_set,elastic_label],f)
	f.close()

	f = open('test.pl','rb')
	elastic_set,elastic_label = cPickle.load(f)
	#show_image(elastic_set[300])
	#show_image(train_set[0][300])

	f2 = open('final2.pkl','w')
	cPickle.dump([concatenate((train_set[0],elastic_set),axis = 0),concatenate((train_set[1],elastic_label),axis = 0)],f2)
	f2.close()

	with open('final2.pkl', 'rb') as f_in, gzip.open('final2.pkl.gz', 'wb') as f_out:
	    shutil.copyfileobj(f_in, f_out)
	#distorted_image.show()
