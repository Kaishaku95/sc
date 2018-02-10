from __future__ import print_function
from ffvideo import VideoStream
import matplotlib.pyplot as plt
import cv2
import numpy as np
import collections

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
from prompt_toolkit.layout.menus import _SelectedCompletionMetaControl

pylab.rcParams['figure.figsize'] = 16, 12


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        #if (area > 70 and h < 30 and h > 12) or (h> 12 and h < 30 and w < 10):
        if 17 < h < 30:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions


def scale_to_range(image):  # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image / 255


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()

i = 0

for frame in VideoStream('videos/video-9.avi'):

    i = i + 1
    if not i % 40 == 0:
        continue

    frame.image().save('temp.png')
    image_color = load_image('temp.png')
    img = invert(image_bin(image_gray(image_color)))
    img_bin = erode(dilate(img))
    selected_regions, numbers = select_roi(image_color.copy(), img)
    new = cv2.addWeighted(image_color, 1, selected_regions, 1, 0)
    new = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
    if i % 40 == 0:
#        cv2.imwrite('i2v/'+str(i)+'.png', new)
        plt.imshow(new)
        plt.show()

#print(i)

