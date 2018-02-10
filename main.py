import houghlines
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import math

mode = 1

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def contact(x, y, k, n):
    if 2 <= (k * x + n) - y <= 6:
        return True
    return False


def democracy(ann, roi):
    roi /= 255
    roi = 1 - roi
    #print roi.shape
    global mode
    listar = np.ndarray(shape=(1, 28, 28, 1)) if mode == 1 else np.ndarray(shape=(1, 784))

    if mode == 1:
        for a in range(0, 28):
            for b in range(0, 28):
                listar[0][a][b][0] = roi[a * 28 + b]
    else:
        for a in range(0, 28):
            for b in range(0, 28):
                listar[0][a * 28 + b] = roi[a * 28 + b]

    res = [0] * 10
    for nn in ann:
        num = nn.predict(listar)
        #        print(num)
        #        print(max( (v, i) for i, v in enumerate(num[0]) )[1])
        res[max((v, i) for i, v in enumerate(num[0]))[1]] += 1
#    print(res)
    return max((v, i) for i, v in enumerate(res))[1]


def nearby(contactpoint, x, y):
    for xx, yy, time in contactpoint:
        if ((xx - x) ** 2 + (yy - y) ** 2) ** 0.5 < 22:
            return True
    return False


#############################################

threshold = 16

ann = []

ann.append(load_model('ann'+str(mode)+'.h5'))

lines = houghlines.find_lines()

fout = open('out.txt', 'w')
fout.write('RA 185/2014 Daniel Njari')
fout.write('\nfile\tsum\n')
fout.close()

for i in range(0, 10):  # 10
    vname = 'videos/video-' + str(i) + '.avi'
    vc = cv2.VideoCapture(vname)
    vc.set(1, 0)

    framec = 0

    blue_line = lines[i][0]
    green_line = lines[i][1]

    print(blue_line.getK(), blue_line.getN(), green_line.getK(), green_line.getN())
    print(blue_line.x1, blue_line.y1, blue_line.x2, blue_line.y2)
    print(green_line.x1, green_line.y1, green_line.x2, green_line.y2)

    contact_points = []

    score = 0
    while True:
        framec += 1
        ret, frame = vc.read()
        if not ret:
            break
        else:
            imgc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = invert(image_bin(image_gray(imgc)))
            img_bin = erode(dilate(img))

            img_, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if h < 16:
                    continue

                #cetvrti kvadrant
                bottomRightX = x + w
                bottomRightY = y + h

                if (blue_line.x1 <= bottomRightX <= blue_line.x2 ) and contact(
                                bottomRightX, bottomRightY, blue_line.getK(),
                        blue_line.getN()) and not nearby(contact_points, bottomRightX, bottomRightY):
                    # print('x1 and x2' + str(blue_line.x1) +' and ' +str(blue_line.x2))
                    roi = img[y:bottomRightY + 1, x:bottomRightX + 1]
                    roi = resize_region(roi)
                    roia = roi.flatten()
                    rez = democracy(ann, roia)
                    score += rez
                    print('added ' + str(rez) + ' at frame ' + str(framec))
                    #print(x + w, y + h)
                    contact_points.append([bottomRightX, bottomRightY, threshold])
                    #plt.imshow(frame)
                    #plt.show()

                if (green_line.x1 <= bottomRightX <= green_line.x2 ) and contact(
                        bottomRightX, bottomRightY, green_line.getK(),
                        green_line.getN()) and not nearby(contact_points, bottomRightX, bottomRightY):
                    roi = img[y:bottomRightY + 1, x:bottomRightX + 1]
                    roi = resize_region(roi)
                    roia = roi.flatten()
                    rez = democracy(ann, roia)
                    score -= rez
                    print('subtracted ' + str(rez) + ' at frame ' + str(framec))
                    #print(x + w, y + h)
                    contact_points.append([bottomRightX, bottomRightY, threshold])
                    #plt.imshow(frame)
                    #plt.show()

        for counter in range(len(contact_points)-1, -1, -1):
            tmp = contact_points[counter]
            tmp[2] -= 1
            if tmp[2] == 0:
                del contact_points[counter]

    fout = open('out.txt', 'a')
    fout.write('video-' + str(i) + '.avi' + '\t' + str(score) + '\n')
    fout.close()
    print(score)
