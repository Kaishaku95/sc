from __future__ import print_function
from ffvideo import VideoStream
import cv2
import numpy as np
from Line import Line
import os
#import matplotlib.pyplot as plt


minLineLength = 150
maxLineGap = 20
rho = 1
theta = np.pi / 180
line = 2
threshold = 20
apertureSize = 3
threshold1 = 100
threshold2 = 200

def find_lines():

    vlines = []

    for v in range(0, 10):
        vname = 'videos/video-' + str(v) + '.avi'
        for frame in VideoStream(vname):
            frame.image().save('frame0/fr' + str(v) + '.png')
            break

    for i in range(0, 10):

        img = cv2.imread('frame0/fr' + str(i) + '.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(image=gray, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize)

        lines = None
        lines = cv2.HoughLinesP(image=edges, rho=rho, theta=theta, threshold=threshold, lines=line,
                                minLineLength=minLineLength,
                                maxLineGap=maxLineGap)

        ll = []
        points = []
        for linee in lines:
            for s in linee:
                ll.append(s)

        for k in range(0, 2):  # otklanjamo visak
            x11, y11, x12, y12 = ll[k]
            for ii in range(len(ll) - 1, k, -1):
                x21, y21, x22, y22 = ll[ii]
                # print(x21 - x11, y21 - y11, x22 - x12, y22 - y12)
                if abs(x21 - x11) + abs(y21 - y11) + abs(x22 - x12) + abs(y22 - y12) < 100:
                    del ll[ii]

        order = []
        if ll[1][3] < ll[0][3]:  ##po potrebi menjamo redosled linija, prva linija je uvek plava
            order = [ll[1], ll[0]]
        else:
            order = [ll[0], ll[1]]

        ll = order
#        tt = 0
        for x1, y1, x2, y2 in ll:
            l = Line(x1, y1, x2, y2)
        #    print(x1, y1, x2, y2)
            points.append(l)
        #    cv2.line(img, (x1, y1), (x2, y2), (255, 255 * (tt), tt * 255), 1)  # why BGR
        #    tt += 1

        vlines.append(points)
        os.remove('frame0/fr'+str(i)+'.png')

    return vlines
