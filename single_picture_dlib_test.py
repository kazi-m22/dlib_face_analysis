from __future__ import division
import dlib
import cv2
import numpy as np

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


img = cv2.imread('cry.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frame_resized = resize(gray, width=120)

dets = detector(frame_resized, 1)
print(dets)
if len(dets) > 0:
    for k, d in enumerate(dets):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(frame_resized, d)
        shape = shape_to_np(shape)
        shape = shape[48:68,:]
        print(shape[0:17,:])
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(img, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
        # cv2.rectangle(img, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()