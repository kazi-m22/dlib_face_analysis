from __future__ import division
import dlib
import cv2
import numpy as np
import math
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import StandardScaler


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
def distance(x,y):
    return math.sqrt( (x[1] - y[1])**2 + (x[0] - y[0])**2 )

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

cap = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


model = Sequential()

model.add(Dense(1000, input_dim=9, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights('misbah_rafid_1000_500_500_100_99p.h5')



while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)

    if len(dets) > 0:
        shape = predictor(gray, dets[0])
        shape = shape_to_np(shape)

        # for (x, y) in shape:
        #     cv2.circle(img, (x,y), 3, (255, 255, 255), -1)
        right_lip = tuple(shape[48])
        left_lip = tuple(shape[54])
        lip_len = distance(right_lip, left_lip)
        up_nose = tuple(shape[27])
        bottom_nose = tuple(shape[33])
        right_nose = tuple(shape[31])
        left_nose = tuple(shape[35])

        top_lip = tuple(shape[51])
        bottom_lip = tuple(shape[57])
        lip_width = distance(top_lip, bottom_lip)

        cv2.line(img, right_lip, left_lip, (255, 0, 0), 5)
        cv2.line(img, top_lip, bottom_lip, (255, 0, 0), 5)

        left_eye = tuple(shape[45])
        right_eye = tuple(shape[36])
        eye_border_len = distance(shape[36], shape[45])
        eye_inner_len = distance(shape[39], shape[42])
        cv2.line(img, tuple(shape[36]), tuple(shape[45]), (255, 255, 255), 2)
        cv2.line(img, tuple(shape[39]), tuple(shape[42]), (0, 0, 255), 2)

        nose_length = distance(up_nose, bottom_nose)
        nose_width = distance(right_nose, left_nose)

        nose_ratio = round(nose_length / nose_width, 3)

        cv2.line(img, left_nose, right_nose, (255, 0, 0), 2)
        cv2.line(img, up_nose, bottom_nose, (255, 0, 0), 2)
        cv2.line(img, left_eye, left_lip, (25, 200, 180), 2)
        cv2.line(img, right_eye, right_lip, (25, 200, 180), 2)

        eye_ratio = round(eye_border_len / eye_inner_len, 3)
        lip_ratio = round(lip_len / lip_width, 3)
        # print(lip_ratio, 'lip')
        # print(eye_ratio,'eye')

        right_angle = round(
            math.degrees(math.atan(abs(right_lip[1] - left_nose[1]) / abs(left_nose[0] - right_lip[0]))), 3)
        left_angle = round(
            math.degrees(math.atan(abs(left_lip[1] - right_nose[1]) / abs(right_nose[0] - left_lip[0]))), 3)
        right_right_angle = round(
            math.degrees(math.atan(abs(right_lip[1] - right_nose[1]) / abs(right_nose[0] - right_lip[0]))), 3)
        left_left_angle = round(
            math.degrees(math.atan(abs(left_lip[1] - left_nose[1]) / abs(left_nose[0] - left_lip[0]))), 3)
        lef_l_e_angle = round(
            math.degrees(math.atan(abs(left_lip[1] - left_eye[1]) / abs(left_lip[0] - left_eye[0]))), 3)
        right_l_e_angle = round(
            math.degrees(math.atan(abs(right_lip[1] - right_eye[1]) / abs(right_lip[0] - right_eye[0]))), 3)

        write_list = [right_angle, right_right_angle, left_angle, left_left_angle, lip_ratio, eye_ratio, nose_ratio,
                      lef_l_e_angle, right_l_e_angle]

        score = model.predict(np.array([write_list,]))
        if int(score>.7):
            print("misbah found")

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

