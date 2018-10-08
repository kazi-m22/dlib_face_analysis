#author: kazi mejbaul islam



from __future__ import division
import dlib
import cv2
import numpy as np
import math
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def distance(x,y):
    return math.sqrt( (x[1] - y[1])**2 + (x[0] - y[0])**2 )

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread('./my_image/1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets = detector(gray, 1)

shape = predictor(gray, dets[0])
shape = shape_to_np(shape)

# for (x, y) in shape:
#     cv2.circle(img, (x,y), 3, (255, 255, 255), -1)
right_lip = tuple(shape[48])
left_lip = tuple(shape[54])
lip_len = distance(right_lip,left_lip)

top_lip = tuple(shape[51])
bottom_lip = tuple(shape[57])
lip_width = distance(top_lip,bottom_lip)

cv2.line(img,right_lip,left_lip,(255,0,0),5)
cv2.line(img,top_lip,bottom_lip,(255,0,0),5)

eye_border_len = distance(shape[36], shape[45])
eye_inner_len = distance(shape[39], shape[42])
cv2.line(img, tuple(shape[36]), tuple(shape[45]), (255,255,255), 2)
cv2.line(img,tuple(shape[39]), tuple(shape[42]), (0,0,255), 2)

nose_left = shape[35]
nose_right = shape[31]

eye_ratio = eye_border_len/eye_inner_len
lip_ratio = lip_len/lip_width
print(lip_ratio, 'lip')
print(eye_ratio,'eye')

right_angle = math.degrees(math.atan(abs(right_lip[1]-nose_left[1])/abs(nose_left[0]-right_lip[0])))
left_angle = math.degrees(math.atan(abs(left_lip[1]-nose_right[1])/abs(nose_right[0]-left_lip[0])))
cv2.imwrite('output.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()