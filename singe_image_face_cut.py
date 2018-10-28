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

img = cv2.imread('./my_image/subject2.jpg')

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


frame_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frame_resized = resize(frame_grey, width=200, height=200)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(frame_resized, 1)
if len(dets) > 0:
    for k, d in enumerate(dets):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(frame_resized, d)
        shape = shape_to_np(shape)



shape = np.array(shape)/ratio

cut_point = (np.array([shape[0], shape[16], shape[19], shape[8]], dtype=int))

for (x, y) in cut_point:
    cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)

cropped = img[cut_point[2][1]:cut_point[3][1], cut_point[0][0]:cut_point[1][0]]
cv2.imshow("image", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()