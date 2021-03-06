#!/usr/bin/env python

import os
import re

import cv2
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tensorflow.python.keras.models import load_model

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
char_lookup = dict(zip([a for a in alphabet], [i for i in range(len(alphabet))]))
char_inverse_lookup = dict(map(reversed, char_lookup.items()))

HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.5

# For project details and context, please visit https://johnsonadavis.github.io/#projects under Autonomous Parking Agent

class license_plate_detection():
    '''
    Class used by robot to detect and read license plates from the ROS feed
    '''

    def __init__(self):
        rospy.init_node('license_plate_detection', anonymous=True)

        self.sess = tf.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()

        self.model = load_model(os.path.dirname(os.path.abspath(__file__)) + '/model2')
        self.model._make_predict_function()

        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback=self._image_callback,
                                           queue_size=1)
        self._license_pub = rospy.Publisher('/internal_plate_msg', String, queue_size=1)

    def _image_callback(self, image):
        '''
        Handles the robot response whenever an image is received from the feed.
        Parses the image and then proceeds to detection if valid.
        :param image: A rospy image
        :return: None
        '''
        try:
            img = self._bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            raise e

        letters, confidence = parse_image(img)
        if letters is not None and confidence is not None and len(letters) == 6:
            location, plate = self._get_plate_text(letters)
            if location is not None:
                self._send_license_plate(location, plate, confidence)

    def _get_plate_text(self, letters):
        '''
        Detects the text on license plates
        :param letters: A list of 6 cv2 images of dimension (150,100) in format (y,x) containing character candidates on
        a license plate
        :return: Location, Letters where Location is an integer with the location of the plate and Letters is an array
        of characters located on the plate
        '''
        output = ''
        for letter in letters:
            letter = letter * 1. / 255
            img_aug = np.expand_dims(letter, axis=0)
            with self.graph.as_default():
                tf.keras.backend.set_session(self.sess)
                y_predict = self.model.predict(img_aug)[0]
                prediction = char_inverse_lookup[np.argmax(y_predict)]
                output = output + prediction
        if is_valid(output):
            return output[1], output[2:]
        else:
            return None, None

    def _send_license_plate(self, loc, plate, confidence):
        '''
        Sends license plates to the robot's controller node
        :param loc: Integer - the location number of the plate
        :param plate: A list of the characters on the plate
        :param confidence: A metric of the confidence in the plate based on the area and the position of the plate in
        the image
        :return:
        '''
        print("info")
        print(loc)
        print(plate)
        print(confidence)
        output = ','.join([loc, plate[0], plate[1], plate[2], plate[3], str(confidence)])
        print("sending output" + str(output))
        self._license_pub.publish(String(output))


def deskew(img, pts1, pts2, shape):
    '''
    :param img: A cv2 image
    :param pts1: an array of points to be mapped to some desired corresponding coordinates
    :param pts2: An array of points for the desired output to map the first points to
    :param shape: A tuple of dimensions of the desired output image (x,y)
    :return: the cv2 image cropped to the desired rectangle and transformed to remove skew
    '''
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, shape)

    return dst


def find_order_pts(points):
    '''
    :param points: a numpy array or list of cartesian points (0,0) being top left in an image
    :return: returns the points ordered as [top left, top right, bottom right, bottom left]
    '''
    points = points[:, 0, :]
    # points array(array(x,y))
    e1 = np.mean(points[:, 0])
    e2 = np.mean(points[:, 1])

    low_low = None
    low_high = None
    high_low = None
    high_high = None

    for i in range(4):
        p = points[i]
        if p[0] > e1:
            if p[1] > e2:
                high_high = p
            else:
                high_low = p
        else:
            if p[1] > e2:
                low_high = p
            else:
                low_low = p

    if low_low is None or low_high is None or high_low is None or high_high is None:
        return points
    else:
        return np.array([low_low, low_high, high_high, high_low])


def parse_image(img):
    '''
    Parses the image for license plates and divides the text on license plates. Then, send the text to be detected.
    :param img: A cv2 image
    :return: Returns either a list of letters located on a license plate in frame, and and associated confidence metric
    based on teh area of the license plate and position in image or None, if none are found
    '''
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shape = (1280, 720)

    normal_lower_hsv = np.array([0, 0, 86])
    normal_upper_hsv = np.array([141, 35, 125])

    darker_lower_hsv = np.array([0, 0, 148])
    darker_upper_hsv = np.array([186, 18, 206])

    img_mask_1 = cv2.inRange(hsv_frame, normal_lower_hsv, normal_upper_hsv)
    img_mask_1 = cv2.resize(img_mask_1, shape)

    img_mask_2 = cv2.inRange(hsv_frame, darker_lower_hsv, darker_upper_hsv)
    img_mask_2 = cv2.resize(img_mask_2, shape)

    img_mask = img_mask_1 + img_mask_2

    img = cv2.resize(img, shape)

    gray = cv2.GaussianBlur(img_mask, (21, 21), 0)
    t, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(gray, 30, 200)

    im2, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt1 = None
    confidence = None

    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        cv2.drawContours(img, [approx], -1, (255, 255, 255), 3)
        if len(approx) == 4:
            if screenCnt1 is None:
                screenCnt1 = approx
                confidence = cv2.contourArea(c)
                break

    if screenCnt1 is None:
        return None, None
    else:
        screenCnt1 = find_order_pts(screenCnt1)

        if 120 > screenCnt1[1, 0]:
            if 50 > screenCnt1[1, 0]:
                confidence = confidence * LOW_CONFIDENCE
            else:
                confidence = confidence * HIGH_CONFIDENCE

        if screenCnt1[2, 0] > 1280 - 120:
            if screenCnt1[2, 0] > 1280 - 50:
                confidence = confidence * LOW_CONFIDENCE
            else:
                confidence = confidence * HIGH_CONFIDENCE

        detected = deskew(img, screenCnt1, [[0, 0], [0, 1800], [600, 1800], [600, 0]], (600, 1800))

        location = detected[:1800 - 298 - 260, :]
        plate = detected[1800 - 298 - 260:1800 - 260, :]

        plate_letters = [plate[50:260, 30:170], plate[50:260, 130:270], plate[50:260, 330:470], plate[50:260, 430:570]]
        location_letters = [location[650:1100, 0:300], location[650:1100, 300:]]

        letters = []
        model_shape = (100, 150)

        try:
            for j in range(2):
                letter = location_letters[j]
                letter = cv2.resize(letter, model_shape)
                letters.append(letter)

            for j in range(4):
                letter = plate_letters[j]
                letter = cv2.resize(letter, model_shape)
                letters.append(letter)
        except Exception as e:
            print(e)
            return None, None

        return letters, confidence


def is_valid(string):
    '''
    Determines if a code to be sent to the scoring node is valid, Must fit P#XX## where X is a letter and # is a number
    :param string: The input string of 6 characters
    :return: True if valid. False if not
    '''
    p = re.compile("^[P][0-9][A-Z][A-Z][0-9][0-9]$")

    return True if p.match(string) else False


if __name__ == "__main__":
    detection = license_plate_detection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down controller")

    cv2.destroyAllWindows()
