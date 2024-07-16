# Importing libaries
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import easyocr
from ultralytics.utils import files


# Loading model to predict the license plate from car images
reader = easyocr.Reader(['en'])
model3 = YOLO('./models/plate_detector_model.pt')




def plate(img):

    '''
    This function takes an image of a car and checks if there ia any license plate or not
    and returns a cropped image of the license plate
    '''


    results = model3.predict(img, verbose=False)
    plate_img=None
    plate_detected = False
    if results[0].boxes.cls.nelement() != 0:
        plate_detected = True
        cor = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = [int(num) for num in cor.tolist()]
        plate_img = results[0].orig_img[y1:y2,x1:x2,:]
    return plate_img, plate_detected


def preprocessing(img_r):

    '''
    This function takes a image and converts the image into text
    '''
    img = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    text = ''
    for _, t, _ in reader.readtext(img):
        text += t
    return text

