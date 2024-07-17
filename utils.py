# Importing libaries
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import easyocr
import re


# Loading model to predict the license plate from car images
reader = easyocr.Reader(['en'])
model3 = YOLO(r'C:\Users\rezaa\spyder\Automated-Vehicle-Number-Plate-Recognition\models\best.pt')




def plate(img):

    '''
    This function takes an image of a car and checks if there ia any license plate or not
    and returns a cropped image of the license plate
    '''


    results = model3.predict(img, verbose=False)
    plate_img=None
    plate_detected = False
    s1 = 0
    if results[0].boxes.cls.nelement() != 0:
        plate_detected = True
        cor = results[0].boxes.xyxy[0]
        r1, s1, r2, s2 = [int(num) for num in cor.tolist()]
        plate_img = results[0].orig_img[s1:s2,r1:r2,:]
    return plate_img, plate_detected, s1


def preprocessing(img_r):

    '''
    This function takes a image and converts the image into text
    '''
    img = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    text = ''
    for _, t, _ in reader.readtext(img):
        text += t
    text = text.upper()
    text = re.sub(r'\W+', '', text)

    return text

if __name__ == '__main__':
    print(preprocessing(cv.imread(r"C:\Users\rezaa\spyder\Automated-Vehicle-Number-Plate-Recognition\sample_video\Screenshot 2024-07-17 112240.jpg")))

