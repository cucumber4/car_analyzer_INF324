import numpy as np
import cv2 as cv2
import easyocr
from statistics import mode

from plate_detect import extrac_text
from color_detect import get_dominant_color
from  logo_detect import crop_logo
from models import *
from termproject.models.test_model_video import get_logo_name


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

img = cv2.imread("photos_to_detect/lexus.jpeg")
print(get_dominant_color(img))
print(extrac_text(img))

ropped = crop_logo(img)
ropped = preprocessing(ropped)
ropped = cv2.resize(ropped,(50,50), interpolation=cv2.INTER_AREA)
print(get_logo_name(ropped))
