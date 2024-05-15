import cv2
import numpy as np
import torch

def crop_logo(img):
    logo_detection = torch.hub.load('ultralytics/yolov5', 'custom', path='models/logo_detection.pt')

    detection_result = logo_detection(img)
    image = cv2.imread(img_path)
    croped_image = None

    car_logo = detection_result.xywh[0]
    print(car_logo)

    try:
        x = car_logo[0][0].item()
        y = car_logo[0][1].item()
        w = car_logo[0][2].item()
        h = car_logo[0][3].item()
        prediction = car_logo[0][4].item()

    except:
        print("can't detect logo")
        exit()

    croped_image = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    return croped_image

img_path = "photos_to_detect/huyndai.jpeg"
croped_image = crop_logo(img_path)

if croped_image is not None:
    cv2.imshow("logo", croped_image)
    cv2.waitKey(0)  # Добавление этой строки позволит окну оставаться открытым
else:
    print("Failed to crop logo")
