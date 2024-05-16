import cv2
import numpy as np
import torch


def crop_logo(img):
    # Загрузка модели детекции логотипов
    logo_detection = torch.hub.load('ultralytics/yolov5', 'custom', path='models/logo_detection.pt')

    # Если img это путь к изображению, загрузить его
    if isinstance(img, str):
        image = cv2.imread(img)
    else:
        image = img

    detection_result = logo_detection(image)
    cropped_image = None

    car_logo = detection_result.xywh[0]

    try:
        x = car_logo[0][0].item()
        y = car_logo[0][1].item()
        w = car_logo[0][2].item()
        h = car_logo[0][3].item()
        prediction = car_logo[0][4].item()
    except:
        print("Can't detect logo")
        return None

    cropped_image = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    resized_image = cv2.resize(cropped_image, (50, 50), interpolation=cv2.INTER_AREA)
    return resized_image


img_path = "photos_to_detect/corolla.jpeg"
cropped_image = crop_logo(img_path)
#
# if cropped_image is not None:
#     cv2.imshow("Cropped Logo", cropped_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to crop logo")
