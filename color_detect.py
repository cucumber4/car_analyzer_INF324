import cv2
import numpy as np
import torch

def get_dominant_color(image, k=4, image_processing_size=None):
    data = np.reshape(image, (-1, 3))
    data = np.float32(data)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    print(dominant)
    return dominant



img_path = 'photos_to_detect/corolla.jpeg'
image = cv2.imread(img_path)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image_rgb)
pred = results.pred[0]

max_area = 0
largest_box = None


for det in pred:
    if int(det[5]) == 2:
        x1, y1, x2, y2 = map(int, det[:4])
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)


x1, y1, x2, y2 = largest_box
cropped_image = image[y1:y2, x1:x2]
dominant_color = get_dominant_color(cropped_image)


cv2.imshow("cropped ", cropped_image)
cv2.waitKey(0)

