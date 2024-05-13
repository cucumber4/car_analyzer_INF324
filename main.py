import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('photos_to_detect/lexus.png')

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Создание объекта BackgroundSubtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Применение метода вычитания фона
foreground_mask = bg_subtractor.apply(gray)

# Пороговая обработка маски
_, thresholded_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

# Найденеие контуров объектов
contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отрисовка контуров на исходном изображении
result_image = image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# Отображение результатов
cv2.imshow('Original Image', image)
cv2.imshow('Foreground Mask', foreground_mask)
cv2.imshow('Thresholded Mask', thresholded_mask)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
