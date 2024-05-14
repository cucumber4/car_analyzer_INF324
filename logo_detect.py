import numpy as np
import cv2 as cv
import easyocr
from statistics import mode
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4

img1 = cv.imread('logos/img.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('photos_to_detect/lexus.jpeg', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)


else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 255),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
texts = []


for point in range(len(good)):
    try:
        print(dst_pts[point])

        center_point = dst_pts[point]
        top_left_point = center_point - np.array([-20, 80])
        bottom_right_point = center_point + np.array([250, 100])

        print(top_left_point, '\n', bottom_right_point)

        # Предположим, что вырезаемый прямоугольник обозначен переменными top_left_point и bottom_right_point

        # Округляем координаты, чтобы они были целыми числами
        top_left_point = np.round(top_left_point).astype(int)
        bottom_right_point = np.round(bottom_right_point).astype(int)

        # Преобразуем координаты в корректный формат
        top_left_point = tuple(top_left_point[0])
        bottom_right_point = tuple(bottom_right_point[0])

        # Вырезаем прямоугольник изображения
        cropped_img = img2[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]


        _, gray = cv.threshold(cropped_img, 120, 255, cv.THRESH_BINARY)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(gray)
        text = str(result[0][1]).upper()

        print(text)
        texts.append(text)

        plt.imshow(cropped_img, 'gray'), plt.show()

        print(len(text))

    except Exception as e:
        print("Error:", e)

plt.imshow(img3, 'gray'), plt.show()

print(texts)
print(mode(texts))


