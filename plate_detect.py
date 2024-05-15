import numpy as np
import cv2 as cv
import easyocr
from statistics import mode
from matplotlib import pyplot as plt


def extrac_text(img2):
    MIN_MATCH_COUNT = 4
    sift = cv.SIFT_create()
    img1 = cv.imread('photos_to_detect/img_1.png', cv.IMREAD_GRAYSCALE)
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
            # print(dst_pts[point])

            center_point = dst_pts[point]
            top_left_point = center_point - np.array([-20, 80])
            bottom_right_point = center_point + np.array([300, 100])

            # print(top_left_point, '\n', bottom_right_point)

            top_left_point = np.round(top_left_point).astype(int)
            bottom_right_point = np.round(bottom_right_point).astype(int)

            top_left_point = tuple(top_left_point[0])
            bottom_right_point = tuple(bottom_right_point[0])

            cropped_img = img2[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]


            _, gray = cv.threshold(cropped_img, 120, 255, cv.THRESH_BINARY)

            reader = easyocr.Reader(['en'])
            result = reader.readtext(gray)

            text = str(result[0][1]).upper()
            letters = [letter for letter in text]
            # print(letters)
            #
            # print(letters[1:4])

            if letters[-1] in [']', '!', ')', '}']:
                letters.pop()


            if letters[-1] in ['S', 'J']:
                letters[-1] = '5'
            if letters[-1] in ['Z']:
                letters[-1] = '2'
            if letters[-1] in ['O']:
                letters[-1] = '0'
            if letters[-1] == 'I':
                letters[-1] = '1'
            if letters[-1] == 'T':
                letters[-1] = '1'

            if letters[0] in ['1','2','Z','I'] and letters[3] in ['0','1','2','3','4','5','6','7','8','9']:
                letters.pop(0)

            if len(letters) > 8:
                letters.pop(-3)

            sub_list_l = letters[3:6]
            # print(sub_list_l)
            for i in range(len(sub_list_l)):
                if sub_list_l[i] == '2':
                    sub_list_l[i] = 'Z'
                if sub_list_l[i] == '4':
                    sub_list_l[i] = 'A'
                if sub_list_l[i] == '0':
                    sub_list_l[i] = 'O'
                if sub_list_l[i] == '1':
                    sub_list_l[i] = 'I'
            letters[3:6] = sub_list_l


            sub_list_n = letters[0:3]
            for i in range(len(sub_list_l)):
                if sub_list_n[i] == "I":
                    sub_list_n[i] = '1'
                if sub_list_n[i] == 'Z':
                    sub_list_n = '2'
                if sub_list_n[i] == '0':
                    sub_list_n = '0'

            if letters[-2] == 'I':
                letters[-2] = '1'
            if letters[-2] == 'D':
                letters[-2] = '0'

            text = "".join(letters)
            # print(text)
            texts.append(text)

            plt.imshow(cropped_img, 'gray'), plt.show()

            # print(len(text))

        except Exception as e:
            print("Error:", e)
    # plt.imshow(img3, 'gray'), plt.show()
    return texts

img2 = cv.imread('photos_to_detect/huyndai.jpeg', cv.IMREAD_GRAYSCALE)

texts = extrac_text(img2)
texts_rm = list(set(texts))
# print(texts)
# print(texts_rm)
# print(mode(texts))


