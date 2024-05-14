import numpy as np
import cv2
import pickle

#############################################

frameWidth = 640  # РАЗРЕШЕНИЕ КАМЕРЫ
frameHeight = 480
brightness = 180
threshold = 0.75  # ПОРОГ ВЕРОЯТНОСТИ
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# НАСТРОЙКА ВИДЕОКАМЕРЫ
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ
pickle_in = open("model_trained.p", "rb")  ## rb = ЧТЕНИЕ БАЙТОВ
model = pickle.load(pickle_in)

# Инициализация алгоритма вычитания фона MOG2
bg_subtractor = cv2.createBackgroundSubtractorMOG2()


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


def getClassName(classNo):
    if classNo == 0:
        return 'hyundai'
    elif classNo == 1:
        return 'lexus'
    elif classNo == 2:
        return 'mazda'
    elif classNo == 3:
        return 'mercedes'
    elif classNo == 4:
        return 'opel'
    elif classNo == 5:
        return 'skoda'
    elif classNo == 6:
        return 'toyota'
    elif classNo == 7:
        return 'volkswagen'

while True:

    success, imgOriginal = cap.read()


    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (50, 50))
    img = preprocessing(img)


    fg_mask = bg_subtractor.apply(imgOriginal)


    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 50, 50, 1)
    cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)


    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
