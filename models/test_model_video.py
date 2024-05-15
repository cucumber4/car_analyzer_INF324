import numpy as np
import cv2
import pickle


def get_logo_name(imgOriginal):
    # def grayscale(imgOriginal):
    #     img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    #     return img
    #
    # def equalize(imgOriginal):
    #     img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    #     img = cv2.equalizeHist(img)
    #     return img
    #
    # def preprocessing(img):
    #     img = grayscale(imgOriginal)
    #     img = equalize(imgOriginal)
    #     img = img / 255
    #     return img

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


    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (50, 50))
    fg_mask = bg_subtractor.apply(imgOriginal)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 50, 50, 1)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return str(getClassName(classIndex))


threshold = 0.75  # ПОРОГ ВЕРОЯТНОСТИ

pickle_in = open("models/model_trained.p", "rb")  ## rb = ЧТЕНИЕ БАЙТОВ
model = pickle.load(pickle_in)

bg_subtractor = cv2.createBackgroundSubtractorMOG2()




