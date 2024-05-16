import numpy as np
import cv2 as cv2
import easyocr
from statistics import mode
import tkinter as tk
from PIL import Image, ImageTk

from plate_detect import extrac_text
from color_detect import get_dominant_color
from logo_detect import crop_logo
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

img = cv2.imread("photos_to_detect/subaru_back.jpeg")
number = extrac_text(img)
color = get_dominant_color(img)

color_show = np.zeros((200, 200, 3), dtype=np.uint8)
color_show[:] = color

cropped = crop_logo(img)
cropped = preprocessing(cropped)
cropped = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_AREA)

logo = get_logo_name(cropped)

number_text = mode(number) if number else "N/A"
logo_text = logo if logo else "N/A"

# Создание окна Tkinter
root = tk.Tk()
root.title("Image Analysis")
root.geometry("600x600")

# Отображение текста номера
number_label = tk.Label(root, text=number_text, font=("Arial", 20))
number_label.place(x=10, y=10)

# Отображение текста логотипа
logo_label = tk.Label(root, text=logo_text, font=("Arial", 20))
logo_label.place(x=10, y=550)

# Отображение изображения цветового поля
color_show = cv2.cvtColor(color_show, cv2.COLOR_BGR2RGB)  # Конвертация для корректного отображения в Tkinter
color_img = Image.fromarray(color_show)
color_img = ImageTk.PhotoImage(image=color_img)

color_label = tk.Label(root, image=color_img)
color_label.place(x=400, y=10)

# Отображение текста "color" поверх цветового поля
color_text_label = tk.Label(root, text="Color", font=("Arial", 20), bg=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}')
color_text_label.place(x=400, y=10)

# Отображение обрезанного изображения
cropped = cv2.cvtColor((cropped * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)  # Преобразование для корректного отображения в Tkinter
cropped_img = Image.fromarray(cropped)
cropped_img = ImageTk.PhotoImage(image=cropped_img)

cropped_label = tk.Label(root, image=cropped_img)
cropped_label.place(x=400, y=400)

# Запуск цикла обработки событий Tkinter
root.mainloop()
