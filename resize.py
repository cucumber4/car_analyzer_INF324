import os
import cv2

# Папка с исходными изображениями
source_folder = 'photos_to_detect'

# Папка для сохранения измененных изображений
target_folder = 'photos_to_detect'

# Размер, к которому вы хотите изменить изображения
target_size = (960, 1280)

# Создаем целевую папку, если она не существует
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Получаем список файлов в папке с исходными изображениями
image_files = os.listdir(source_folder)

# Проходим по каждому файлу и изменяем размер
for image_file in image_files:
    # Получаем путь к исходному изображению
    source_path = os.path.join(source_folder, image_file)
    # Загружаем изображение
    image = cv2.imread(source_path)
    # Изменяем размер изображения
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    # Получаем путь к целевому изображению
    target_path = os.path.join(target_folder, image_file)
    # Сохраняем измененное изображение
    cv2.imwrite(target_path, resized_image)

print("Изображения успешно изменены и сохранены в", target_folder)
