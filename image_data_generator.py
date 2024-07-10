import os
import zipfile
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "Could not import PIL.Image. The use of `load_img` requires PIL. "
        "Please install it using `pip install pillow`."
    )

# Проверка наличия scipy
try:
    import scipy
except ImportError:
    raise ImportError(
        "This script requires the scipy module. Please install it using `pip install scipy`."
    )

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
)

original_dataset_dir = 'C:/Users/inger/PycharmProjects/week7_day4/ExercisesXP/Execise5/Dog Images/images/images/n02085620-Chihuahua'
augmented_dataset_dir = 'C:/Users/inger/PycharmProjects/week7_day4/ExercisesXP/Execise5/Augmented_dataset/n02085620-Chihuahua'
os.makedirs(augmented_dataset_dir, exist_ok=True)

image_files = os.listdir(original_dataset_dir)

max_images = 10

for filename in image_files[:max_images]:
    img_path = os.path.join(original_dataset_dir, filename)
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dataset_dir, save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 5:
            break

    print(f"Augmented images saved for {filename}")

print("Augmentation process completed.")