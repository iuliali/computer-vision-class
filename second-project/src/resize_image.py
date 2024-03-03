import glob
import os

import cv2 as cv

from constants import *

chosen_character = BARNEY

read_folder = f"../data/cropped/{chosen_character}"
resized_folder = f"../data/{TRAINING_DIR}/{POSITIVES}"


def resize_image(image):
    h, w = image.shape[:2]
    if w > h:
        return cv.resize(image, (BIGGER_SIZE, SMALLER_SIZE))
    else:
        return cv.resize(image, (SMALLER_SIZE, BIGGER_SIZE))


def resize_images(read_dir, resized_dir):
    images_path = os.path.join(read_dir, '*.jpg')
    files = glob.glob(images_path)
    num_images = len(files)

    for i in range(num_images):
        img = cv.imread(files[i], cv.IMREAD_COLOR)
        # show_image(img)
        resized_img = resize_image(img)
        # show_image(resized_img)
        filename = f"{resized_dir}/{chosen_character}_{files[i].split('/')[-1]}"
        print(filename, resized_img.shape)
        cv.imwrite(filename, resized_img)

if __name__ == "__main__":
    resize_images(read_folder, resized_folder)
