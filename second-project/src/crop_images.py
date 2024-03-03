import os

import cv2 as cv
import numpy as np

from annotations import *
from src.resize_image import resize_image
from src.show_image import show_image

### CROP POSITIVE AND NEGATIVE EXAMPLES FOR SVM TRAINING
chosen_character = WILMA

barney_folder = os.path.join(ANTRENARE_DIR, chosen_character)
barney_write_folder = os.path.join(DATA_DIR, POSITIVES)
path_annotations_barney = os.path.join(ANTRENARE_DIR, annotation(chosen_character))

WRITE_NEGATIVES = os.path.join(DATA_DIR, TRAINING_DIR, NEGATIVES)


def aspect_ratio_formatted(width, height):
    if width < height:
        return f" 1 : {height / width} "
    else:
        return f" {width / height} : 1 "


def resize_face_to_closest_square(image, rect: Rectangle):
    h, w = rect.height(), rect.width()
    tolerance = 3
    dif = abs(w - h) // 2
    if w - h > tolerance:
        return image[rect.top: rect.top + w,
               max(rect.left - dif, 0): min(rect.left - dif + w, image.shape[1])]
    elif h - w > tolerance:
        return image[max(rect.top - dif, 0): min(rect.top - dif + h, image.shape[0]),
               rect.left: rect.left + h]
    else:
        return image[rect.top: rect.top + h,
               rect.left: rect.left + h]


def crop_images_with_coords(path_dir, path_annotations, write_folder):
    # read annotations
    annotations = annotations_reader(path_annotations)

    for image_name in os.listdir(path_dir):
        print(f"Reading {image_name}")
        image = cv.imread(os.path.join(path_dir, image_name), cv.IMREAD_COLOR)
        # show_image(image)
        print(image.shape)
        assert image.shape == (H_IMAGE, W_IMAGE, 3), f"crop_images_with_coords: OTHER DIM FOR READ IMAGES"
        rects_in_image = get_coords_from_annotations(annotations, image_name)
        for rect, character in rects_in_image:
            image_name_save = f"{character}_{str(rect)}{JPG_FORMAT}"
            print(f"Annotation size: w {rect.width()}, h {rect.height()}")
            print(f"Aspect ratio: {aspect_ratio_formatted(rect.width(), rect.height())}")
            face = image[rect.top: rect.bottom,
                   rect.left: rect.right]
            # face = resize_face_to_closest_square(image, rect)
            # print(face.shape)
            # print(f"Face of : {character}")
            # print(image_name_save)
            # show_image(face)
            # print(os.path.exists(write_folder))
            print(cv.imwrite(os.path.join(write_folder, f"{image_name_save}{JPG_FORMAT}"), face))


def crop_negatives_examples(path_dir, path_annotations, write_folder_negatives):
    # read annotations
    annotations = annotations_reader(path_annotations)

    for image_name in os.listdir(path_dir):
        print(f"Reading {image_name}...")
        image = cv.imread(os.path.join(path_dir, image_name), cv.IMREAD_COLOR)
        # show_image(image)
        print(image.shape)
        rects_in_image = get_coords_from_annotations(annotations, image_name)
        num_cols, num_rows = image.shape[:2]
        patch1 = get_random_patch_except_rect(image, num_cols - BIGGER_SIZE, num_rows - SMALLER_SIZE,
                                              BIGGER_SIZE, SMALLER_SIZE, rects_in_image)
        patch2 = get_random_patch_except_rect(image, num_cols - SMALLER_SIZE, num_rows - BIGGER_SIZE,
                                              SMALLER_SIZE, BIGGER_SIZE, rects_in_image)
        # show_image(patch1)
        # show_image(patch2)
        if patch1 is not None:
            print("Patch 1 shape: ", patch1.shape)
            filename = os.path.join(write_folder_negatives,
                                    f"{image_name.strip(JPG_FORMAT)}_{chosen_character}3{JPG_FORMAT}")
            print(cv.imwrite(filename, patch1))
        if patch2 is not None:
            print("Patch 2 shape: ", patch2.shape)
            filename = os.path.join(write_folder_negatives,
                                    f"{image_name.strip(JPG_FORMAT)}_{chosen_character}4{JPG_FORMAT}")
            print(cv.imwrite(filename, patch2))


def get_random_patch_except_rect(image, high_random1, high_random2, dim_h, dim_w, rects_excepted: Rectangle):
    tries = 5
    while tries > 0:
        x = np.random.randint(low=0, high=high_random2)
        y = np.random.randint(low=0, high=high_random1)
        rect: Rectangle = Rectangle(x, y, x + dim_w, y + dim_h)
        patch = image[y: y + dim_h, x: x + dim_w]
        tries -= 1
        if not rect.intersects_any(rects_excepted):
            return patch
    return None


def move_only_character_images(from_path, to_path, character):
    bigger_first = 0
    smaller_first = 0
    for image_name in os.listdir(from_path):
        if image_name.split("_")[0] == character:
            print(image_name)
            image = cv.imread(os.path.join(from_path, image_name), cv.IMREAD_COLOR)
            image = resize_image(image)
            if image.shape[0] == BIGGER_SIZE:
                bigger_first += 1
            else:
                smaller_first += 1
            cv.imwrite(os.path.join(to_path, image_name), image)

    print(f"bigger first: {bigger_first}")


def crop_interactive(from_folder, to_folder):
    for image_name in os.listdir(from_folder):
        print(f"Reading {image_name}...")
        image = cv.imread(os.path.join(from_folder, image_name), cv.IMREAD_COLOR)
        x0, y0, width, height = cv.selectROI(np.uint8(image))
        cropped = image[y0:  y0 + height, x0:x0 + width]
        show_image(cropped)
        print(cv.imwrite(os.path.join(to_folder, f"new_crop_{x0}_{y0}_{width}_{height}_{image_name}"), cropped))


def crop_negatives_examples2(path_dir, write_folder_negatives):
    for image_name in os.listdir(path_dir):
        print(f"Reading {image_name}...")
        image = cv.imread(os.path.join(path_dir, image_name), cv.IMREAD_COLOR)
        # show_image(image)
        print(image.shape)
        try:
            # random1 = np.random.randint(0, image.shape[0] - BIGGER_SIZE)
            # random2 = np.random.randint(0, image.shape[1] - SMALLER_SIZE)
            # random3 = np.random.randint(0, image.shape[0] - SMALLER_SIZE)
            # random4 = np.random.randint(0, image.shape[1] - BIGGER_SIZE)
            # patch1 = image[random1:random1+BIGGER_SIZE, random2:random2+ SMALLER_SIZE]
            # patch2 = image[random3:random3+SMALLER_SIZE, random4:random4+BIGGER_SIZE]
            patch1 = resize_image(image)
            patch2 = None
        except Exception as e:
            continue
        # show_image(patch1)
        # show_image(patch2)
        if patch1 is not None:
            print("Patch 1 shape: ", patch1.shape)
            filename = os.path.join(write_folder_negatives,
                                    f"new_{image_name.strip(JPG_FORMAT)}1{JPG_FORMAT}")
            print(cv.imwrite(filename, patch1))
        if patch2 is not None:
            print("Patch 2 shape: ", patch2.shape)
            filename = os.path.join(write_folder_negatives,
                                    f"new_{image_name.strip(JPG_FORMAT)}2{JPG_FORMAT}")
            print(cv.imwrite(filename, patch2))


if __name__ == "__main__":
    # crop_interactive(barney_folder, barney_write_folder)
    # crop_images_with_coords(barney_folder, path_annotations_barney, barney_write_folder)
    crop_negatives_examples2(barney_write_folder, "../data/cropped_3_negatives")
    # from_dir = os.path.join(DATA_DIR, "cropped_2")
    # to_dir = os.path.join(DATA_DIR, TRAINING_DIR, POSITIVES, chosen_character)
    # move_only_character_images(from_dir, to_dir, chosen_character)
