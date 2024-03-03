import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob
from Parameters import *


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.directory_test, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            print((detection[0], detection[1]), (detection[2], detection[3]))
            print(detection[3] - detection[1], detection[2] - detection[0])
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.imwrite(os.path.join(params.directory_saved_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """

    ground_truth_bboxes = np.loadtxt(params.path_test_annotations, dtype='str')
    test_images_path = os.path.join(params.directory_test, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            print(image.shape)
            print((detection[0], detection[1]), (detection[2], detection[3]))
            print(detection[3] - detection[1], detection[2] - detection[0])
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                         (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(params.directory_saved_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_characters(character, detections, scores, file_names, params: Parameters):
    # ground_truth_bboxes = np.loadtxt(os.path.join(params.base_directory, "test",
    #                                               f"task2_{character}_gt_validare.txt"), dtype='str')
    test_images_path = os.path.join(params.directory_test, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            print(image.shape)
            print((detection[0], detection[1]), (detection[2], detection[3]))
            print(detection[3] - detection[1], detection[2] - detection[0])
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 125, 0), 2)
        # annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # # show ground truth bboxes
        # for detection in annotations:
        #     cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
        #                  (0, 255, 0), thickness=1)

        # cv.imwrite(os.path.join(params.directory_saved_files, f"{character}_detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


if __name__ == "__main__":
    params = Parameters()
    solution_path = "../results/task2/"
    character = "betty"
    detections = np.load(solution_path + f"{character}_detections.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(detections.shape)

    scores = np.load(solution_path + f"{character}_scores.npy", allow_pickle=True, fix_imports=True,
                     encoding='latin1')
    print(scores.shape)

    file_names = np.load(solution_path + f"{character}_file_names.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(file_names.shape)
    show_characters(character, detections, scores, file_names, params)
