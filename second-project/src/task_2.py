import numpy as np

from FacialRecognizer import FacialRecognizer
from Parameters import Parameters
from constants import Character

def task_2(path_to_save="../results/task2/"):
    task1_detections = np.load("../results/task1/detections_all_faces_before_resnet.npy")
    task1_file_names = np.load("../results/task1/file_names_all_faces_before_resnet.npy")

    params = Parameters()
    facial_recognizer = FacialRecognizer(   path_images_test=params.directory_test,
                                            path_recognizers="../models",
                                            characters=None)
    for character in Character:
        facial_recognizer.predict_and_save(character=character, task1_file_names=task1_file_names,
                                       task1_detections=task1_detections, path_to_save=path_to_save)

if __name__ == "__main__":
    task_2()