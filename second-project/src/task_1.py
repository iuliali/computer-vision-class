import os
import pickle

import numpy as np

from FacialDetector import FacialDetector
from Parameters import Parameters
from visualise import show_detections_with_ground_truth, show_detections_without_ground_truth


def task_1(results_dir_task_1=os.path.join("..", "results/task1/"), verbose=False):
    # direct incarca modelul de svm si resnet si ruleaza detectia
    params = Parameters()
    facial_detector = FacialDetector(params)
    facial_detector.best_model = pickle.load(open(os.path.join("../models", "model_svm_1"), "rb"))
    detections, scores, file_names = facial_detector.run()

    # save for running task 2
    RESULTS_DIR = results_dir_task_1
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print('directory created: {} '.format(RESULTS_DIR))
    before_resnet = "before_resnet"
    np.save(os.path.join(RESULTS_DIR, f"detections_all_faces_{before_resnet}.npy"), detections)
    np.save(os.path.join(RESULTS_DIR, f"file_names_all_faces_{before_resnet}.npy"), file_names)
    np.save(os.path.join(RESULTS_DIR, f"scores_all_faces_{before_resnet}.npy"), scores)

    if verbose:
        show_detections_without_ground_truth(detections, scores, file_names, params)

    # take detections from svm classifier and run them through resnet
    print("Filtering using resnet18...")
    detections, scores, file_names = facial_detector.filter(detections, scores, file_names, params)

    # save detections, scores and file_names
    np.save(os.path.join(RESULTS_DIR, "detections_all_faces.npy"), detections)
    np.save(os.path.join(RESULTS_DIR, "file_names_all_faces.npy"), file_names)
    np.save(os.path.join(RESULTS_DIR, "scores_all_faces.npy"), scores)
    if verbose:
        show_detections_without_ground_truth(detections, scores, file_names, params)

if __name__ == "__main__":
    task_1(verbose=True)
