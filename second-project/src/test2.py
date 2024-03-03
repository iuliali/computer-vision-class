import os

import numpy as np

from FacialDetector import FacialDetector
from Parameters import Parameters
from visualise import show_detections_with_ground_truth, show_detections_without_ground_truth

RESULTS_DIR = os.path.join("..", "results-1/task1/")
params = Parameters()
facial_detector = FacialDetector(params)

task1_detections = np.load("../results-1/task1/detections_all_faces_before_resnet.npy")
task1_file_names = np.load("../results-1/task1/file_names_all_faces_before_resnet.npy")
task1_scores = np.load("../results-1/task1/scores_all_faces_before_resnet.npy")

task1_detections, task1_scores, task1_file_names = facial_detector.filter(task1_detections, task1_scores,
                                                                          task1_file_names, params)

facial_detector.eval_detections(task1_detections, task1_scores, task1_file_names)
show_detections_without_ground_truth(task1_detections, task1_scores, task1_file_names, params)

np.save(os.path.join(RESULTS_DIR, "detections_all_faces.npy"), task1_detections)
np.save(os.path.join(RESULTS_DIR, "file_names_all_faces.npy"), task1_file_names)
np.save(os.path.join(RESULTS_DIR, "scores_all_faces.npy"), task1_scores)
