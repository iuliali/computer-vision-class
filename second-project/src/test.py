import numpy as np

from evalueaza_solutie import eval_detections
from FacialDetector import FacialDetector
from Parameters import Parameters

params = Parameters()

def test_evalueaza_solutie():
    solution_path_root = "../results/"
    ground_truth_path_root = "../data/test/"
    #
    # task1
    solution_path = solution_path_root
    ground_truth_path = ground_truth_path_root + "task1_gt_validare.txt"
    # print(solution_path)
    detections = np.load(solution_path + "detections_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    # print(detections.shape)

    scores = np.load(solution_path + "scores_all_faces.npy", allow_pickle=True, fix_imports=True, encoding='latin1')
    # print(scores.shape)

    file_names = np.load(solution_path + "file_names_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    # print(file_names.shape)

    eval_detections(detections, scores, file_names, ground_truth_path)

    detections, scores, file_names = FacialDetector.filter(detections, scores, file_names, params)

    print("dupa filtrare cu resnet")

    eval_detections(detections, scores, file_names, ground_truth_path)


test_evalueaza_solutie()


