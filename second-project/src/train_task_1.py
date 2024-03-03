import os
import time

import numpy as np

from FacialDetector import FacialDetector
from Parameters import Parameters
from constants import TRAINING_DIR, POSITIVES
from visualise import show_detections_with_ground_truth, show_detections_without_ground_truth

params: Parameters = Parameters()

params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente

POSITIVE_FEATURE_PATH = os.path.join(params.directory_saved_files,
                                     'descriptors_positives' + str(params.dim_hog_cell) + '_' +
                                     str(params.number_positive_examples) + '.npy')
if os.path.exists(POSITIVE_FEATURE_PATH):
    positive_features = np.load(POSITIVE_FEATURE_PATH)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    params.directory_positive_examples = os.path.join(params.base_directory, TRAINING_DIR, POSITIVES)
    positive_features = facial_detector.get_positive_descriptors()
    params.number_positive_examples = positive_features.shape[0]
    print("positives", positive_features.shape[0])
    np.save(POSITIVE_FEATURE_PATH, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % POSITIVE_FEATURE_PATH)

# exemple negative
negative_features_path = os.path.join(params.directory_saved_files,
                                      'descriptors_negatives' + str(params.dim_hog_cell) + '_' +
                                      str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    params.number_negative_examples = negative_features.shape[0]
    print("negatives", negative_features.shape[0])
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 4. Invatam clasificatorul liniar

pos_character = positive_features
np.random.shuffle(negative_features)
neg_character = negative_features[:int(pos_character.shape[0] * 1.2)]
train_labels = np.concatenate((np.ones(pos_character.shape[0]), np.zeros(neg_character.shape[0])))
training_examples = np.concatenate((np.squeeze(pos_character),
                                    np.squeeze(neg_character)), axis=0)
facial_detector.train_classifier(training_examples, train_labels)
# scores = facial_detector.best_model.decision_function(training_examples)
# strong_negatives = training_examples[len(pos_character):][scores[len(pos_character):] > 0]

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare
# if params.use_hard_mining:
#     print("ANTRENARE CU EXEMPLE PUTERNIC NEGATIVE")
#     training_examples_hard_mining = np.concatenate((np.squeeze(training_examples), np.squeeze(strong_negatives)), axis=0)
#     train_labels_hard_mining = np.concatenate((train_labels, np.zeros(strong_negatives.shape[0])))
#     facial_detector.train_classifier(training_examples_hard_mining, train_labels_hard_mining)


detections, scores, file_names = facial_detector.run()


# save for running task 2
RESULTS_DIR = os.path.join("..", "results-1/task1/")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print('directory created: {} '.format(RESULTS_DIR))
before_resnet = "before_resnet"
np.save(os.path.join(RESULTS_DIR, f"detections_all_faces_{before_resnet}.npy"), detections)
np.save(os.path.join(RESULTS_DIR, f"file_names_all_faces_{before_resnet}.npy"), file_names)
np.save(os.path.join(RESULTS_DIR, f"scores_all_faces_{before_resnet}.npy"), scores)

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    # show_detections_with_ground_truth(detections, scores, file_names, params)

time.sleep(10)

print("Filtering..")
# fiter detections
detections, scores, file_names = facial_detector.filter(detections, scores, file_names, params)

# save detections, scores and file_names

RESULTS_DIR = os.path.join("..", "results-1/task1/")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print('directory created: {} '.format(RESULTS_DIR))

np.save(os.path.join(RESULTS_DIR, "detections_all_faces.npy"), detections)
np.save(os.path.join(RESULTS_DIR, "file_names_all_faces.npy"), file_names)
np.save(os.path.join(RESULTS_DIR, "scores_all_faces.npy"), scores)

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
