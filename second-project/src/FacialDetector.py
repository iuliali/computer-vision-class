import glob
import ntpath
import os
import pickle
import timeit
from copy import deepcopy

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet18

from Parameters import Parameters
from Rectangle import Rectangle


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None
        self.scores = None

    def filter_detections(self, detections, scores, dif=0.4):
        filtered_detections = []
        filtered_scores = []
        for i in range(len(detections)):
            to_remove = False
            rect_i = Rectangle(*detections[i])
            for j in range(len(detections)):
                if i != j:
                    rect_j = Rectangle(*detections[j])
                    if rect_i.is_included_in(rect_j) or rect_j.is_included_in(rect_i):
                        if scores[i] < scores[j] and abs(scores[i] - scores[j]) > dif: # marcam 'de eliminat' pe cel cu scor mai mic
                            to_remove = True
                            break
            if not to_remove:
                filtered_detections.append(detections[i])
                filtered_scores.append(scores[i])
        return filtered_detections, filtered_scores

    def get_hog_features(self, img, flip=False, feature_vector=True, resize=True):
        if resize:
            resized = cv.resize(img, (self.params.window, self.params.window))
        else:
            resized = img
        if flip:
            resized = np.fliplr(resized)
        return hog(resized, orientations=self.params.orientations, block_norm=self.params.block_norm,
                   pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                   cells_per_block=self.params.cells_per_block, feature_vector=feature_vector)


    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.directory_positive_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # print(img.shape)
            features = self.get_hog_features(img, feature_vector=True)
            print("D= ", len(features))

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = self.get_hog_features(img, flip=True, feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor neg
        # iar D - dimensiunea descriptorului
        # D = (params.window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.directory_negative_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negative numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            print(img.shape)
            features = self.get_hog_features(img, feature_vector=True)
            print("D= ", len(features))

            negative_descriptors.append(features)
        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels) -> np.array:
        svm_file_name = os.path.join(self.params.directory_saved_files, f'best_model_%d_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples, self.params.orientations))
        if not self.params.use_hard_mining:
            if os.path.exists(svm_file_name):
                self.best_model = pickle.load(open(svm_file_name, 'rb'))
                return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = self.params.Cs
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        self.scores = best_model.decision_function(training_examples)
        self.best_model = best_model

        positive_scores = self.scores[train_labels > 0]
        negative_scores = self.scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        # print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def get_scaled_image(self, original, scale, ratio):
        if ratio < 1:
            w_scale = ratio * scale
            h_scale = scale
        else:
            w_scale = scale
            h_scale = (1 / ratio) * scale

        # Resize the image using bilinear interpolation
        return cv.resize(original, None, fx=w_scale, fy=h_scale, interpolation=cv.INTER_LINEAR), w_scale, h_scale

    def run(self, show_every_image=False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.directory_test, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array(
            [])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        num_test_images = len(test_files)
        resize_scales = self.params.resize_scales
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i + 1, num_test_images))
            original = cv.imread(test_files[i], cv.IMREAD_COLOR)
            image_scores = []
            image_detections = []

            maximum_min = self.params.window / original.shape[0]
            resize_scales = [scale for scale in resize_scales if scale > maximum_min]
            # print(resize_scales)
            for scale in resize_scales:
                for ratio in self.params.aspect_ratios:
                    img, w_scale, h_scale = self.get_scaled_image(original, scale, ratio)

                    if img.shape[0] < self.params.window or img.shape[1] < self.params.window:
                        continue


                    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    # canny = cv.bilateralFilter(gray_img, 9, 75, 75)
                    # canny = cv.Canny(canny, 100, 200)
                    hog_descriptors = self.get_hog_features(gray_img, feature_vector=False, resize=False)

                    num_cols = gray_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = gray_img.shape[0] // self.params.dim_hog_cell - 1

                    num_cell_in_template = self.params.window // self.params.dim_hog_cell - 1
                    if num_cols < 1 or num_rows < 1:
                        continue

                    for y in range(0, num_rows - num_cell_in_template):
                        for x in range(0, num_cols - num_cell_in_template):

                            descr = hog_descriptors[y:y + num_cell_in_template,
                                    x:x + num_cell_in_template].flatten()
                            score = np.dot(descr, w)[0] + bias

                            if score > self.params.threshold:
                                # print("SCORE HOG: {}".format(score))

                                x_min = int((x // w_scale) * self.params.dim_hog_cell)
                                y_min = int((y // h_scale) * self.params.dim_hog_cell)
                                x_max = int((x * self.params.dim_hog_cell + self.params.window) // w_scale)
                                y_max = int((y * self.params.dim_hog_cell + self.params.window) // h_scale)

                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

                                if show_every_image:
                                    cv.rectangle(original,
                                                 (x_min, y_min), (x_max, y_max),
                                                 (255, 255, 0), thickness=2)
                                    cv.putText(original, 'score:' + str(score)[:4], (x_min, y_min),
                                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    cv.imshow('image', np.uint8(original))
                                    cv.waitKey(0)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores), original.shape,
                                                                              )
                image_detections, image_scores = self.filter_detections(image_detections, image_scores, 0.4)
            if len(image_scores) > 0:
                # for g in range(len(image_detections)):
                #     detection = image_detections[g]
                #     score = image_scores[g]
                #     cv.rectangle(original,
                #                  (x_min, y_min), (x_max, y_max),
                #                  (255, 255, 0), thickness=2)
                #     cv.putText(original, 'score:' + str(score)[:4], (x_min, y_min),
                #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                #     cv.imshow('image', np.uint8(original))
                #     cv.waitKey(0)

                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_test_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1: -1], np.int32)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= self.params.overlap:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.directory_saved_files, f'precizie_medie_{average_precision}.png'))
        plt.show()

    @staticmethod
    def transform_for_resnet(img, box):
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        x_min, y_min, x_max, y_max = box
        img = img[y_min:y_max, x_min:x_max]
        img_test = Image.fromarray(img)
        t_img = Variable(normalize(to_tensor(scaler(img_test)))).unsqueeze(0)
        return t_img

    @staticmethod
    def filter(detections, scores, file_names, params: Parameters):
        device = torch.device('cpu')
        model = resnet18(num_classes=2)
        model.load_state_dict(torch.load("../models/model_dict_task_1_1", map_location=device))
        model.to(device)
        model.eval()
        new_detections = []
        new_scores = []
        new_file_names = []
        for i in range(len(detections)):
            image = cv.imread(os.path.join(params.directory_test, file_names[i]), cv.IMREAD_COLOR)
            # print(os.path.join(params.directory_test, file_names[i]))
            img = FacialDetector.transform_for_resnet(image, detections[i])
            predictions = model(img).detach().numpy()[0]
            label = np.argmax(predictions)
            score = predictions[label]  # new score saved, score given by resnet !!
            if label == 1:
                # show_image(image[detections[i][1]:detections[i][3], detections[i][0]:detections[i][2]])
                new_detections.append(detections[i])
                new_scores.append(score)
                new_file_names.append(file_names[i])

        return np.array(new_detections), np.array(new_scores), np.array(new_file_names)
