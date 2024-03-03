import os

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.models import resnet18

from constants import Character
from mappings import mapping


class FacialRecognizer:
    def __init__(self, path_images_test="..", path_recognizers="..", characters=None):
        if characters is None:
            characters = [Character.BARNEY, Character.BETTY, Character.FRED, Character.WILMA]
        self.recognizer = {character: None for character in Character}
        self.path_images = path_images_test
        self.path_recognizer = path_recognizers
        self.characters = characters
        self.load_recognizers(path_recognizers)

    def load_recognizers(self, path_recognizers, device=torch.device('cpu')):
        def get_path_to_recognizer(character):
            return os.path.join(path_recognizers, f"model_dict_{int(character)}_2")

        for character in self.characters:
            model = resnet18(num_classes=2)
            model.load_state_dict(torch.load(get_path_to_recognizer(character),
                                             map_location=device))
            self.recognizer[character] = model
            model.to(device)
            model.eval()

    @staticmethod
    def transform_for_resnet(path_to_image, box):
        image = cv.imread(path_to_image, cv.IMREAD_COLOR)
        image = image[box[1]:box[3], box[0]:box[2]]
        image = Image.fromarray(image)
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        return Variable(normalize(to_tensor(scaler(image)))).unsqueeze(0)

    def predict(self, image, character):
        predictions = self.recognizer[character](image).detach().numpy()[0]
        label = np.argmax(predictions)
        # print(predictions, label)
        return predictions[label], label

    def predict_all(self, character, task1_file_names, task1_detections):
        indices_with_character = []  # selecteaza doar detectiile cu caracterul din outputul de la task 1
        scores = []
        for idx, file_name in enumerate(task1_file_names):
            path = os.path.join(self.path_images, file_name)
            image = self.transform_for_resnet(path, task1_detections[idx])
            score, prediction = self.predict(image, character)  # este caracterul din parametru sau nu
            if prediction == 1:
                indices_with_character.append(idx)
                scores.append(score)
        file_names = np.array(task1_file_names)[np.array(indices_with_character)]
        detections = np.array(task1_detections)[np.array(indices_with_character)]
        scores = np.array(scores)

        return file_names, detections, scores

    def predict_and_save(self, character, task1_file_names, task1_detections, path_to_save):
        file_names, detections, scores = self.predict_all(character, task1_file_names, task1_detections)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        print(file_names.shape, detections.shape, scores.shape)
        print(os.path.join(path_to_save, f"{mapping(character)}_scores.npy"))
        np.save(os.path.join(path_to_save, f"{mapping(character)}_scores.npy"), scores)
        np.save(os.path.join(path_to_save, f"{mapping(character)}_detections.npy"), detections)
        np.save(os.path.join(path_to_save, f"{mapping(character)}_file_names.npy"), file_names)
