from Rectangle import Rectangle
from constants import *


def annotation(character_name):
    return f"{character_name}_{ANNOTATIONS}{TXT_FORMAT}"


def annotations_reader(path_annotations) -> dict:
    annotations = {}
    with open(path_annotations, "r") as file_annotations:
        lines = file_annotations.readlines()
    lines = [line.split() for line in lines]  # splitted now
    for line in lines:
        image_name = line[0]
        if image_name in annotations:
            annotations[image_name].append((Rectangle(line[1], line[2], line[3], line[4]), line[5]))
        else:
            annotations[image_name] = [(Rectangle(line[1], line[2], line[3], line[4]), line[5])]
    return annotations


def get_coords_from_annotations(annotations: dict, image_name: str):
    assert image_name in annotations, f"get_coords_from_annotations: Image name gived not found in annotations!"
    coords_with_character = annotations[image_name]
    return coords_with_character
