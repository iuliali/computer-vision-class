import os
from constants import *


class Parameters:
    def __init__(self):
        # -----------------  directory paths ----------------
        self.base_directory = DATA_DIR
        self.directory_positive_examples = os.path.join(self.base_directory, TRAINING_DIR, POSITIVES)
        self.directory_negative_examples = os.path.join(self.base_directory, TRAINING_DIR, NEGATIVES)
        self.directory_test = os.path.join(self.base_directory, TEST_DIR)
        self.path_test_annotations = os.path.join(self.base_directory, TEST_DIR, f"validare_{ANNOTATIONS}{TXT_FORMAT}")
        self.directory_saved_files = os.path.join(self.base_directory, SAVED_FILES)
        if not os.path.exists(self.directory_saved_files):
            os.makedirs(self.directory_saved_files)
            print('directory created: {} '.format(self.directory_saved_files))
        else:
            print('directory {} exists '.format(self.directory_saved_files))

        self.barney_folder = os.path.join(ANTRENARE_DIR, BARNEY)
        self.betty_folder = os.path.join(ANTRENARE_DIR, BETTY)
        self.fred_folder = os.path.join(ANTRENARE_DIR, FRED)
        self.wilma_folder = os.path.join(ANTRENARE_DIR, WILMA)

        # ------------- facial-detector params --------------
        self.use_flip_images = True
        self.use_hard_mining = False
        self.window = 48
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 48  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 1608  # numarul exemplelor pozitive (not real)
        self.number_negative_examples = 15354  # numarul exemplelor negative (not real)
        self.has_annotations = False
        self.threshold = 3
        self.orientations = 12
        self.block_norm = 'L2'
        self.cells_per_block = (2, 2)
        self.resize_scales = [x / 100 for x in range(5, 30, 3)] + [x / 100 for x in range(20, 150, 5)]
        self.Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
        self.aspect_ratios = [0.6, 0.8, 0.9, 1, 1.1, 1.2, 1.3]

