## characters' names
import enum

BARNEY = "barney"
BETTY = "betty"
FRED = "fred"
WILMA = "wilma"


class Character(enum.Enum):
    BARNEY = 0,
    BETTY = 1,
    FRED = 2,
    WILMA = 3

    def __int__(self):
        return int(self.value[0]) if type(self.value) == tuple else int(self.value)

    def __str__(self):
        return character_to_string(self)


## various file names and formats
ANNOTATIONS = "annotations"
TXT_FORMAT = ".txt"
JPG_FORMAT = ".jpg"
PTH_FORMAT = ".PTH"

## dirs
DATA_DIR = "../data"
TRAINING_DIR = "training"
TEST_DIR = "test"
SAVED_FILES = "saved_files"

ANTRENARE_DIR = "../data/antrenare"

## dimensions
DIM_IMAGE = (480, 360)
W_IMAGE = DIM_IMAGE[0]
H_IMAGE = DIM_IMAGE[1]

## facial-detector
ASPECT_RATIO = 0.8  # empiric
BIGGER_SIZE = 40
SMALLER_SIZE = 32

NEGATIVES = "negatives"
POSITIVES = "positives"

REVERSE_MAPPING = {
    0: BARNEY,
    1: BETTY,
    2: FRED,
    3: WILMA,
    4: "unknown"
}


def character_to_string(character_from_enum: Character) -> str:
    if character_from_enum == Character.BARNEY:
        return "barney"
    elif character_from_enum == Character.BETTY:
        return "betty"
    elif character_from_enum == Character.FRED:
        return "fred"
    elif character_from_enum == Character.WILMA:
        return "wilma"
    else:
        return "unknown"
