import os
import time

import cv2 as cv

from DominoDetector import DominoDetector
from GameTable import GameTable
from TableDetector import TableDetector
from display_image_methods import show_image

table = cv.imread("data/imagini_auxiliare/01.jpg")
img_0 = TableDetector(table)

img_0.detect_bigger_table()
img_0.table_extraction()
print(img_0.rectangle)

rect = img_0.rectangle
last_img = img_0.table_image
last_pos = []

GAME_NUMBERS = [1, 2, 3, 4, 5]
JPG_FORMAT = ".jpg"
TXT_FORMAT = ".txt"
MUTARI = "mutari"


def run_double_double_domino_detector(path_test, path_write_folder=None, write=False, games=None):
    global rect, last_img
    if games is None:
        games = GAME_NUMBERS
    if path_write_folder is None:
        path_write_folder = path_test

    for game in games:
        print(f"Game {game} starts")
        file_mutari = os.path.join(path_test, f"{game}_{MUTARI}{TXT_FORMAT}")
        last_img = img_0.table_image # setting last table for every game to be empty table at start
        with open(file_mutari) as mutari:
            moves = mutari.readlines()
            moves = [x.split() for x in moves]
        print(len(moves), " <- should be 20")
        game_table = GameTable()
        k = 0
        for move in moves[:20]:  # should assert len(moves) == 20
            img, player = move
            print(img[:-4])
            print(f"{player} moves")
            image = cv.imread(os.path.join(path_test, img))

            detect = TableDetector(image)
            detect.detect_bigger_table(False)

            detect.detect_corners(corners=rect, show_img=False)
            k += 1
            game_table.current_player = player  # set current player
            domino_detect = DominoDetector(detect.table_image, last_img, game_table=game_table)

            pos = domino_detect.cut_possible_pieces()

            print(len(pos), " poz detectate")

            last_img = detect.table_image # setting last table 

            print(game_table.get_output_current_move(added_positions=pos, write=write,
                                                     file_to_write=os.path.join(path_write_folder,
                                                                                img[:-4] + TXT_FORMAT)))
            # time.sleep(0.5) # sometimes python does not manage to write to files and sleep is needed!


if __name__ == "__main__":
    run_double_double_domino_detector(
        path_test='data/test',  # reads move files and img from here
        path_write_folder='fisiere_solutie',  # writes the results in this folder-> if None => path_write_folder = path_test
        write=True,  # write results, default False
        games=[1])  # if None => games = [1, 2, 3, 4, 5]
