import os

import numpy as np
import cv2 as cv
from GameTable import GameTable
from TableDetector import WIDTH
from display_image_methods import show_image

TABLE_SIZE = WIDTH
PIECE_WIDTH = 102
SAFE_MARGIN = 2


class DominoDetector:
    def __init__(self, table, empty, game_table=GameTable()):
        self.table = table
        self.game = game_table
        self.empty_table = empty

    def cut_possible_pieces(self, verbose=False):
        added_positions = []
        table = self.table.copy()
        hsv = cv.cvtColor(table, cv.COLOR_BGR2HSV)
        low = (55, 0, 215)
        high = (255, 112, 255)

        mask = cv.inRange(hsv, low, high)
        # show_image("maks", mask)

        last_img = self.empty_table.copy()
        # show_image("last", last_img)
        hsv2 = cv.cvtColor(last_img, cv.COLOR_BGR2HSV)
        mask2 = cv.inRange(hsv2, low, high)
        # show_image("mask2", mask2)
        if verbose:
            show_image("dif", mask - mask2)

        mask = mask - mask2

        image_m_blur = cv.medianBlur(mask, 5)  # filtru blur median
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)  # filtru blur gaussian
        image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8,
                                         0)
        if verbose:
            show_image("sharpened", image_sharpened)

        kernel = np.ones((3, 3), np.uint8)  # dim filtru
        eroded = cv.erode(image_sharpened, kernel)  # erosion 
        if verbose:
            show_image("eroded", eroded)


        if verbose: print(table.shape)
        if verbose: print(len([x for x in range(5, TABLE_SIZE - 20, PIECE_WIDTH)]))
        x = -1

        for line in range(0, TABLE_SIZE - 1, PIECE_WIDTH):
            x += 1
            y = -1
            for column in range(0, TABLE_SIZE - 1, PIECE_WIDTH):
                y += 1
                margin = 1
                margin2 = 1
                piece = table[max(0, line - margin): line + PIECE_WIDTH + margin2,
                        max(0, column - margin):column + PIECE_WIDTH + margin2]
                piece_empty = self.empty_table[max(0, line - margin): line + PIECE_WIDTH + margin2,
                              max(0, column - margin):column + PIECE_WIDTH + margin2]
                dif = np.abs(np.mean(piece[:, :, 2]) - np.mean(piece_empty[:, :, 2]))
                piece_mask = mask[max(0, line - margin): line + PIECE_WIDTH + margin2,
                             max(0, column - margin):column + PIECE_WIDTH + margin2]
                blue_mean = np.mean(piece[:, :, 0])
                mean = np.mean(piece_mask)
                if x == 7 and y == 8:
                    if verbose: print(f"mean: {mean}")
                if verbose: print(x, y)
                coef = (blue_mean - 170) / 170 * 0.3 + (dif - 25) / 25 * 0.3 + (mean - 45) / 45 * 0.4

                if verbose:
                    print(coef)
                if verbose:
                    print(blue_mean)
                if verbose:
                    print(dif)

                if verbose: print(mean)

                if coef > 0:
                    # check if not write
                    if self.game.game_table[x][y] is not None:
                        continue
                    if 6 <= x <= 7 and 6 <= y <= 8 and not (x == 7 and y == 7):
                        # could be write
                        if mean < 100:
                            gray_piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
                            if self.is_matching_write(gray_piece=gray_piece, poz=y + 1):
                                continue

                    self.game.game_table[x][y] = DominoDetector.detect_piece(piece, verbose)
                    if True:
                        # print(f"Cu template matching: {self.template_matching(piece)}")
                        print(f"Cu Hough: {self.game.game_table[x][y]}")

                        if verbose: show_image("pm", piece_mask)
                    added_positions.append((x, y))

                else:
                    pass

        if len(added_positions) == 2:
            piece_highlight = np.array((TABLE_SIZE, TABLE_SIZE), dtype='uint8')
            piece_highlight = np.full((TABLE_SIZE, TABLE_SIZE), 0, dtype='uint8')
            piece_added = []
            for pos in added_positions:
              
                margin = 15

                h0 = max(pos[1] * PIECE_WIDTH - margin, 0)
                h1 = min((pos[1] + 1) * PIECE_WIDTH + margin, TABLE_SIZE)
                w0 = max(pos[0] * PIECE_WIDTH - margin, 0)
                w1 = min((pos[0] + 1) * PIECE_WIDTH + margin, TABLE_SIZE)
                piece_highlight[w0: w1, h0: h1] = \
                    np.full((w1 - w0, h1 - h0), 255, dtype='uint8')

                piece_added.append(self.game.game_table[pos[0]][pos[1]])

        else:
            print("Warning: Detected less/more pieces")

        return added_positions

    def is_matching_write(self, gray_piece, poz=7):
        templ = cv.imread(f"data/templates/{poz}.jpg")
        templ = cv.cvtColor(templ, cv.COLOR_BGR2GRAY)
        corr = cv.matchTemplate(gray_piece, templ, cv.TM_CCOEFF_NORMED)
        print(corr)
        return np.max(corr) > 0.25

    @staticmethod
    def detect_piece(halfpiece, verbose=False):
        if verbose:
            show_image("bl", halfpiece, False)
        minDist = 7
        param1 = 50
        param2 = 20
        minRadius = 9
        maxRadius = 14
        gray = cv.cvtColor(halfpiece, cv.COLOR_BGR2GRAY)

        blurred = cv.bilateralFilter(gray, 5, 50, 50)
        if verbose:
            show_image("bl", blurred, False)

        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                  minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                if verbose:
                    cv.circle(halfpiece, center, 1, (255, 255, 255), 2)
                radius = i[2]
                if verbose:
                    cv.circle(halfpiece, center, radius, (0, 0, 255), 2)
                # print("circle radius", radius)

        if verbose: cv.imshow("detected circles", halfpiece)
        if verbose: cv.waitKey(0)
        if verbose: print("cercuri detetctate:", len(circles[0]) if circles is not None else 0)
        return len(circles[0]) if circles is not None else 0

    @staticmethod
    def detect_one_piece(piece_photo):
        # x0, y0, width, height = cv.selectROI(np.uint8(piece_photo))
        # piece = piece_photo[y0:  y0+height, x0:x0+width]
        piece = piece_photo.copy()
        show_image("piece", piece)
        hsv_1 = cv.cvtColor(piece_photo, cv.COLOR_BGR2HSV)
        low_1 = (0, 0, 0)
        high_1 = (190, 255, 255)
        mask_1 = cv.inRange(hsv_1, low_1, high_1)
        blurr = cv.medianBlur(piece, 7)
        show_image("blurr", blurr, False)
        blurr = cv.cvtColor(blurr, cv.COLOR_BGR2GRAY)

        thresh_1 = cv.adaptiveThreshold(blurr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        show_image("thresh", thresh_1)

        # dilate
        kernel00 = np.ones((3, 3), np.uint8)
        dil = cv.dilate(thresh_1, kernel00)
        show_image("dilated", dil)

        # erode
        eroded_1 = cv.erode(dil, (3, 3))
        show_image("eroded", eroded_1)

        edges = cv.Canny(eroded_1, 25, 100)
        show_image("edges", edges, False)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) > 3]
        print(f"found : {len(contours)} contours")
        k = 0

        for c in contours:

            x, y, w, h = cv.boundingRect(c)
            if 9 < w < 25 and 50 < h < 100 or 9 < h < 25 and 50 < w < 100:
                # if 90 < w < 110 and 190 < h < 210 or 90 < h < 110 and 190 < w < 210 and 22000 > w * h > 15000:
                print(k + 1, "area : ", w * h)
                k += 1
                cv.rectangle(piece,
                             (x, y), (x + w, y + h),
                             (0, 255, 0),
                             3)
                print(f"curr area :", cv.contourArea(c))
                print(w, h)
                # cv.drawContours(piece_photo, [c], -1, (255, 0, 255), 3)
                show_image("contours", piece)

    def template_matching(self, piece):
        piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
        piece = cv.bilateralFilter(piece, 10, 50, 50)
        # show_image("piece", piece)

        piece = cv.adaptiveThreshold(piece, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY_INV, 11, 2)
        maxi = -np.inf
        poz = -1
        for j in range(0, 7):
            img_template = cv.imread('data/templates/' + str(j) + '.jpg')
            img_template = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
            img_template = cv.bilateralFilter(img_template, 10, 50, 50)

            img_template = cv.adaptiveThreshold(img_template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv.THRESH_BINARY_INV, 11, 2)

            for k in range(4):
                img_template = cv.rotate(img_template, cv.ROTATE_90_CLOCKWISE)

                for rot in [-10, -5, 0, 5, 10]:
                    rotated = self.rotate(img_template, rot)
                    # show_image("template ", img_template)
                    corr = cv.matchTemplate(rotated, img_template, cv.TM_CCOEFF_NORMED)
                    corr = np.max(corr)  # corelatie
                    print(j, corr)
                    if corr > maxi:
                        maxi = corr
                        poz = j
        return poz

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(image, M, (w, h))

        return rotated
