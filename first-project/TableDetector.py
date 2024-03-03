import numpy as np
import cv2 as cv
from numpy.array_api import int32

from display_image_methods import show_image

WIDTH = 1530
HEIGHT = 1530
SAFE_MARGIN = 15


class TableDetector:
    def __init__(self, img):
        self.table_image = img
        self.rectangle = []


    def table_extraction(self, show_img=False):
        if show_img: show_image("img", self.table_image)

        low = (60, 150, 40)
        high = (255, 255, 255)

        img_hsv = cv.cvtColor(self.table_image, cv.COLOR_BGR2HSV)
        l = np.array(low)
        u = np.array(high)
        mask_table_hsv = cv.inRange(img_hsv, l, u)

        if show_img:show_image('image_processed', mask_table_hsv)
        image_m_blur = cv.medianBlur(mask_table_hsv, 21)  # filtru blur median
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 21)  # filtru blur gaussian
        image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8,
                                         0)  # combinare filtre
        if show_img:show_image('image_sharpened', image_sharpened)
        _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)  # dim filtru
        thresh = cv.erode(thresh, kernel)  # erosion 

        if show_img:show_image('image_thresholded', thresh)

        edges = cv.Canny(thresh, 200, 400)
        if show_img:show_image('edges', edges)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  
        max_area = 0
        top_left, bottom_left, top_right, bottom_right = 0, 0, 0, 0

        for i in range(len(contours)):
            if len(contours[i]) > 3:
                # print(cv.contourArea(contours[i]))
                # cv.drawContours(edges, contours=[contours[i]], contourIdx=-1, color=(255, 255, 0), thickness=2)

                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                            possible_bottom_right[1]:
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis=1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                            [possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array(
                        [[possible_top_left], [possible_top_right], [possible_bottom_right],
                         [possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left

        rect = (top_left, top_right, bottom_right, bottom_left)
        MARGIN = 7

        # print(rect)
        rect = (np.array((rect[0][0] - MARGIN, rect[0][1] - MARGIN), dtype=int32),
                np.array((rect[1][0] + MARGIN, rect[1][1] - MARGIN), dtype=int32),
                np.array((rect[2][0] + MARGIN, rect[2][1] + MARGIN), dtype=int32),
                np.array((rect[3][0] - MARGIN, rect[3][1] + MARGIN), dtype=int32)
                )
        # print(rect)
        self.rectangle = rect
        self.detect_corners(corners=rect, show_img=show_img)
        return self.table_image

    def detect_corners(self, corners, show_img=False):
        image_copy = self.table_image.copy()
        top_left, top_right, bottom_right, bottom_left = corners
        cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
        if show_img:
            show_image("detected corners", image_copy)
        puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        destination_puzzle = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype="float32")
        M = cv.getPerspectiveTransform(puzzle, destination_puzzle)
        result = cv.warpPerspective(self.table_image, M, (WIDTH, HEIGHT))
        self.table_image = result
        if show_img: show_image("table", self.table_image)
        return result


    def detect_bigger_table(self, show_img=False):
        if show_img: show_image("img", self.table_image)
        h, w, _ = self.table_image.shape
        percent_10 = int(0.1 * h)
        self.table_image = self.table_image[percent_10: h - percent_10, :]
        if show_img: show_image("img", self.table_image)

        low = (0, 0, 0)
        high = (45, 255, 255)
        img_hsv = cv.cvtColor(self.table_image, cv.COLOR_BGR2HSV)
        l = np.array(low)
        u = np.array(high)
        mask_table_hsv = cv.inRange(img_hsv, l, u)


        image_m_blur = cv.medianBlur(mask_table_hsv, 21)  # filtru blur median
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 21)  # filtru blur gaussian
        image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8,
                                         0)  # combinare filtre
        if show_img: show_image('image_sharpened', image_sharpened)

        _, thresh = cv.threshold(image_sharpened, 100, 255, cv.THRESH_BINARY_INV)
        if show_img: show_image('image_thresholded', thresh)

        # threshold
        kernel = np.ones((15, 15), np.uint8)  # dim filtru
        thresh = cv.erode(thresh, kernel)  # erosion 
        if show_img: show_image('eroded', thresh)

        # threshold
        kernel = np.ones((5, 5), np.uint8)  # dim filtru
        thresh = cv.dilate(thresh, kernel)  # dilation
        if show_img: show_image('dilated', thresh)

        edges = cv.Canny(thresh, 200, 400)
        if show_img: show_image('edges', edges)

        contours, _ = cv.findContours(edges
                                      , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # fct f smechera!!!
        max_area = 0
        top_left, bottom_left, top_right, bottom_right = 0, 0, 0, 0
        print(len(contours))
        for i in range(len(contours)):
            if len(contours[i]) > 3:
                x, y, w, h = cv.boundingRect(contours[i])
                # print( f"are contour: {cv.contourArea(contours[i])}    area rect: {w*h}  w:{w}  h: {h}")
                # if show_img:
                #     cv.drawContours(self.table_image, [contours[i]], contourIdx=-1,  color=(255, 0, 0), thickness=3)
                #     show_image("contour", self.table_image)

                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                            possible_bottom_right[1]:
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis=1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if 550 * 10 ** 4 < cv.contourArea(
                        np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                  [possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array(
                        [[possible_top_left], [possible_top_right], [possible_bottom_right],
                         [possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left
        # width = top_right[0]-top_left[0]
        # bottom_left = (top_left[0], top_left[1] + width)
        # bottom_right = (top_right[0], top_right[1] + width)
        self.rectangle = (top_left, top_right, bottom_right, bottom_left)
        return self.detect_corners(corners=(top_left, top_right, bottom_right, bottom_left), show_img=show_img)
