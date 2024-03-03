import cv2 as cv


def show_image(title, image, resize=True):
    if resize:
        image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
