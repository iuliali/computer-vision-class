import cv2 as cv


def show_image(img, title='Image'):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
