import numpy as np
import cv2

from ex1_utils import IMG_INT_MAX_VAL, LOAD_GRAY_SCALE, LOAD_RGB

gamma_slider_max = 100
gamma_range = 10
title_window = 'Gamma Correction'


def on_trackbar(val):
    gamma_f = gamma_range * (val / gamma_slider_max)
    print("\rGamma {}".format(gamma_f), end='')
    dst = ((img / IMG_INT_MAX_VAL) ** (gamma_f) * IMG_INT_MAX_VAL).astype(np.uint8)
    cv2.imshow(title_window, dst)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, title_window, gamma_slider_max // gamma_range, gamma_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(gamma_slider_max // gamma_range)
    # Wait until user press some key
    cv2.waitKey()


if __name__ == '__main__':
    gammaDisplay('beach.jpg', LOAD_RGB)
