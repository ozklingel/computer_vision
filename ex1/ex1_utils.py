from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])
IMG_INT_MAX_VAL = 255


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 666  # The ID of a close friend


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename, -1)
    if representation is LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to range [0,1]
    img = img.astype(np.float) / IMG_INT_MAX_VAL

    if representation is LOAD_GRAY_SCALE \
            and len(img.shape) > 2:
        b, g, r = np.split(img, 3, axis=2)
        img = 0.3 * r + 0.59 * g + 0.11 * b
        img = img.squeeze()
    elif representation is LOAD_RGB \
            and len(img.shape) < 3:
        img = np.stack((img, img, img), axis=2)

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.gray()
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return imgRGB.dot(YIQ_MAT.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    rgb = imgYIQ.dot(np.linalg.inv(YIQ_MAT).T)

    # Values clipping
    rgb[rgb > 1] = 1
    rgb[rgb < 0] = 0
    return rgb


def calHist(img: np.ndarray, bins=256, hist_range: tuple = (0, 256)) -> np.ndarray:
    """
    Calculate the image histogram
    :param img: Input image
    :param bins: Number of bins to group the values
    :param hist_range: The range of values
    :return: An np.array of size (bins)
    """
    img_flat = img.ravel()
    min_val = hist_range[0]
    max_val = hist_range[1]
    hist = np.zeros(bins)

    for pix in img_flat:
        quan_val = int(bins * (pix - min_val) / (max_val - min_val))
        hist[quan_val] += 1

    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    """
    Calculate the Cumulitive Sum of an array
    :param arr: Input array
    :return: A CumSum array of size (len(arr))
    """
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]

    return cum_sum


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    gray = imgOrig
    yiq = None
    if len(imgOrig.shape) > 2:
        yiq = transformRGB2YIQ(imgOrig)
        gray = yiq[:, :, 0]

    gray = (gray * IMG_INT_MAX_VAL).astype(np.uint8)

    hist_org = calHist(gray)
    cumsum = calCumSum(hist_org)
    cumsum_n = cumsum / cumsum.max()

    LUT = (IMG_INT_MAX_VAL * cumsum_n).astype(int)

    eq_img = np.zeros_like(gray, dtype=np.float)
    for old_color, new_color in enumerate(LUT):
        eq_img[gray == old_color] = new_color

    hist_eq = calHist(eq_img)
    eq_img /= IMG_INT_MAX_VAL

    if yiq is not None:
        yiq[:, :, 0] = eq_img
        eq_img = transformYIQ2RGB(yiq)

    return eq_img, hist_org, hist_eq


def kmeans(pnts: np.ndarray, k: int, iter_num: int = 7) -> (np.ndarray, List[np.ndarray]):
    """
    Calculates K-Means
    :param pnts: The data, a [nXm] array, **n** samples of **m** dimentions
    :param k: Numbers of means
    :param iter_num: Number of K-Means iterations
    :return: [k,m] centers,
             List of the history of assignments, each assignment is a [n,1]
             array with assignment of each data point to center,
    """
    if len(pnts.shape) > 1:
        n, m = pnts.shape
    else:
        n = pnts.shape[0]
        m = 1
    pnts = pnts.reshape((n, m))
    assign_array = np.random.randint(0, k, n).astype(np.uint8)
    centers = np.zeros((k, m))

    assign_history = []
    for it in range(iter_num):
        # Find centers
        for i, _ in enumerate(centers):
            if sum(assign_array == i) < 1:
                continue
            centers[i] = pnts[assign_array == i, :].mean(axis=0)

        # Assign pixels to centers
        for i, p in enumerate(pnts):
            center_dist = np.power(centers - p, 2).sum(axis=1)
            assign_array[i] = np.argmin(center_dist)
        assign_history.append(assign_array.copy())

    return centers, assign_history


def quantizeImageKmeans(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of iterations for the KMeans
    :return: (List[qImage_i],List[error_i])
    """

    gray = imOrig / imOrig.max()
    if len(gray.shape) > 2:
        data = gray.reshape((-1, 3))
        height, width, _ = gray.shape
    else:
        data = gray.reshape(-1)

    cs, assignment = kmeans(data, nQuant, nIter)

    assign_list = []
    error_list = []
    for assign_i in assignment:
        qunt_img = np.zeros(data.shape)
        for i, c in enumerate(cs):
            qunt_img[assign_i == i] = c
        qunt_img = qunt_img.reshape(gray.shape)

        error = np.square((qunt_img - gray) * IMG_INT_MAX_VAL).mean()
        assign_list.append(qunt_img)
        error_list.append(error)

    return assign_list, error_list


def getWeightedMean(vals: np.ndarray, weights: np.ndarray) -> int:
    """
    Calculats the weighted mean of
    :param vals:
    :param weights:
    :return:
    """
    val = (vals * weights).sum() / (weights.sum() + np.finfo('float').eps)
    return val.astype(int)


def recolor(img: np.ndarray, borders: np.ndarray, weighted_mean: np.ndarray) -> np.ndarray:
    """
    Recolors the image according to the borders and the cells's means
    :param img: The original image
    :param borders: The borders
    :param weighted_mean: The cells's means
    :return: The recolored image
    """
    img_quant = np.zeros_like(img)
    for q in range(len(borders) - 1):
        left_b = borders[q]
        right_b = borders[q + 1]
        min_val_matrix = img >= left_b
        max_val_matrix = img < right_b
        img_quant[max_val_matrix * min_val_matrix] = weighted_mean[q]
    return img_quant


def dispBorders(hist: np.ndarray, borders: np.ndarray, qs: np.ndarray, err) -> None:
    """
    Displays the histogram with the borders and the cells's means
    :param hist: The histograms
    :param borders: The borders array
    :param qs: The cells's means array
    :param err: The error
    :return: None
    """
    plt.ion()
    max_val = hist.max()
    plt.bar(range(len(hist)), hist, color='b')
    plt.plot(qs, hist[qs.astype(int)], 'r*')
    for b in borders:
        plt.axvline(x=b, ymin=0, ymax=max_val, c='y')
    plt.title("Error: {}".format(err))
    plt.waitforbuttonpress()


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int, verbos=False) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    gray = imOrig
    if len(imOrig.shape) > 2:
        yiq = transformRGB2YIQ(imOrig)
        gray = yiq[:, :, 0]
        comp_img = imOrig
    else:
        comp_img = (gray * IMG_INT_MAX_VAL).astype(int)
    gray = (gray * IMG_INT_MAX_VAL).astype(int)

    hist = calHist(gray)
    cumsum = calCumSum(hist)
    cumsum /= cumsum[-1]

    # Initiating the borders
    borders = np.arange(gray.min(), gray.max(), gray.max() / nQuant, dtype=np.float)
    max_val = gray.max() + 1
    borders = np.hstack([borders, max_val])
    weighted_mean = np.array([borders[x:x + 2].mean() for x in range(len(borders[:-1]))])

    # Calculate histogram
    error_trk = []
    img_quant_trk = []
    # dispBorders(hist, borders, weighted_mean, 100)
    for it in range(nIter):
        # Find quants means
        for q in range(nQuant):
            left_b = int(np.floor(borders[q]))
            right_b = int(np.ceil(borders[q + 1]))
            q_vals = hist[left_b:right_b]
            weighted_mean[q] = getWeightedMean(np.arange(left_b, right_b), q_vals)

        # Make shift borders
        borders = np.zeros_like(borders)
        borders[-1] = max_val
        for q in range(1, nQuant):
            borders[q] = weighted_mean[q - 1:q + 1].sum() / 2

        # Recolor the image
        tmp = recolor(gray, borders, weighted_mean)
        if len(imOrig.shape) > 2:
            tmp_yiq = yiq.copy()
            tmp_yiq[:, :, 0] = tmp / IMG_INT_MAX_VAL
            tmp = transformYIQ2RGB(tmp_yiq)
            tmp[tmp > 1] = 1
            tmp[tmp < 0] = 0
        img_quant_trk.append(tmp)
        error_trk.append(np.power(comp_img - tmp, 2).mean())

        if verbos:
            print(len(set(borders)))
            print('#{} Error: {}'.format(it, error_trk[-1]))
            dispBorders(hist, borders, weighted_mean, error_trk[-1])

    # plt.figure()
    # dispBorders(hist, borders, weighted_mean, error_trk[-1])
    # plt.ioff()
    # plt.show()
    return img_quant_trk, error_trk
