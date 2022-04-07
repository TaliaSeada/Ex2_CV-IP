import math
import numpy as np
import cv2
import math

from matplotlib import pyplot as plt
from numpy import inexact


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    k_size = np.flip(k_size)
    res = []
    k_len = len(k_size)
    in_size = len(in_signal)

    # padding
    length_k = len(k_size) - 1
    for i in range(length_k):
        in_signal = np.insert(in_signal, 0, 0)
        in_signal = np.append(in_signal, 0)

    for j in range(in_size + k_len - 1):
        calc = (k_size * in_signal[j: j + k_len]).sum()
        res.append(calc)

    return np.array(res)


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    k_shape = kernel.shape
    padd_r = math.floor(k_shape[0] / 2)
    padd_c = math.floor(k_shape[1] / 2)
    image = cv2.copyMakeBorder(in_image, padd_r, padd_r, padd_c, padd_c, cv2.BORDER_REPLICATE, None, value=0)
    kernel = np.flip(kernel)

    im_shape = in_image.shape
    im_r = im_shape[0]
    im_c = im_shape[1]

    res_img = np.zeros(in_image.shape)
    for i in range(im_r):
        for j in range(im_c):
            res_img[i][j] = calc(image, kernel, i, j)

    return res_img


def calc(image, kernel, k, t):
    res = 0
    k_shape = kernel.shape
    r = k_shape[0]
    c = k_shape[1]
    for i in range(r):
        for j in range(c):
            res += kernel[i][j] * image[k + i][t + j]
    return np.round(res)


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    v = np.array([[1, 0, -1]])
    X = cv2.filter2D(in_image, -1, v)
    Y = cv2.filter2D(in_image, -1, v.T)

    dirG = np.arctan2(Y, X).astype(np.float64)
    magG = np.sqrt(X ** 2 + Y ** 2).astype(np.float64)
    return dirG, magG


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    pass


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
    k = cv2.getGaussianKernel(k_size, -1)
    img = cv2.filter2D(in_image, -1, k, borderType=cv2.BORDER_REPLICATE)
    return img


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    lap_mat = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    img_conv = cv2.filter2D(img, -1, lap_mat, borderType=cv2.BORDER_REPLICATE)

    res_img = np.zeros(img.shape)
    shape = img.shape
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            if (sum_nei(img_conv, i, j) > 0):
                res_img[i][j] = 1
    return res_img


def sum_nei(img_conv, i, j):
    if img_conv[i - 1][j - 1] > 0 and img_conv[i + 1][j + 1] < 0:
        return 1
    elif img_conv[i - 1][j - 1] < 0 and img_conv[i + 1][j + 1] > 0:
        return 1

    if img_conv[i - 1][j] > 0 and img_conv[i + 1][j] < 0:
        return 1
    elif img_conv[i - 1][j] < 0 and img_conv[i + 1][j] > 0:
        return 1

    if img_conv[i - 1][j + 1] > 0 and img_conv[i + 1][j - 1] < 0:
        return 1
    elif img_conv[i - 1][j + 1] < 0 and img_conv[i + 1][j - 1] > 0:
        return 1

    if img_conv[i][j + 1] > 0 and img_conv[i][j - 1] < 0:
        return 1
    elif img_conv[i][j + 1] < 0 and img_conv[i][j - 1] > 0:
        return 1

    return 0


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    # https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    smoothed_im = cv2.GaussianBlur(img, (9, 9), 0)
    return edgeDetectionZeroCrossingSimple(smoothed_im)


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension :param I: Input image
    :param img: image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    img = img.squeeze()
    rows = img.shape[0]
    cols = img.shape[1]
    img_edges = cv2.Canny((img * 255).astype(np.uint8), img.shape[0], img.shape[1])
    circles = []
    for radius in range(min_radius, max_radius):
        voting = np.zeros(img_edges.shape)
        for i in range(rows):
            for j in range(cols):
                if img_edges[i, j] == 255:
                    for angel in range(360):
                        a = int(j - np.cos(angel * np.pi / 180) * radius)
                        b = int(i - np.sin(angel * np.pi / 180) * radius)
                        if 0 <= a < cols and 0 <= b < rows:
                            voting[b, a] += 1
        if voting.max() > 150:
            voting[voting < 150] = 0
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if voting[i, j] >= 150:
                        avg_sum = voting[i - 1:i + 2, j - 1:j + 2].sum() / 9
                        if avg_sum >= 150 / 9:
                            if all((j - xc) * 2 + (i - yc) * 2 > rc ** 2 for xc, yc, rc in circles):
                                circles.append((j, i, radius))
                                voting[i - radius:i + radius, j - radius:j + radius] = 0
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    ans = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    imgbi = np.zeros_like(in_image)

    for y in range(k_size, in_image.shape[0] - k_size):
        for x in range(k_size, in_image.shape[1] - k_size):
            pivot_v = in_image[y, x]
            neighbor_hood = in_image[
                            y - k_size:y + k_size + 1,
                            x - k_size:x + k_size + 1
                            ]
            sigma = sigma_color
            up = []
            for nei in neighbor_hood:
                diff = abs(neighbor_hood.astype(int) - pivot_v)
                diff_gau = np.exp(-(diff / (2 * sigma_color)))
                gaus = cv2.getGaussianKernel(2 * k_size + 1, k_size)
                gaus = gaus.dot(gaus.T)
                combo = gaus * diff_gau
                result = ((combo * neighbor_hood) / combo.sum()).mean()
                imgbi[y, x] = result * 255

    return ans, imgbi
