import math
import numpy as np
import cv2
import math

from matplotlib import pyplot as plt


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
    in_len = len(in_signal)
    k = max(k_len, in_len)
    # padding
    length_k = len(k_size) - 1
    length_in = len(in_signal) - 1
    if length_in < length_k:
        for i in range(length_k):
            in_signal = np.insert(in_signal, 0, 0)
            in_signal = np.append(in_signal, 0)
    elif length_in > length_k:
        for i in range(length_in):
            k_size = np.insert(k_size, 0, 0)
            k_size = np.append(k_size, 0)

    t = 0
    while k >= 0:
        calc = 0
        if in_len < k_len:
            for j in range(len(k_size)):
                calc += k_size[j] * in_signal[j + t]
            res.append(calc)
            k -= 1
            t += 1
        elif in_len > k_len:
            for j in range(len(in_signal)):
                calc += k_size[j + t] * in_signal[j]
            res.append(calc)
            k -= 1
            t += 1

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
    # v = np.array([[1, 0, -1]])
    # row_conv = conv2D(in_image, v)
    # col_conv = conv2D(in_image, v.T)
    #
    # dirG = np.arctan(col_conv, row_conv)
    # magG = np.sqrt(np.power(row_conv, 2) + np.power(col_conv, 2))

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

    pass


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    pass


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    pass


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    pass
