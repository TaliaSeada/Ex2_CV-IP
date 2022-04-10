import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # first flip the kernel
    k_size = np.flip(k_size)
    res = []
    k_len = len(k_size)
    in_size = len(in_signal)

    # padding
    length_k = len(k_size) - 1
    for i in range(length_k):
        in_signal = np.insert(in_signal, 0, 0)
        in_signal = np.append(in_signal, 0)

    # calculate the convolution
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
    # padding
    k_shape = kernel.shape
    padd_r = math.floor(k_shape[0] / 2)
    padd_c = math.floor(k_shape[1] / 2)
    image = cv2.copyMakeBorder(in_image, padd_r, padd_r, padd_c, padd_c, cv2.BORDER_REPLICATE, None, value=0)

    # flip the kernel
    kernel = np.flip(kernel)

    # calculate the convolution
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
    # x derivative
    X = cv2.filter2D(in_image, -1, v)
    # y derivative
    Y = cv2.filter2D(in_image, -1, v.T)

    # direction
    dirG = np.arctan2(Y, X).astype(np.float64)
    # magnitude
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
    # laplacian matrix
    lap_mat = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    # convolution with the laplacian matrix
    img_conv = cv2.filter2D(img, -1, lap_mat, borderType=cv2.BORDER_REPLICATE)

    # check all the neighbors, look for {+, 0, -} or {-, 0, +} or {+, -} or {-, +}
    res_img = np.zeros(img.shape)
    shape = img.shape
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            # use help function to find edges
            if (sum_nei(img_conv, i, j) > 0):
                res_img[i][j] = 1
    return res_img


def sum_nei(img_conv, i, j):
    # returns 1 only if found {+, 0, -} or {-, 0, +} or {+, -} or {-, +}
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

    # else returns 0
    return 0


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    # https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    # smooth image with the gaussian blur then send it to previous function
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
    # create number of jumps for the radii
    if max_radius - min_radius < 11:
        jump = 1
    elif max_radius - min_radius < 51:
        jump = 3
    else:
        jump = 5

    rows = img.shape[0]
    cols = img.shape[1]

    # Canny Edge detector as the edge detector
    img = cv2.Canny((img * 255).astype(np.uint8), img.shape[0], img.shape[1])
    # set the radii list
    radius = range(min_radius, max_radius)
    circles = []
    # threshold for the circles detection
    threshold = 20
    for r in range(0, len(radius), jump):
        # function to find the circles in hough space
        voting = create(img, rows, cols, radius, r)
        # threshold
        if np.max(voting) > threshold:
            voting[voting < threshold] = 0
            # function to find the circles for r
            circles = find_circles(rows, cols, voting, threshold, circles, radius, r)
    return circles


def create(img, rows, cols, radius, r):
    voting = np.zeros(img.shape)
    # Make voting array
    for i in range(rows):
        for j in range(cols):
            if img[i, j] == 255:
                for angle in range(0, 360, 10):
                    b = j - round(np.sin(angle * np.pi / 180) * radius[r])
                    a = i - round(np.cos(angle * np.pi / 180) * radius[r])
                    if 0 <= a < rows and 0 <= b < cols:
                        voting[a, b] += 1
    return voting


def find_circles(rows, cols, voting, threshold, circles, radius, r):
    # find circles in the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if voting[i, j] >= threshold:
                # calculate the average of the neighbors
                avg_sum = voting[i - 1:i + 2, j - 1:j + 2].sum() / 9
                if avg_sum >= threshold / 9:
                    # adding only the circles that dont touch other circles
                    if all((i - xc) ** 2 + (j - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                        circles.append((j, i, radius[r]))
                        # "remove" any other circle with those (i, j, r)
                        voting[i - radius[r]:i + radius[r], j - radius[r]:j + radius[r]] = 0
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
    # cv2 function
    cv_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    shape = in_image.shape
    # padding
    pad = math.floor(k_size / 2)
    padded_image = cv2.copyMakeBorder(in_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    res = np.zeros(shape)

    # for each pixel activate the bilateral filter, then add the result to the new image
    for x in range(shape[0]):
        for y in range(shape[1]):
            pivot_v = in_image[x, y]
            neighbor_hood = padded_image[x:x + k_size, y:y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            gaus = cv2.getGaussianKernel(k_size, k_size)
            gaus = gaus.dot(gaus.T)
            combo = gaus * diff_gau
            # bilateral formula
            result = (combo * neighbor_hood).sum() / combo.sum()
            res[x][y] = result
    return cv_image, res