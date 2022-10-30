import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import cv2
from scipy import ndimage
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

def preprocess(image_name):
    # Tunable parameters. Can play around with these
    f = 11 # Boosting factor
    threshold = 20 # Threshold for binary image

    im = np.array(Image.open(image_name).convert("L"))

    # Sharpen image
    alpha = 0.05
    blurred = ndimage.gaussian_filter(im, 2)
    sharpened = im + alpha*(im-blurred)
    # new_img = Image.fromarray(sharpened.astype(np.uint8))
    # new_img.save("sharpened.jpg")

    # Opening operation
    element = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
    im = opening(sharpened, element)

    # Boost filtering
    dx_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    dy_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(im, dx_filter).astype(float)
    Iy = convolve(im, dy_filter).astype(float)
    sobel_im = (Ix*Ix + Iy*Iy)**0.5
    for i in range(sobel_im.shape[0]):
        for j in range(sobel_im.shape[1]):
            if sobel_im[i][j] > 255:
                sobel_im[i][j] = 255.
    
    sobel_im = sobel_im/255.

    g = im/255.
    z = np.zeros(im.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if sobel_im[i][j] > 0.5:
                z[i][j] = 0.4
            else:
                z[i][j] = 0.8
    w = np.zeros(im.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i][j] = (z[i][j]*(1 - sobel_im[i][j]) + (1-z[i][j])*(1-g[i][j])) * (f-8) + 8
    
    boosted_im = np.zeros(im.shape)
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            high_boost_mask = 1/9*np.array([[-1, -1, -1], [-1, w[i][j], -1], [-1, -1, -1]], np.float32)
            temp = convolve(im[i-1:i+2, j-1:j+2], high_boost_mask)
            boosted_im[i][j] = temp[1][1]
    
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            if boosted_im[i][j] > threshold:
                boosted_im[i][j] = 255
            else:
                boosted_im[i][j] = 0

    new_img = Image.fromarray(boosted_im.astype(np.uint8))
    new_img.save("boosted.jpg")
    return boosted_im

            
# preprocess('data/Friends/Train/Joey/joey (16).jpg')
preprocess('data/leonardo.jpg')