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

    # new_img = Image.fromarray(boosted_im.astype(np.uint8))
    # new_img.save("boosted.jpg")
    return boosted_im

# Returns the neighbors of a given point
def neighbors(im, point):
    res = []
    if (point[0] < im.shape[0] - 1):
        res.append((point[0]+1, point[1]))
    if (point[0] > 1):
        res.append((point[0]-1, point[1]))
    if (point[1] < im.shape[1] - 1):
        res.append((point[0], point[1]+1))
    if (point[1] > 1):
        res.append((point[0], point[1]-1))
    return res

def search(labeled_im, cur_label, r, c):
    stack = []
    stack.append((r,c))
    while stack:
        cur_node = stack.pop()
        labeled_im[cur_node[0]][cur_node[1]] = cur_label
        cur_neighbors = neighbors(labeled_im, cur_node)
        for i,j in cur_neighbors:
            if (labeled_im[i][j] == -1):
                stack.append((i,j))

def find_components(labeled_im):
    cur_label = 0
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if labeled_im[i][j] == -1:
                cur_label += 1
                search(labeled_im, cur_label, i, j)
    print(str(cur_label) + " connected components")
    return labeled_im

def label(im):
    labeled_im = np.zeros(im.shape)
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if im[i][j] == 0:
                labeled_im[i][j] = -1
            else:
                labeled_im[i][j] = 0
    
    return find_components(labeled_im)

            
# preprocess('data/Friends/Train/Joey/joey (16).jpg')
im = preprocess('data/leonardo.jpg')
labeled_im = label(im)

############ Calculating centre of mass and orientation and grouping ############
def calculate_mu(x_coords, y_coords, x_bar, y_bar, p, q):
    mu = 0
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            mu += np.power((x_coords[i] - x_bar),p) * np.power((y_coords[j] - y_bar),q)

    return mu

def get_component_indices(labeled_im, cur_label):
    block_number_coords = np.zeros(cur_label)
    x_bar_all = np.zeros(cur_label)
    y_bar_all = np.zeros(cur_label)
    theta_all = np.zeros(cur_label)
    for cur_label in range(1, cur_label + 1):
        x_coords = np.where( labeled_im == cur_label )[0] )
        y_coords = np.where( labeled_im == cur_label )[1] )
        block_number_coords = len(x_coords)

        x_bar[cur_label - 1] = np.int( np.average(x_coords) )
        y_bar[cur_label - 1] = np.int( np.average(y_coords) )

        mu_1_1 = calculate_mu(x_coords, y_coords, x_bar[cur_label - 1], y_bar[cur_label - 1], 1, 1)
        mu_2_0 = calculate_mu(x_coords, y_coords, x_bar[cur_label - 1], y_bar[cur_label - 1], 2, 0)
        mu_0_2 = calculate_mu(x_coords, y_coords, x_bar[cur_label - 1], y_bar[cur_label - 1], 0, 2)
        
        theta[cur_label - 1] = 0.5 * ( np.arctan2( ( 2*mu_1_1 ) / (mu_2_0 - mu_0_2) ) )

    return x_bar_all, y_bar_all, theta_all, block_number_coords

def find_blocks_within_radius(cur_block_idx, labeled_im, x_bar, y_bar, radius):
    min_y = np.floor(y_bar - r)
    max_y = np.ceil(y_bar + r)

    blocks_within_radius = []
    for y in range(min_y + 1, max_y):
        # From the formula: r^2 = (x-h)^2 + (y-k)^2, where 
        # h,k are the distances from x,y respectively
        x_diff = np.power(radius,2) - np.power(y-y_bar,2)
        min_x = np.floor(x_bar - x_diff)
        max_x = np.ceil(x_bar + x_diff)

        for x in range(min_x + 1, max_x):
            if labeled_im[x,y] != 0 and labeled_im[x,y] != cur_block_idx:
                if labeled_im[x,y] not in blocks_within_radius:
                    blocks_within_radius.append(labeled_im[x,y])


def find_nearest_block(cur_block_idx, blocks_within_radius, x_bar, y_bar):
    min_distance_block = None
    min_distance = np.inf
    for block in blocks_within_radius:
        x_dist = np.power(x_bar[block - 1] - x_bar[cur_block_idx], 2)
        y_dist = np.power(y_bar[block - 1] - y_bar[cur_block_idx], 2)
        distance_to_cur_block = np.sqrt(x_dist + y_dist)
        if distance_to_cur_block < min_distance:
            min_distance = distance_to_cur_block
            min_distance_block = block

    return min_distance_block 


def grouping(labeled_im, cur_label, block_number_coords, x_bar, y_bar):
    N_max = 30 #??????? This changes based on the image - need to test different values
    for N in range(1, N_max + 1):
        radius = 8 - ( 6 * ( (N - 1) / (N_max -1) ) )
        
        for i in range(cur_label):
            blocks_within_radius = []
            if block_number_coords[i] == N:
                blocks_within_radius = find_blocks_within_radius(i, labeled_im, x_bar[i], y_bar[i], radius)
            
            if len(blocks_within_radius) > 0:
                nearest_block = find_nearest_block(i, blocks_within_radius, x_bar, y_bar)

            #if block_number_coords[i] < N_max #### Merge step (based on what the matching algorithm needs)