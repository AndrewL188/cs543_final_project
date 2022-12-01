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
    threshold = 30 # Threshold for binary image

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

# Returns the neighbors of a given point
def neighbors(im, point):
    res = []
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            if (i == point[0] and j == point[1]):
                continue
            if i > 0 and i < im.shape[0] and j > 0 and j < im.shape[1]:
                res.append((i,j))
    # if (point[0] < im.shape[0] - 1):
    #     res.append((point[0]+1, point[1]))
    # if (point[0] > 1):
    #     res.append((point[0]-1, point[1]))
    # if (point[1] < im.shape[1] - 1):
    #     res.append((point[0], point[1]+1))
    # if (point[1] > 1):
    #     res.append((point[0], point[1]-1))
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
    return labeled_im, cur_label

def label(im):
    labeled_im = np.zeros(im.shape)
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if im[i][j] == 0:
                labeled_im[i][j] = -1
            else:
                labeled_im[i][j] = 0
    
    return find_components(labeled_im)

import random
def label_illustration(labeled_im):
    label_to_color = {}
    r_channel = np.zeros(labeled_im.shape)
    g_channel = np.zeros(labeled_im.shape)
    b_channel = np.zeros(labeled_im.shape)
    ctr = 0
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if labeled_im[i][j] == 0:
                r = 255
                g = 255
                b = 255
            elif labeled_im[i][j] in label_to_color:
                r = label_to_color[labeled_im[i][j]][0]
                g = label_to_color[labeled_im[i][j]][1]
                b = label_to_color[labeled_im[i][j]][2]
            else:
                r = random.randint(0,255)
                g = random.randint(0,255)
                b = random.randint(0,255)
                # ctr += 1
                # if (ctr > 19):
                #     r = 255
                #     g = 255
                #     b = 255
                label_to_color[labeled_im[i][j]] = (r, g, b)
                
            r_channel[i][j] = r
            g_channel[i][j] = g
            b_channel[i][j] = b
    
    new_img = Image.fromarray(np.dstack((r_channel, g_channel, b_channel)).astype(np.uint8))
    new_img.save("labeled_image.jpg")

# im = preprocess('data/test_face.jpg')
# im = preprocess('data/leonardo.jpg')
# labeled_im, cur_label = label(im)
# label_illustration(labeled_im)

############ Calculating centre of mass and orientation and grouping ############
def calculate_mu(x_coords, y_coords, x_bar, y_bar, p, q):
    mu = 0
    x_diff = np.power(y_coords - y_bar, p)
    y_diff = np.power(x_coords - x_bar, q)
    mu = np.sum(x_diff)
    mu += np.sum(y_diff)
    # for i in range(len(x_coords)):
    #     for j in range(len(y_coords)):
    #         mu += np.power((x_coords[i] - x_bar),p) * np.power((y_coords[j] - y_bar),q)
    return mu

def get_component_indices(labeled_im, num_labels):
    block_number_coords = np.zeros(num_labels)
    x_bar_all = np.zeros(num_labels)
    y_bar_all = np.zeros(num_labels)
    theta_all = np.zeros(num_labels)
    x_coords_all = []
    y_coords_all = []
    for cur_label in range(1, num_labels + 1):
        # print(cur_label)
        x_coords = np.where( labeled_im == cur_label )[0] 
        y_coords = np.where( labeled_im == cur_label )[1] 
        x_coords_all.append(x_coords)
        y_coords_all.append(y_coords)
        block_number_coords[cur_label - 1] = len(x_coords)

        x_bar_all[cur_label - 1] = np.average(x_coords)
        y_bar_all[cur_label - 1] = np.average(y_coords)

        if (len(x_coords)>5000 or len(y_coords)>5000 or num_labels > 100):
            continue
        mu_1_1 = calculate_mu(x_coords, y_coords, x_bar_all[cur_label - 1], y_bar_all[cur_label - 1], 1, 1)
        mu_2_0 = calculate_mu(x_coords, y_coords, x_bar_all[cur_label - 1], y_bar_all[cur_label - 1], 2, 0)
        mu_0_2 = calculate_mu(x_coords, y_coords, x_bar_all[cur_label - 1], y_bar_all[cur_label - 1], 0, 2)

        if (mu_2_0 - mu_0_2) == 0: 
            continue
        theta_all[cur_label - 1] = 0.5 * ( np.arctan2( 2*mu_1_1, (mu_2_0 - mu_0_2) ) )

    return x_bar_all, y_bar_all, theta_all, block_number_coords, x_coords_all, y_coords_all

# test_im = np.zeros((100,100))
# for i in range(45,55):
#     test_im[50][i] = 1.
#     test_im[51][i] = 1.
#     test_im[60][i] = 2.
#     test_im[61][i] = 2.
#     test_im[i][98] = 3.
#     test_im[i][99] = 3.
# x_bar_all, y_bar_all, theta_all, block_number_coords, x_coords_all, y_coords_all = get_component_indices(test_im, 3)
# print(x_bar_all)
# print(y_bar_all)
# print(theta_all)
# print(block_number_coords)
# print(x_coords_all)
# print(y_coords_all)

def find_blocks_within_radius(cur_block_idx, labeled_im, x_bar, y_bar, radius):
    min_y = int(np.floor(y_bar - radius))
    max_y = int(np.ceil(y_bar + radius))

    blocks_within_radius = []
    for y in range(min_y + 1, max_y):
        # From the formula: r^2 = (x-h)^2 + (y-k)^2, where 
        # h,k are the distances from x,y respectively
        x_diff = np.power(radius,2) - np.power(y-y_bar,2)
        min_x = int(np.floor(x_bar - x_diff))
        max_x = int(np.ceil(x_bar + x_diff))

        for x in range(min_x + 1, max_x):
            if x < labeled_im.shape[0] and y < labeled_im.shape[1] and labeled_im[x,y] != 0 and labeled_im[x,y] != cur_block_idx+1:
                if labeled_im[x,y] not in blocks_within_radius:
                    blocks_within_radius.append(labeled_im[x,y])
    return blocks_within_radius


def find_nearest_block(cur_block_idx, blocks_within_radius, x_bar, y_bar):
    min_distance_block = None
    min_distance = np.inf
    for block in blocks_within_radius:
        x_dist = np.power(x_bar[int(block - 1)] - x_bar[cur_block_idx], 2)
        y_dist = np.power(y_bar[int(block - 1)] - y_bar[cur_block_idx], 2)
        distance_to_cur_block = np.sqrt(x_dist + y_dist)
        if distance_to_cur_block < min_distance:
            min_distance = distance_to_cur_block
            min_distance_block = block

    return min_distance_block 

def grouping(labeled_im, cur_label, block_number_coords, x_bar, y_bar, x_coords_all, y_coords_all):
    N_max = 100 #??????? This changes based on the image - need to test different values
    for N in range(1, N_max + 1):
        radius = 8 - ( 6 * ( (N - 1) / (N_max - 1) ) )
        count = 0
        for i in range(cur_label):
            blocks_within_radius = []
            if block_number_coords[i] == N:
                blocks_within_radius = find_blocks_within_radius(i, labeled_im, x_bar[i], y_bar[i], radius)
            else:
                continue

            if len(blocks_within_radius) > 0:
                nearest_block = int(find_nearest_block(i, blocks_within_radius, x_bar, y_bar))
            else:
                continue

            #lies on same axis if any of the x coordinates are the same
            # lies_on_same_axis = bool(set(x_coords_all[nearest_block - 1]) & set(x_coords_all[i]))
            # lies_on_same_axis = False
            # if block_number_coords[nearest_block - 1] < N_max or lies_on_same_axis:
            block_to_merge_with = max(i + 1, nearest_block)
            block_to_merge = min(i + 1, nearest_block)

            # print(str(i+1) + " " + str(nearest_block))
            # print(str(block_to_merge_with) + " " + str(block_to_merge))

            x_coords = x_coords_all[block_to_merge - 1]
            y_coords = y_coords_all[block_to_merge - 1]

            labeled_im[x_coords, y_coords] = block_to_merge_with
            block_number_coords[block_to_merge_with-1] = block_number_coords[block_to_merge_with-1] + block_number_coords[block_to_merge-1]
            block_number_coords[block_to_merge-1] = 0
            new_x_coords = np.append(x_coords_all[block_to_merge_with-1], x_coords)
            new_y_coords = np.append(y_coords_all[block_to_merge_with-1], y_coords)
            x_coords_all[block_to_merge_with-1] = new_x_coords
            y_coords_all[block_to_merge_with-1] = new_y_coords
            x_coords_all[block_to_merge-1] = np.array([])
            y_coords_all[block_to_merge-1] = np.array([])

    return labeled_im

# Make labels consecutive in order after grouping
def relabel_grouped_im(labeled_im, num_labels):
    old_label_to_new_map = {}
    ctr = 0
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if labeled_im[i][j] == 0:
                continue
            if labeled_im[i][j] not in old_label_to_new_map:
                ctr += 1
                old_label_to_new_map[labeled_im[i][j]] = ctr
            labeled_im[i][j] = old_label_to_new_map[labeled_im[i][j]]
    if ctr != num_labels:
        print("something broke")
        print(ctr)
        print(num_labels)

# For testing
def use_only_good_labels(labeled_im, labels_to_use):
    num_labels = len(labels_to_use)
    old_label_to_new_map = {}
    ctr = 0
    for i in range(labeled_im.shape[0]):
        for j in range(labeled_im.shape[1]):
            if labeled_im[i][j] == 0:
                continue
            if labeled_im[i][j] not in labels_to_use:
                labeled_im[i][j] = 0
                continue
            if labeled_im[i][j] not in old_label_to_new_map:
                ctr += 1
                old_label_to_new_map[labeled_im[i][j]] = ctr
            labeled_im[i][j] = old_label_to_new_map[labeled_im[i][j]]
    if ctr != num_labels:
        print("something broke")
        print(ctr)
        print(num_labels)

            
# Default 60 length for testing
# Need to find a way to find length of semimajor axis
def get_block_lengths(x_bar_all, y_bar_all, theta_all, x_coords_all, y_coords_all):
    print(x_bar_all.shape)
    block_lengths = np.zeros(len(theta_all))
    for i in range(len(block_lengths)):
        # if i == 8: print(x_coords_all[i], y_coords_all[i], theta_all[i])
        x = np.linspace(min(x_coords_all[i]), max(x_coords_all[i]), num=max(x_coords_all[i]) - min(x_coords_all[i]) + 1)
        y = np.linspace(min(y_coords_all[i]), max(y_coords_all[i]), num=max(y_coords_all[i]) - min(y_coords_all[i]) + 1)
        # if i == 0: print(x,y)
        max_x = -np.Inf
        max_y = -np.Inf
        min_x = np.Inf
        min_y = np.Inf
        for a in x:
            for j in y:
                alpha = a * np.cos(theta_all[i]) + j * np.sin(theta_all[i])
                beta = -a * np.sin(theta_all[i]) + j * np.cos(theta_all[i]) 
                if alpha < min_x:
                    min_x = alpha
                if alpha > max_x:
                    max_x = alpha
                if beta < min_y:
                    min_y = beta
                if beta > max_y:
                    max_y = beta
        block_lengths[i] = max(max_x - min_x, max_y - min_y)
        
#         block_lengths[i] = 60
    return block_lengths

from matching import Matching, Block
def classifyFace(image_name):
    im = preprocess(image_name)
    labeled_im, cur_label = label(im)
    x_bar_all, y_bar_all, theta_all, block_number_coords, x_coords_all, y_coords_all = get_component_indices(labeled_im, cur_label)
    labeled_im = grouping(labeled_im, cur_label, block_number_coords, x_bar_all, y_bar_all, x_coords_all, y_coords_all)
    print(str(len(np.unique(labeled_im)) - 1) + " Connected Components")
    num_labels = len(np.unique(labeled_im)) - 1
    relabel_grouped_im(labeled_im, num_labels)

    # good_labels = {20, 21, 22, 23, 24, 25}
    # use_only_good_labels(labeled_im, good_labels)
    # num_labels = len(np.unique(labeled_im)) - 1
    label_illustration(labeled_im)

    x_bar_all, y_bar_all, theta_all, block_number_coords, x_coords_all, y_coords_all = get_component_indices(labeled_im, num_labels)
    block_lengths = get_block_lengths(x_bar_all, y_bar_all, theta_all, x_coords_all, y_coords_all)
    print("block lengths")
    # print(x_bar_all, y_bar_all)
    print(block_lengths)
    print("block lengths end")
    # print(x_bar_all)
    # print(y_bar_all)
    # print(theta_all)


    blocks = []
    for i in range(len(block_lengths)):
        blocks.append(Block(y_bar_all[i], x_bar_all[i], theta_all[i], block_lengths[i]))
    print(Matching(blocks))

classifyFace('data/test_face.jpg')