def DrawRectangles(filename,face_locations):
    #start = time.time()
    image = cv2.imread(filename)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    for each in face_locations:
        top,right,bottom,left = each
        for i in range(left,right):
            R[top][i] = 255
            R[top+1][i] = 255
            if top-1 >= 0:
                R[top-1][i] = 255
            R[bottom][i] = 255
            if bottom + 1 < height:
                R[bottom+1][i] = 255
            R[bottom-1][i] = 255
            
        for j in range(top,bottom):
            R[j][left] = 255
            R[j][right] = 255
            R[j][left+1] = 255
            
            if left - 1 >= 0:
                R[j][left-1] = 255
            if right + 1 < width:
                R[j][right+1] = 255
            R[j][right-1] = 255
    
    
    outfile = filename.split(".")[0] + "_out.jpg"
    print(outfile)
    im = Image.fromarray(np.stack([B,G,R],axis=2))
    im.show()
    cv2.imwrite(outfile,np.stack([R,G,B],axis=2))
    
def PrintAllBlocks(image_name):
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
    block_lengths, alpha_max, alpha_min, beta_max, beta_min = get_block_lengths(x_bar_all, y_bar_all, theta_all, x_coords_all, y_coords_all)
    print("block lengths")
    # print(x_bar_all, y_bar_all)
    print(block_lengths)
    print("block lengths end")
    face_locations = []
    for i in range(len(block_lengths)):
        top = int(alpha_min[i])
        bottom = int(alpha_max[i])
        left = int(beta_min[i])
        right = int(beta_max[i])
        face_locations.append([top,right,bottom,left])
        
    DrawRectangles(image_name,face_locations)
    