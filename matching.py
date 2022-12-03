import numpy as np
import math
from tqdm import tqdm
class Block:
    def __init__(self,xcenter,ycenter,theta,length):
        #xcenter, ycenter: The center of our block
        #Theta: The orientation of the block
        #Length: The length of semimajor axis
        self.x = ycenter
        self.y = xcenter
        self.angle = theta
        self.length = length

    def GetCenter(self):
        return (self.x,self.y)

    def GetAngle(self):
        return self.angle

    def GetLength(self):
        return self.length

    def ComputeDistance(self,other):
        x2,y2 = other.GetCenter()
        return np.sqrt((self.x-x2)**2+(self.y-y2)**2)

def Baseline(block1,block2,threshold=0.7):
    x1,y1 = block1.GetCenter()
    x2,y2 = block2.GetCenter()
    a = y2-y1
    b = x1-x2
    if b == 0:
        return []
    c = x2 * y1 - x1 * y2
    theta = math.atan(-a/b)
    if theta < math.pi * -1/2 or theta >= math.pi/2:
        return []
    #print(a,b,c)
    D = block1.ComputeDistance(block2)
    l1 = block1.GetLength()/D
    l2 = block2.GetLength()/D
    ep =  (l1 - l2)**2
    ep += (l1 + l2-1)**2
    ep += (block1.GetAngle() - theta)**2
    ep += (block2.GetAngle() - theta)**2
    ep = -1.2 *ep
    score = math.exp(ep)
    #print(score)
    if score < threshold:
        return []

    return [a,b,c,D,score]

def ComputeProbability(a,b,c,D,block,factor):
    x,y = block.GetCenter()
    d = np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)
    dlocal = factor * D
    ep = (d - dlocal)/D
    ep =-4.8 * (ep ** 2)
    return math.exp(ep)

def TotalProbability(le,re,reb,leb,mouth,nose,Weights=[0.5,0.2,0.1,0.1,0.1],Factors=[1,0.3,0.6],threshold=0.7):
    #We add additional constraint here
    lex,ley = le.GetCenter()
    rex,rey = re.GetCenter()
    #Left eye should be on the left
    if lex >= rex:
        return 0
    
    lebx,leby = leb.GetCenter()
    rebx,reby = reb.GetCenter()
    
    #Eyebrow should be above eyes
    if leby > ley:
        return 0
    if lebx >= rebx:
        return 0
    if reby > rey:
        return 0
    
    mx,my = mouth.GetCenter()
    nx,ny = nose.GetCenter()
    
    #Mouth and nose should be in-between
    if mx > rebx or mx < lebx:
        return 0
    if nx > rebx or nx < lebx:
        return 0
    
    #Mouth should be below the nose
    if my < ny:
        return 0
    if ny < ley or ny < rey:
        return 0
    
    ret = Baseline(re,le,threshold)
    if len(ret) == 0:
        return 0
    
    a,b,c,D,score = ret
    score = score * Weights[0]
    score += Weights[1] * ComputeProbability(a,b,c,D,mouth,Factors[0])
    score += Weights[2] * ComputeProbability(a,b,c,D,reb,Factors[1])
    score += Weights[3] * ComputeProbability(a,b,c,D,leb,Factors[1])
    score += Weights[4] * ComputeProbability(a,b,c,D,nose,Factors[2])
    return score

def Matching(blocks):
    scores = []
    indices = []
    length = len(blocks)
    for le in tqdm(range(length)):
        
        for re in range(length):
            if re == le:
                continue
            ret = Baseline(blocks[le],blocks[re])
            if len(ret) == 0:
                continue

            for reb in range(length):
                if reb == le or reb == re:
                    continue

                for leb in range(length):
                    if leb == le or leb == re or leb == reb:
                        continue

                    for mouth in range(length):
                        if mouth == reb or mouth == leb or mouth == le or mouth == re:
                            continue

                        for nose in range(length):
                            if nose == mouth or nose == reb or nose == leb or nose == le or nose == re:
                                continue
                            score = TotalProbability(blocks[le],blocks[re],blocks[reb],blocks[leb],blocks[mouth],blocks[nose]) 
                            if score > 0.5:
                                scores.append(score)
                                indices.append((le,re,reb,leb,mouth,nose))

    max_idx = np.argmax(scores)
    print(len(scores))
    print("The block # for left eye, right eye, left eyebrow, right eyebrow, mouth and nose are ",indices[max_idx])
    print("The score is ",scores[max_idx])
    return indices[max_idx]

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
    
def PrintAllBlocks(image_name,block_indices = None):
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
        if block_indices is not None and i not in block_indices:
            continue
        
        top = int(alpha_min[i])
        bottom = int(alpha_max[i])
        left = int(beta_min[i])
        right = int(beta_max[i])
        face_locations.append([top,right,bottom,left])
        
    DrawRectangles(image_name,face_locations)

def FaceDetection(image_name,threshold = 20):
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
    # print(x_bar_all)
    # print(y_bar_all)
    # print(theta_all)


    block_indices = []
    blocks = []
    for i in range(len(block_lengths)):
        if block_lengths[i] > threshold:
            blocks.append(Block(x_bar_all[i], y_bar_all[i], theta_all[i], block_lengths[i]))
            block_indices.append(i)

    ret = Matching(blocks)
    final = []
    for each in ret:
        final.append(block_indices[each])
    face_locations = []
    for each in final:
        #x,y = blocks[each].GetCenter()
        top = int(alpha_min[each])
        bottom = int(alpha_max[each])
        left = int(beta_min[each])
        right = int(beta_max[each])
        face_locations.append([top,right,bottom,left])
    
    DrawRectangles(image_name,face_locations)

def FaceDetectionFinal(image_name,threshold = 20):
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
    # print(x_bar_all)
    # print(y_bar_all)
    # print(theta_all)


    block_indices = []
    blocks = []
    for i in range(len(block_lengths)):
        if block_lengths[i] > threshold:
            blocks.append(Block(x_bar_all[i], y_bar_all[i], theta_all[i], block_lengths[i]))
            block_indices.append(i)

    ret = Matching(blocks)
    final = []
    for each in ret:
        final.append(block_indices[each])
    face_locations = []
    for each in final:
        #x,y = blocks[each].GetCenter()
        top = int(alpha_min[each])
        bottom = int(alpha_max[each])
        left = int(beta_min[each])
        right = int(beta_max[each])
        face_locations.append([top,right,bottom,left])
    
    print(face_locations)
    dummy1 = np.max(np.array(face_locations),axis=0)
    dummy2 = np.min(np.array(face_locations),axis=0)
    face_locations = [[dummy2[0],dummy1[1],dummy1[2],dummy2[3]]]
    print(face_locations)
    DrawRectangles(image_name,face_locations)
