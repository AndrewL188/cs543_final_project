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
    
    eyelength = max(le.GetLength(),re.GetLength())
    if leb.GetLength() > 2.5 * eyelength:
        return 0
    if reb.GetLength() > 2.5 * eyelength:
        return 0
    
    if nose.GetLength() > 3 * eyelength:
        return 0
    
    if mouth.GetLength() > 4 * eyelength:
        return 0
    
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