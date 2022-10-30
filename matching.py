import numpy as np
import math
class Block:
	def __init__(self,xcenter,ycenter,theta,length):
		#xcenter, ycenter: The center of our block
		#Theta: The orientation of the block
		#Length: The length of semimajor axis
		self.x = xcenter
		self.y = ycenter
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

def Baseline(block1,block2,threshold):
	x1,y1 = block1.GetCenter()
	x2,y2 = block2.GetCenter()
	a = y2-y1
	b = x1-x2
	c = x2 * y1 - x1 * y2
	theta = math.atan(-a/b)
	if theta < math.pi * -1/2 or theta >= math.pi/2:
		return []

	ep =  (block1.GetLength() - block2.GetLength())**2
	ep += (block1.GetLength() + block2.GetLength()-1)**2
	ep += (block1.GetAngle() - theta)**2
	ep += (block2.GetAngle() - theta)**2
	ep = -1.2 *ep
	score = math.exp(ep)
	#print(score)
	if score < threshold:
		return []

	return [a,b,c,block1.ComputeDistance(block2),score]

def ComputeProbability(a,b,c,D,block,factor):
	x,y = block.GetCenter()
	d = np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)
	dlocal = factor * D
	ep = (d - dlocal)/D
	ep =-4.8 * (ep ** 2)
	return math.exp(ep)

def TotalProbability(reb,leb,mouth,nose,Weights=[0.5,0.2,0.1,0.1,0.1],Factors=[1,0.3,0.6],threshold=0.7):
	ret = Baseline(reb,leb,threshold)
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
	for reb in range(length):

		for leb in range(length):
			if reb == leb:
				continue

			for mouth in range(length):
				if mouth == reb or mouth == leb:
					continue

				for nose in range(length):
					if nose == mouth or nose == reb or nose == leb:
						continue
					score = TotalProbability(blocks[reb],blocks[leb],blocks[mouth],blocks[nose]) 
					if score > 0:
						scores.append(score)
						indices.append((reb,leb,mouth,nose))

	return (len(scores) != 0)


if __name__ == '__main__':
	import sys
	block1 = Block(50,70,0.4,12)
	block2 = Block(20,90,0.1,20)
	block3 = Block(100,150,0.2,7)
	block4 = Block(190,160,0.1,17)
	block5 = Block(200,140,0.1,27)
	block6 = Block(300,250,0.1,15)
	print(Matching([block1,block2,block3,block4,block5,block6]))

