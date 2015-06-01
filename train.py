import cv2
import numpy as np
import random
import time
import glob
import math	
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

def createFeatures(height,width):
#	height *= 10
#	width *= 10 
	a = []	
	for x in range(5000):
		Xs  = []
		Ys  = []
		b = []	
		z = int(random.random()*9)
		b.append(z)
		f = int(random.random()*12)
		b.append(f)
		for y in range(4):
			z = int(random.random()*(width - 1))
			w = int(random.random()*(height - 1))
			Xs.append(w)
			Ys.append(z)
		Xs.sort()
		Ys.sort()
		if f >= 8:
			b.append(Xs[0])#2
			b.append(Ys[0])#3
			b.append(Xs[1])#4
			b.append(Ys[1])#5
			b.append(Xs[2])#6
			b.append(Ys[2])#7
			b.append(Xs[3])#8
			b.append(Ys[3])#9
		elif f >= 4:
			b.append(Xs[1])#2
			b.append(Ys[0])#3
			b.append(Xs[2])#4
			b.append(Ys[1])#5
			b.append(Xs[0])#6
			b.append(Xs[3])#7
			b.append(Ys[2])#8
			b.append(Ys[3])#9
		elif f >= 0:
			b.append(Xs[0])#2
			b.append(Ys[1])#3
			b.append(Xs[1])#4
			b.append(Ys[2])#5
			b.append(Ys[0])#6
			b.append(Ys[3])#7
			b.append(Xs[2])#8
			b.append(Xs[3])#9
		a.append(b)
	np.savetxt("../csvs/some%d%d.csv"%(height,width),a,delimiter = ",")
	return a

def LoadImage(a):
	img = cv2.imread(a,-1)
	cv2.line(img,(632,230),(710,318),(100,100,100),4)
	cv2.imshow(a,img)
	cv2.waitKey(0)
	cv2.destroyWindow(a)
	return img 



def ShowImage(img):
	cv2.imshow("w",img)
	cv2.waitKey(0)
	cv2.destroyWindow("w")

def readFeatures(height,width):
	height /= 10
	width /= 10
	height *= 10
	width *= 10
	a = np.loadtxt(open("../csvs/some%d%d.csv"%(height,width),"rb"),delimiter=",")
	return a.astype(int)

def Evaluate(a,b,row,col):
	vector = []
	for z in a:
		x = b[z[0]]
		f = z[1]
		if f < 1: #  2 < 4  and 6 < 3 < 5
			y = x[row + z[2],col + z[3]]*2 + x[row + z[4],col + z[5]] - x[row + z[2],col + z[5]] - x[row + z[4],col + z[3]]*2 - x[row + z[2],col + z[6]]  + x[row + z[4],col + z[6]]
		elif f < 2: #  2 < 4  and 6 < 3 < 5
			y = -x[row + z[2],col + z[3]]*2 - x[row + z[4],col + z[5]] + x[row + z[2],col + z[5]] + x[row + z[4],col + z[3]]*2 + x[row + z[2],col + z[6]]  - x[row + z[4],col + z[6]]
		elif f < 3: #  2 < 4 and 6 < 3 < 5 < 7
			y = x[row + z[2],col + z[3]]*2 + 2*x[row + z[4],col + z[5]] - 2*x[row + z[2],col + z[5]] - x[row + z[4],col + z[3]]*2 - x[row + z[2],col + z[6]] + x[row + z[4],col + z[6]] - x[row + z[4],col + z[7]] +  x[row + z[2],col + z[7]]
		elif f < 4: #  2 < 4  and 6 < 3 < 5 < 7
			y = -(x[row + z[2],col + z[3]]*2 + 2*x[row + z[4],col + z[5]] - 2*x[row + z[2],col + z[5]] - x[row + z[4],col + z[3]]*2 - x[row + z[2],col + z[6]] + x[row + z[4],col + z[6]] - x[row + z[4],col + z[7]] +  x[row + z[2],col + z[7]])
		elif f < 5: #   3 < 5 and 6 < 2 < 4 < 7
			y = x[row + z[2],col + z[3]]*2 + x[row + z[4],col + z[5]] - x[row + z[2],col + z[5]]*2 - x[row + z[4],col + z[3]] - x[row + z[6],col + z[3]]  + x[row + z[4],col + z[5]]
		elif f < 6: #   3 < 5 and 6 < 2 < 4 < 7
			y = -(x[row + z[2],col + z[3]]*2 + x[row + z[4],col + z[5]] - x[row + z[2],col + z[5]]*2 - x[row + z[4],col + z[3]] - x[row + z[6],col + z[3]]  + x[row + z[4],col + z[5]])
		elif f < 7: #   3 < 5 and 6 < 2 < 4 < 7
			y = x[row + z[2],col + z[3]]*2 + x[row + z[4],[5]] - x[row + z[2],[5]]*2 - 2*x[row + z[4],col + z[3]] - x[row + z[6],col + z[3]]  + x[row + z[4],col + z[5]] - x[row + z[7],col + z[5]] +  x[row + z[7],col + z[3]]
		elif f < 8: #   3 < 5 and 6 < 2 < 4 < 7
			y = -(x[row + z[2],col + z[3]]*2 + x[row + z[4],col + z[5]]*2 - x[row + z[2],col + z[5]]*2 - 2*x[row + z[4],col + z[3]] - x[row + z[6],col + z[3]]  + x[row + z[4],col + z[5]] - x[row + z[7],col + z[5]] +  x[row + z[7],col + z[3]])
		elif f < 9:  # ascending
			y = x[row + z[2],col + z[3]] + x[row + z[8],col + z[9]]  - x[row + z[2],col + z[9]] -x[row + z[8],col + z[3]] - x[row + z[4],col + z[5]] - x[row + z[6],col + z[7]]  + x[row + z[4],col + z[7]] +x[row + z[6],col + z[5]] 
		elif f < 10: 
			y = -(x[row + z[2],col + z[3]] + x[row + z[8],col + z[9]]  - x[row + z[2],col + z[9]] -x[row + z[8],col + z[3]] - x[row + z[4],col + z[5]] - x[row + z[6],col + z[7]]  + x[row + z[4],col + z[7]] +x[row + z[6],col + z[5]])
		elif f < 11:
			y = x[row + z[2],col + z[3]]  +  x[row + z[6],col + z[7]] +  x[row + z[2],col + z[7]]  +   x[row + z[6],col + z[3]]  + 4*x[row + z[4],col + z[5]]  - 2*x[row + z[2],col + z[5]] - 2*x[row + z[4],col + z[3]] - 2*x[row + z[6],col + z[5]] - 2*x[row + z[4],col + z[7]] 
		elif f < 12:
			y = -(x[row + z[2],col + z[3]]  +  x[row + z[6],col + z[7]] +  x[row + z[2],col + z[7]]  +   x[row + z[6],col + z[3]]  + 4*x[row + z[4],col + z[5]]  - 2*x[row + z[2],col + z[5]] - 2*x[row + z[4],col + z[3]] - 2*x[row + z[6],col + z[5]] - 2*x[row + z[4],col + z[7]]) 
		vector.append(y)
	return vector

def run(img,a= 0,rows= 0,cols= 0,bdt= 0):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	Y,U,V = cv2.split(YUV)
	r,g,b = cv2.split(img)
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[0] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	dst = cv2.filter2D(img,cv2.CV_16S,kernel1)
#	print dst.size
	rv,gv,bv = cv2.split(dst)
	rv1 = np.int16(rv)
	gv1 = np.float32(gv)
	bv1 = np.float32(bv)
	rv1 = cv2.pow(rv1,2)
	gv1 = cv2.pow(gv1,2)
	bv1 = cv2.pow(bv1,2)
	dst = cv2.filter2D(img,cv2.CV_16S,kernel2)
	rh,gh,bh = cv2.split(dst)
	rh1 = np.int16(rh)
	gh1 = np.float32(gh)
	bh1 = np.float32(bh)
	rh1 = cv2.pow(rh1,2)
	gh1 = cv2.pow(gh1,2)
	bh1 = cv2.pow(bh1,2)
	r1 = rh1 + rv1
	g1 = gh1 + gv1
	b1 = bh1 + bv1
	r1 = np.float32(r1)
	g1 = np.float32(g1)
	b1 = np.float32(b1)
	rfinal = cv2.sqrt(r1).astype(np.uint8)
	bfinal = cv2.sqrt(g1).astype(np.uint8)
	gfinal = cv2.sqrt(b1).astype(np.uint8)
	red = (rfinal > bfinal) & (rfinal > gfinal)
	red = red.astype(np.uint8)
	blue = (bfinal > rfinal) & (bfinal > gfinal)
	blue = blue.astype(np.uint8)
	green = (gfinal > bfinal) & (gfinal > rfinal)
	green = green.astype(np.uint8)
	redfh = red*rh
	bluefh = blue*bh
	greenfh = green*gh
	redfv = red*rv
	bluefv = blue*bv
	greenfv = green*gv
	redm = red*rfinal
	bluem = blue*bfinal
	greenm = green*gfinal
	finalh =  redfh + bluefh + greenfh
	finalv = redfv + bluefv + greenfv
	finalm = redm + bluem + greenm
	UporDown = (finalv > 0 ).astype(int)
	LeftorRight = 2*(finalh > 0).astype(int)
	absh = map(abs, finalh)
	absv = map(abs, finalv)
	absv[:] = [x*1.732 for x in absv]
	absh = np.float32(absh)
	absv = np.float32(absv)
	high = 4*(absv > absh).astype(int)
	out = high + LeftorRight + UporDown
	features = []
	for x in range(6):
		hrt = np.zeros(out.shape[:2],np.uint8)
		features.append(hrt)
	kernelg = np.ones((6,6),np.float32)
	for x in range(out.shape[:2][0]):
		for y in range(out.shape[:2][1]):
			z = out[x][y]
			if z == 4 or z == 6:
#				print "a",z
				features[4][x][y] = finalm[x][y]
			elif z == 5 or z == 7:
				features[5][x][y] = finalm[x][y]
#				print "b",z
			else:
				features[z][x][y] = finalm[x][y]
#				print z
	lastFeatures = []	
	for img in features:
		#img1 =  cv2.filter2D(img,-1,kernelg)
		lastFeatures.append(img)
	lastFeatures.append(finalm)
	lastFeatures.append(Y)
	lastFeatures.append(U)
	lastFeatures.append(V)

	integrals = []
	index = 0
	files = "yyy/usefel"
	ind = 0
#	for img in lastFeatures:
#		integ,waste,tilted = cv2.integral3(img)
#		integrals.append(integ)
#		cv2.imwrite("window" + str(ind) + '.jpg',img)
#		cv2.imshow("window.jpg",img)
#		cv2.waitKey(0)
#		cv2.destroyAllWindows()
#		ind += 1
#	print  integ.shape[:2]
#	height, width = integ.shape[:2]
#	a = readFeatures(height-1,width-1)
#	results = []
#	results = Evaluate(a, integrals,rows,cols)
#	start = time.time()
#	for x in range(0,rows - 22,10):
#		for y in range(0,cols - 22,10):
#			h = Evaluate(a, integrals,x,y)
#			h = np.lib.pad(h, (0,5000 - len(h)), 'constant', constant_values=(0))
#			r = bdt.predict(h)
#			an = [r,x,y]
#3			print an
#			results.append(an)
#	end = time.time()
#	print end -start
#	return results
	return lastFeatures


def testResize():
#	files = glob.glob('*.ppm')
	files = ['00000_00002.ppm']
	filed = 0
	totals = np.zeros((19),np.float32)
	for filer in files:
		print filed
		if filed > 1000:
			break
		img = cv2.imread(filer,-1)
		results = [] 
		r = run(img)
		for x in np.arange(0.1,2.0,0.1):
			img1 = cv2.resize(img,(0,0),fx = x,fy = x)
			l = np.array(run(img1)).astype(np.float32)
			new = []
			for im in r :
				cupper = cv2.resize(im,(0,0),fx = x,fy  =x).astype(np.float32)
				new.append(cupper)
			new = np.array(new)
			size = img1.size*10
			out = new - l
			out = np.absolute(out)
			score = np.sum(out)
			results.append(score/size)
		filed += 1
		results = np.array(results)
		totals += results
	totals /=filed
	print totals
		


def runTest(img,a,rows,cols,size,step,bdt):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	Y,U,V = cv2.split(YUV)
	r,g,b = cv2.split(img)
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[0] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	dst = cv2.filter2D(img,cv2.CV_16S,kernel1)
	rv,gv,bv = cv2.split(dst)
	rv1 = np.int16(rv)
	gv1 = np.float32(gv)
	bv1 = np.float32(bv)
	rv1 = cv2.pow(rv1,2)
	gv1 = cv2.pow(gv1,2)
	bv1 = cv2.pow(bv1,2)
	dst = cv2.filter2D(img,cv2.CV_16S,kernel2)
	rh,gh,bh = cv2.split(dst)
	rh1 = np.int16(rh)
	gh1 = np.float32(gh)
	bh1 = np.float32(bh)
	rh1 = cv2.pow(rh1,2)
	gh1 = cv2.pow(gh1,2)
	bh1 = cv2.pow(bh1,2)
	r1 = rh1 + rv1
	g1 = gh1 + gv1
	b1 = bh1 + bv1
	r1 = np.float32(r1)
	g1 = np.float32(g1)
	b1 = np.float32(b1)
	rfinal = cv2.sqrt(r1).astype(np.uint8)
	bfinal = cv2.sqrt(g1).astype(np.uint8)
	gfinal = cv2.sqrt(b1).astype(np.uint8)
	red = (rfinal > bfinal) & (rfinal > gfinal)
	red = red.astype(np.uint8)
	blue = (bfinal > rfinal) & (bfinal > gfinal)
	blue = blue.astype(np.uint8)
	green = (gfinal > bfinal) & (gfinal > rfinal)
	green = green.astype(np.uint8)
	redfh = red*rh
	bluefh = blue*bh
	greenfh = green*gh
	redfv = red*rv
	bluefv = blue*bv
	greenfv = green*gv
	redm = red*rfinal
	bluem = blue*bfinal
	greenm = green*gfinal
	finalh =  redfh + bluefh + greenfh
	finalv = redfv + bluefv + greenfv
	finalm = redm + bluem + greenm
	UporDown = (finalv > 0 ).astype(int)
	LeftorRight = 2*(finalh > 0).astype(int)
	absh = map(abs, finalh)
	absv = map(abs, finalv)
	absv[:] = [x*1.732 for x in absv]
	absh = np.float32(absh)
	absv = np.float32(absv)
	high = 4*(absv > absh).astype(int)
	out = high + LeftorRight + UporDown
	features = []
	for x in range(6):
		hrt = np.zeros(out.shape[:2],np.uint8)
		features.append(hrt)
	kernelg = np.ones((6,6),np.float32)
	for x in range(out.shape[:2][0]):
		for y in range(out.shape[:2][1]):
			z = out[x][y]
			if z == 4 or z == 6:
#				print "a",z
				features[4][x][y] = finalm[x][y]
			elif z == 5 or z == 7:
				features[5][x][y] = finalm[x][y]
#				print "b",z
			else:
				features[z][x][y] = finalm[x][y]
#				print z
	lastFeatures = []	
	for img in features:
		img1 =  cv2.filter2D(img,-1,kernelg)
		lastFeatures.append(img1)
	lastFeatures.append(finalm)
	lastFeatures.append(Y)
	lastFeatures.append(U)
	lastFeatures.append(V)

	integrals = []
	for img in lastFeatures:
		integ = cv2.integral(img)
		integrals.append(integ)
	print  integ.shape[:2]
	height, width = integ.shape[:2]
#	a = readFeatures(height-1,width-1)
	results = []
#	results = Evaluate(a, integrals,rows,cols)
	start = time.time()
	for x in range(0,rows - size,step):
		for y in range(0,cols - size,step):
			h = Evaluate(a, integrals,x,y)
			h = np.lib.pad(h, (0,5000 - len(h)), 'constant', constant_values=(0))
			r = bdt.predict(h)
			an = [r,x,y]
			#print an
			results.append(an)
	end = time.time()
	print end -start
	return results




def ReducedRun(img):
	img = cv2.resize(img,(26,26))
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	Y,U,V = cv2.split(YUV)
	r,g,b = cv2.split(img)
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[0] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	dst = cv2.filter2D(img,cv2.CV_16S,kernel1)
	rv,gv,bv = cv2.split(dst)
	rv1 = np.int16(rv)
	gv1 = np.float32(gv)
	bv1 = np.float32(bv)
	rv1 = cv2.pow(rv1,2)
	gv1 = cv2.pow(gv1,2)
	bv1 = cv2.pow(bv1,2)
	dst = cv2.filter2D(img,cv2.CV_16S,kernel2)
	rh,gh,bh = cv2.split(dst)
	rh1 = np.int16(rh)
	gh1 = np.float32(gh)
	bh1 = np.float32(bh)
	rh1 = cv2.pow(rh1,2)
	gh1 = cv2.pow(gh1,2)
	bh1 = cv2.pow(bh1,2)
	r1 = rh1 + rv1
	g1 = gh1 + gv1
	b1 = bh1 + bv1
	r1 = np.float32(r1)
	g1 = np.float32(g1)
	b1 = np.float32(b1)
	rfinal = cv2.sqrt(r1).astype(np.uint8)
	bfinal = cv2.sqrt(g1).astype(np.uint8)
	gfinal = cv2.sqrt(b1).astype(np.uint8)
	red = (rfinal > bfinal) & (rfinal > gfinal)
	red = red.astype(np.uint8)
	blue = (bfinal > rfinal) & (bfinal > gfinal)
	blue = blue.astype(np.uint8)
	green = (gfinal > bfinal) & (gfinal > rfinal)
	green = green.astype(np.uint8)
	redfh = red*rh
	bluefh = blue*bh
	greenfh = green*gh
	redfv = red*rv
	bluefv = blue*bv
	greenfv = green*gv
	redm = red*rfinal
	bluem = blue*bfinal
	greenm = green*gfinal
	finalh =  redfh + bluefh + greenfh
	finalv = redfv + bluefv + greenfv
	finalm = redm + bluem + greenm
	UporDown = (finalv > 0 ).astype(int)
	LeftorRight = 2*(finalh > 0).astype(int)
	absh = map(abs, finalh)
	absv = map(abs, finalv)
	absv[:] = [x*1.732 for x in absv]
	absh = np.float32(absh)
	absv = np.float32(absv)
	high = 4*(absv > absh).astype(int)
	out = high + LeftorRight + UporDown
	features = []
	for x in range(6):
		hrt = np.zeros(out.shape[:2],np.uint8)
		features.append(hrt)
	for x in range(out.shape[:2][0]):
		for y in range(out.shape[:2][1]):
			z = out[x][y]
			if z == 4 or z == 6:
				features[4][x][y] = finalm[x][y]
			elif z == 5 or z == 7:
				features[5][x][y] = finalm[x][y]
			else:
				features[z][x][y] = finalm[x][y]
	lastFeatures = []	
	kernelg = np.ones((5,5),np.float32)
	for img in features:
		img1 =  cv2.filter2D(img,-1,kernelg)
		lastFeatures.append(img1)
	lastFeatures.append(finalm)
	lastFeatures.append(Y)
	lastFeatures.append(U)
	lastFeatures.append(V)

	return lastFeatures




def ClassFeatures(img):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(YUV,(26,26))
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[0] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	dst = cv2.filter2D(img,cv2.CV_16S,kernel1)
	dstv1 = np.int16(dst)
	dstv2 = cv2.pow(dstv1,2)
	dst = cv2.filter2D(img,cv2.CV_16S,kernel2)
	dsth1 = np.int16(dst)
	dsth2 = cv2.pow(dsth1,2)
	dst1 = dsth2 + dstv2
	dst1 = np.float32(dst1)
	dstfinal = cv2.sqrt(dst1).astype(np.uint8)
	finalh =  dsth1
	finalv = dstv1
	finalm = dstfinal
	UporDown = (finalv > 0 ).astype(int)
	LeftorRight = 2*(finalh > 0).astype(int)
	absh = map(abs, finalh)
	absv = map(abs, finalv)
	absv[:] = [x*1.732 for x in absv]
	absh = np.float32(absh)
	absv = np.float32(absv)
	high = 4*(absv > absh).astype(int)
	out = high + LeftorRight + UporDown
	features = []
	for x in range(6):
		hrt = np.zeros(out.shape[:2],np.uint8)
		features.append(hrt)
	kernelg = np.ones((6,6),np.float32)
	for x in range(out.shape[:2][0]):
		for y in range(out.shape[:2][1]):
			z = out[x][y]
			if z == 4 or z == 6:
#				print "a",z
				features[4][x][y] = finalm[x][y]
			elif z == 5 or z == 7:
				features[5][x][y] = finalm[x][y]
#				print "b",z
			else:
				features[z][x][y] = finalm[x][y]
#				print z
	lastFeatures = []	
	for img in features:
		img1 =  cv2.filter2D(img,-1,kernelg)
		lastFeatures.append(img1)
	lastFeatures.append(finalm)
	lastFeatures.append(Y)
	lastFeatures.append(U)
	lastFeatures.append(V)

	integrals = []
	for img in lastFeatures:
		integ = cv2.integral(img)
		integrals.append(integ)
	print  integ.shape[:2]
	height, width = integ.shape[:2]
#	a = readFeatures(height-1,width-1)
	results = []
	results = Evaluate(a, integrals,rows,cols)
#	for x in range(0,rows - 73,10):
#		for y in range(0,cols - 73,10):
#			h = Evaluate(a, integrals,x,y)
#			h = np.lib.pad(h, (0,5000 - len(h)), 'constant', constant_values=(0))
#			r = bdt.predict(h)
#			an = [r,x,y]
#			print an
#			results.append(an)
	return results








def load():
	images = []
	results = []
	index = 0
	samples = 0 
	files  = glob.glob('../50x50negatives/*')
#	print files
	b = readFeatures(21,21)
	for g in range(5):
		for h in range(10):
			for i in range(10):
				print "y",samples
				if samples > 5000:
					break
#				b = np.loadtxt(open("../000%d%d/GT-000%d%d.csv"%(g,h,g,h),"rb"),delimiter=";",usecols = (3,4,5,6),skiprows = 1)
				for j in range(4):
					for k in range(10):
						img = cv2.imread("../20x20/000%d%d/0000%d_000%d%d.ppm"%(g,h,i,j,k)) 
						if not img is None:
							a = run(img,b,0,0,0)
							images.append(a)
							results.append(1)
							samples += 1
	np.savetxt("../20x20/images20.csv",images,delimiter = ",")
	np.savetxt("../20x20/results20.csv",results,delimiter = ",")


	print samples
	for filer in files:
		print filer
		imag = cv2.imread(filer)
		print samples 
		if samples < 0:
			break
		print "y"
		for x in range(0,550,50):
			for y in range(0,550,50):
				img = imag[x:x + 25,y:y + 25]
				height,width = img.shape[:2]
				height /= 10
				width  /= 10
				if height == 2 and 2 == width:
					samples -= 1
					print img.shape[:2]
					a = run(img,b,0,0,0)
					images.append(a)
					results.append(0)

	np.savetxt("../20x20/imagenegs20.csv",images,delimiter = ",")
	np.savetxt("../20x20/resultsnegs20.csv",results,delimiter = ",")
#	images.extend(np.loadtxt(open("../images.csv","rb"),delimiter=","))
#	results.extend(np.loadtxt(open("../results.csv","rb"),delimiter = ","))
#	images.extend(np.loadtxt(open("../images.csv","rb"),delimiter=","))
#	results.extend(np.loadtxt(open("../results.csv","rb"),delimiter = ","))

	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME", n_estimators=50)
	bdt.fit(images, results)
	joblib.dump(bdt, '50estimators/adaboost20x20.pkl')



def Training():
	files  = glob.glob('../Allnegatives/*')
	for s in [20]:
		samples = 0
		images = []
		results = []
		images1 = []
		results1 = []
		b = readFeatures(s+1,s+1)		
		for g in range(5):
			for h in range(9):
#				b = np.loadtxt(open("../000%d%d/GT-000%d%d.csv"%(g,h,g,h),"rb"),delimiter=";",usecols = (3,4,5,6),skiprows = 1)
#				index = 0
				for i in range(10):
					for j in range(4):
						print "y",samples
						for k in range(10):
							img = cv2.imread("../%dx%d/000%d%d/0000%d_000%d%d.ppm"%(s,s,g,h,i,j,k)) 
							if not img is None:
								samples += 1
								r = run(img,b,0,0,0)
								images.append(r)
								results.append(1)



		np.savetxt("speed adaboost/%d/images.csv"%s,images,delimiter = ",")
		np.savetxt("speed adaboost/%d/results.csv"%s,results,delimiter = ",")
	
	
		print samples
		for filer in files:
			print filer
			imag = cv2.imread(filer)
			print samples 
			if samples < 0:
				break
			print "y"
			for x in range(0,550,s):
				for y in range(0,550,s):
					img = imag[x:x +s + 5,y:y + s + 5]
					height,width = img.shape[:2]
					height /= 10
					width  /= 10
					if height*10 == s and s == 10*width:
						samples -= 1
						print img.shape[:2]
						a = run(img,b,0,0,0)
						images1.append(a)
						results1.append(0)
	
		np.savetxt("speed adaboost/%d/imagesnegs.csv"%s,images,delimiter = ",")
		np.savetxt("speed adaboost/%d/resultsnegs.csv"%s,results,delimiter = ",")
#	images.extend(np.loadtxt(open("../images.csv","rb"),delimiter=","))
#	results.extend(np.loadtxt(open("../results.csv","rb"),delimiter = ","))
#	images.extend(np.loadtxt(open("../images.csv","rb"),delimiter=","))
#	results.extend(np.loadtxt(open("../results.csv","rb"),delimiter = ","))
		images.extend(images1)
		results.extend(results1)
	
		bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME", n_estimators=50)
		bdt.fit(images, results)
		joblib.dump(bdt, 'speed adaboost/%d/adaboost.pkl'%s)
	









def testing():
	for x in range(10):
		print "0000%d"%x
	for x in range(10,43):
		print "000%d"%x


def train():
	images = []
	results = []
	images.extend(np.loadtxt(open("../images70.csv","rb"),delimiter=","))
	results.extend(np.loadtxt(open("../results70.csv","rb"),delimiter = ","))
	images.extend(np.loadtxt(open("../imagenegs70.csv","rb"),delimiter=","))
	results.extend(np.loadtxt(open("../resultsnegs70.csv","rb"),delimiter = ","))

	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME", n_estimators=50)
	bdt.fit(images, results)
	joblib.dump(bdt, '50estimators/adaboost70x70.pkl')

	

def shows():
	for x in range(30,110,10):
		show(x)
		print x


def show(a):
	bdt = joblib.load('speed adaboost/%d/adaboost.pkl'%a)
	b = readFeatures(a + 1,a + 1)
	features = []
	group = {}
	for clf in bdt.estimators_:
		for x in range(len(clf.tree_.feature)):
			h  = clf.tree_.feature[x]
			if h >=0:
				if h in group:
					clf.tree_.feature[x] = group[h]
				else:
					clf.tree_.feature[x] = len(features)
					group[h] = len(features)
					features.append(b[h])
	np.savetxt("../csvs/some%d%d.csv"%(a,a),features,delimiter = ",")
	joblib.dump(bdt,'speed adaboost/%d/adaboost.pkl'%a)


def test():
	a = [[1,2,3],[4,5,6],[7,8,9]]
	for x in a:
		x[0] = 4
	print a

def testFun(a):
	images = []
	results = []
	samples = 0
	bdt = joblib.load('speed adaboost/%d/adaboost.pkl'%a)
#	for h in range(9):
#		for i in range(10):
#			for j in range(4):
#				print "y",samples
#				if samples > 5000:
#					break
#				for k in range(10):
#					img1 = cv2.imread("../TS2010/0000%d_000%d%d.ppm"%(h,i,j)) 
##					if not img1 is None:
#						samples += 1
#						images.append(img1)

	img = cv2.imread("../TS2010/SourceImages/_0004.bmp")
	images.append(img)
	start = time.time()
	b = readFeatures(a+ 1,a+1)
	for fraction in [7,6]:
		results = []
		for image in images:
			img = cv2.resize(image,(0,0),fx = 2.0/fraction,fy = 2.0/fraction)
			height,width = img.shape[:2]
			r = runTest(img,b,height,width,a+1,10,bdt)
			results.append(r)
			positive = 0 
			for t in results:
				for u in t:
					if u[0][0] == 1:
						positive += 1
						cv2.rectangle(img,(u[2]+4,u[1]),(u[2]+4 + a,u[1] + a),(255,255,255),2)
				#print "positives: ",positive
				cv2.imwrite("heller.jpg",img)
				ht = cv2.waitKey()
				cv2.destroyAllWindows()
	end = time.time()
	print end - start


def RunImage():
	start_time = time.time()
	image = cv2.imread("../1.jpg") 
	hheight, wwidth = image.shape[:2]
	results = []
	b = readFeatures(51,51)
	a = run(image,b,hheight,wwidth)
	print "time: %s seconds "%(time.time()  - start_time)	



def loadandcheck():
	maxheight = 0
	maxwidth = 0
	minheight = 1000
	minwidth = 1000
	for g in range(5):
		for h in range(10):
			for i in range(6):
				for j in range(4):
					for k in range(10):
						img = cv2.imread("../000%d%d/0000%d_000%d%d.ppm"%(g,h,i,j,k)) #specify filename and the extension
						if not img is None:
							height, width = img.shape[:2]
							if height > maxheight:
								maxheight = height							
							if width > maxwidth:
								maxwidth = width
							if height < minheight:
								minheight = height							
							if width < minwidth:
								minwidth = width 
#							cv2.imshow("window",img)
#							cv2.waitKey()
	maxheight /= 10
	maxwidth /= 10
	minheight /= 10
	minwidth /= 10
	print maxheight
	print maxwidth 
	print minheight
	print minwidth 
	for x in range(minheight,maxheight+1):
		for y in range(minwidth,maxwidth+1):
			createFeatures(x,y)
	sizearray = np.zeros((maxheight + 1,maxwidth + 1),int)
#	print sizearray
	for g in range(5):
		for h in range(10):
			for i in range(6):
				for j in range(4):
					for k in range(10):
						img = cv2.imread("../000%d%d/0000%d_000%d%d.ppm"%(g,h,i,j,k)) #specify filename and the extension
						if not img is None:
							height, width = img.shape[:2]
							height /= 10
							width /= 10
							sizearray[height,width] +=1
#	print sizearray




def test_set():
	a = np.zeros((12630))
#	for g in range(5):
#		for h in range(10):
#			img = cv2.imread("../000%d%d/00000_00029.ppm"%(g,h)) #specify filename and the extension
#			if not img is None:
#				print cv2.imwrite("../%d%d.ppm"%(g,h),img)
	files = glob.glob('../../../../GTSRB/Final_Test/Images/*')
	for filer in files:
		img = cv2.imread(filer) #specify filename and the extension
		cv2.imshow("img",img)
		h = cv2.waitKey(0)
		g = cv2.waitKey(0)
		cv2.destroyAllWindows()
		a[int(filer[-9:-4])] = h*10 + g
	np.savetxt("../results.csv",a,delimiter = ",")

def readFile():
	with open('/home/keith/vision/Dataset/GTSRB (2)/Final_Training/Images/TS2010/SourceImages/sources1.log') as f:
    		lines = f.readlines()
	dict = {}
	count = {}
	for line in lines:
		signs = []
		yup = []
		for x in range(len(line)):
			if line[x] == '(':
				for y in range(len(line[x:])):
					if line[x +y] == ')':
						signs.append(line[x+1:x + y])
						pre = line[x-4:x-1]
						if pre[2] == '&' :
							count[yup] = count.get(yup, 0) + 1
						else:
							count[pre] = count.get(pre, 0) + 1
						yup = pre
						break
		dict[line[:12]] = signs
	return dict


def CreateBlanks():
	d = readFile()
	files = '/home/keith/vision/Dataset/GTSRB (2)/Final_Training/Images/TS2010/SourceImages/'
	destfiles = '/home/keith/vision/Dataset/GTSRB (2)/Final_Training/Images/TS2010/CutImages/'
	for key, value in d.items():
		filer = files + key 
		img = cv2.imread(filer,-1)
		if img is not None:
			for vals in value:
				vals = [x for x in vals if x not in ['x','=',',']]
				print vals
				for i in range(len(vals)):
					if vals[i] == 'y':
						xs = int(''.join(vals[:i]))
						print xs
						for j in range(i, len(vals),1):
							if vals[j] == 'w':
								ys = int(''.join(vals[i+1:j]))
								print ys
								for k in range(j,len(vals),1):
									if vals[k] == 'h':
										ws = int(''.join(vals[j+1:k]))
										hs = int(''.join(vals[k+1:]))
										print ws
										print hs
				
				print vals
				if ys <0 or xs < 0:
					break
				img[ys:ys + hs,xs:xs + ws,:] = np.zeros((min(hs,576-ys),min(ws,720 - xs),3),np.uint8)
			filerdst = destfiles + key
			cv2.imwrite(filerdst,img)


def createTrains():
	files = glob.glob('/home/keith/vision/Dataset/GTSRB (2)/Final_Training/Images/TS2010/CutImages/*.bmp')
	dst = '/home/keith/vision/Dataset/GTSRB (2)/Final_Training/Images/TS2010/trains/'
	samples = 0
	for t in range(3):
		for filer in files:
			img = cv2.imread(filer,-1)
			dester = dst + str(samples) + '.bmp'
			x = random.random()
			x = int(x*400)
			y = random.random()
			y = int(y*400)
			img1 = img[x:x + 70,y:y  +70] 
			cv2.imwrite(dester,img1)
			samples += 1
	

def L2Normalize(x):
	v = max(x)
	y = np.array(x)
	y.remove(v)
	v2 = max(y)
	knorm = v+ v2
	under = math.sqrt(math.pow(knorm,2) + 0.001)
	print  x/under
	return x/under

def L1Normalize(x):
	print x
	v = max(x)
	x = np.array(x)
	under = math.sqrt(math.pow(v,2) + 0.001)
	print  x/under
	return x/under
	
	
	
def HOG(img,angleBins = 8):
#	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	img = cv2.resize(img,(40,40))
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[2] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	print img 
	dsth = cv2.filter2D(img,cv2.CV_16S,kernel1).astype(np.float32)
	dstv = cv2.filter2D(img,cv2.CV_16S,kernel2).astype(np.float32)
	print dsth
	Mag,direction = cv2.cartToPolar(dsth,dstv,angleInDegrees = 1)
	print Mag
	final =  np.zeros((img.shape[0],img.shape[1],2),np.float32)
	for rows in range(len(Mag)):
		for cols in range(len(Mag[rows])):
			final[rows,cols][0] = max(Mag[rows,cols])
			final[rows,cols][1] = direction[rows,cols,Mag[rows,cols].tolist().index(max(Mag[rows,cols]))]			
	print final
	Hists = []
	for row in final:
		histCol = []
		for col in row:
			hist = [0]*angleBins
			index = int(col[1]/(360/angleBins))
			angle = col[1] - index*(360/angleBins)	
			try:
				i = index%angleBins
			except:
				 i = 0
			hist[i] = ( angle/(360/angleBins))*col[0]
			hist[angleBins%(index + 1)] = (1 - ( angle/(360/angleBins)))*col[0]
			histCol.append(hist)
		Hists.append(histCol)
	histos = np.array(Hists).astype(np.float32)
	print histos
#	histos = np.swapaxes(hists,0,2)
#	print hists
#	print histos
	kernel = np.ones((5,5),np.float32)
	histos = cv2.filter2D(histos,cv2.CV_32FC1,kernel).astype(np.float32)
	histos = histos[::5,::5,:]
	final = []
	for x in range(len(histos) - 1):
		row = []
		for y in range(len(histos[0]) - 1):
			lister = []
			lister.extend(histos[x,y])
			lister.extend(histos[x + 1,y])
			lister.extend(histos[x,y + 1])
			lister.extend(histos[x + 1,y + 1])
			row.append(lister)
		final.append(row)
	print histos
#	final = np.array(final)
	print final
#	print final.shape
	histos = [L2Normalize(x) for row in final for x in row]
	histos = [x for row in histos for x in row ]
	print histos
	print np.array(histos).shape
	return histos



	

if __name__ == "__main__":
	load()
