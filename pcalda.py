import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import glob
import os
import math

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



images1 = []
images2 = []
images3 = []

def L2Normalize(x):
	v = max(x)
	f = np.array(x)
	x.remove(v)
	v2 = max(x)
	knorm = v+ v2
	under = math.sqrt(math.pow(knorm,2) + 0.001)
	return f/under



def L2HysNormalize(x):
	v = max(x)
	f = np.array(x)
	x.remove(v)
	v2 = max(x)
	knorm = v+ v2
	under = math.sqrt(math.pow(knorm,2) + 0.001)
	inter =  f/under
	inter = [min(x,0.2) for x in inter]
	return L2Normalize(inter)	




def L1Normalize(x):
	print x
	v = max(x)
	x = np.array(x)
	under = math.sqrt(math.pow(v,2) + 0.001)
	return x/under
	
	
	
def HOG(img,angleBins = 8):
#	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	img = cv2.resize(img,(40,40))
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[2] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
#	print img 
	dsth = cv2.filter2D(img,cv2.CV_16S,kernel1).astype(np.float32)
	dstv = cv2.filter2D(img,cv2.CV_16S,kernel2).astype(np.float32)
#	print dsth
	Mag,direction = cv2.cartToPolar(dsth,dstv,angleInDegrees = 1)
#	print Mag
	final =  np.zeros((img.shape[0],img.shape[1],2),np.float32)
	for rows in range(len(Mag)):
		for cols in range(len(Mag[rows])):
			final[rows,cols][0] = max(Mag[rows,cols])
			final[rows,cols][1] = direction[rows,cols,Mag[rows,cols].tolist().index(max(Mag[rows,cols]))]			
#	print final
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
#	print histos
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
#	print histos
#	final = np.array(final)
#	print final
#	print final.shape
	histos = [L2HysNormalize(x) for row in final for x in row]
	histos = [x for row in histos for x in row ]
#	print histos
#	print np.array(histos).shape
	return histos


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

def readFeatures(height,width):
	a = np.loadtxt(open("../csvs/some%d%d.csv"%(height,width),"rb"),delimiter=",")
	return a.astype(int)


'''
def HOG(img):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	YUV = cv2.resize(YUV,(28,28))
	Y,U,V = cv2.split(YUV)
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

	kernelg1 = 0.125*np.ones((4,4),np.float32)
	kernelg2 = 0.25*np.ones((2,2),np.float32)
	lastFeatures = []	
	for img in features:
		cv2.imwrite("window1.jpg",img)
		img1 =  cv2.filter2D(img,-1,kernelg1)
		cv2.imwrite("window2.jpg",img1)
		img2 = img1[::4,::4]
		cv2.imwrite("window3.jpg",img2)
		img3 = cv2.filter2D(img2,-1,kernelg2)
		cv2.imwrite("window4.jpg",img3)
		lastFeatures.append(img3)
		cv2.waitKey()
		cv2.destroyAllWindows()
	return lastFeatures
'''

def Pyramid(img):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	YUV = cv2.resize(YUV,(40,40))
	Y,U,V = cv2.split(YUV)
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
	kernelg1 = 0.125*np.ones((4,4),np.float32)
	kernelg2 = 0.25*np.ones((2,2),np.float32)
	lastFeatures = []	
	for img in features:
		tote = cv2.sumElems(img)
		tote = tote/img.size
		img = img/tote
		print img
		print cv2.sumElems(img)
		print img.size
		lastFeatures.append(img1)
	return lastFeatures


def Pyramid1(img):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	YUV = cv2.resize(YUV,(40,40))
	Y,U,V = cv2.split(YUV)
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(YUV,(26,26))
	kernel1 = np.ones((3,1),np.float32)
	kernel2 = np.ones((1,3),np.float32)
	kernel1[0] = -1
	kernel1[1] = 0
	kernel2[0] = [-1,0,1]
	dsth = cv2.filter2D(img,cv2.CV_16S,kernel1)
	dstv = cv2.filter2D(img,cv2.CV_16S,kernel2)
	Mag,direction = cv2.cartToPolar(dsth,dstv,angleInDegrees)
	print Direction
#	kernelg1 = 0.125*np.ones((4,4),np.float32)
#	kernelg2 = 0.25*np.ones((2,2),np.float32)
#	lastFeatures = []	
#	for img in features:
#		tote = cv2.sumElems(img)
#		tote = tote/img.size
#		img = img/tote
#		print img
#		print cv2.sumElems(img)
#		print img.size
#		lastFeatures.append(img1)
#	return lastFeatures


def RunC(img1):
	img = cv2.resize(img1,(26,26))
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
	for img in lastFeatures:
		integ = cv2.integral(img)
		integrals.append(integ)
	print  integ.shape[:2]
	height, width = integ.shape[:2]
	a = readFeatures(26,26)
	results = Evaluate(a, integrals,0,0)
	return results


def RunG(img):
	YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	YUV = cv2.resize(YUV,(26,26))
	Y,U,V = cv2.split(YUV)
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
	a = readFeatures(26,26)
	results = []
	results = Evaluate(a, integrals,0,0)
	return results


def ReducedRun(img):
	img = cv2.resize(img,(40,40))
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


def CreateResizedSamples():
	yes = 0
	addr = []
#	addr.append("/home/keith/vision/Dataset/GTSRB (4)/Final_Training/HOG/HOG_01")
	addr.append("/home/keith/vision/Dataset/GTSRB (4)/Final_Training/HOG/HOG_02")
#	addr.append("/home/keith/vision/Dataset/GTSRB (4)/Final_Training/HOG/HOG_03")
	for address in addr:
		directory = os.listdir(address)
		AllImages = []
		AllResults = []
		for folder in directory:
			images = []
			results = []
			thefolder = address + '/'+ folder + '/*.txt'
			files = glob.glob(thefolder)
			for filer in files:
#				img1 = cv2.imread(filer) 
#				if not img1 is None:		
				a = np.loadtxt(open(filer,"rb"),delimiter=",")
#				print filer
#				print a
#				x = [item    for  sublist in img1 for item in sublist ]
				sign = int(folder)
#				img2 = ReducedRun(img1)
#				img3 = RunG(img1)
#				img4 = RunC(img1)
				images.append(a)
#				images1.append(img3)
#				images2.append(img4)
#				images3.append(x)
				results.append(sign)
			print yes
			yes += 1
		AllImages.append(images)
		AllResults.append(results)		
	return AllImages,AllResults


def Train():
	yes = 0
	directory = os.listdir("../cropped")
	print directory
	images = []
	results = []
	for folder in directory:
		thefolder =   '../cropped/'+ folder + '/*.ppm'
		files = glob.glob(thefolder)
		for filer in files:
			img1 = cv2.imread(filer)
			if not img1 is None:
				a = HOG(img1)
#				x = [item    for  sublist in img1 for item in sublist ]
				sign = int(folder)
#				img2 = ReducedRun(img1)
#				img3 = RunG(img1)
#				img4 = RunC(img1)
				images.append(a)
#				images1.append(img3)
#				images2.append(img4)
#				images3.append(x)
				results.append(sign)	
	return images,results


def Test():
	b = np.loadtxt(open('../../../../GTSRB Test/GT-final_test.csv',"rb"),delimiter=";",usecols = (6,7),skiprows = 1)
	corrects = [item[1] for item in b]
	images = []
	files = glob.glob('../../../../GTSRB Test/Final_Test/Images/*')
	for filer in files:
		img1 = cv2.imread(filer)
		if not img1 is None:
			a = HOG(img1)
			images.append(a)
	return images,corrects




def LoademUp():
	X = []
	y = []
	files = glob.glob('../../../../GTSRB (3)/Final_Test/HOG/HOG_02/*')
	b = np.loadtxt(open('../../../../GTSRB Test/GT-final_test.csv',"rb"),delimiter=";",usecols = (6,7),skiprows = 1)
	for filer in files:
		x = int(filer[-9:-4])
		result = b[x]
		a = np.loadtxt(open(filer,"rb"),delimiter=",")
		X.append(a)
		y.append(result[1])
	return X,y
		


#X,y = Train()
#np.savetxt("../csvs/MyHOGHys2.csv",X,delimiter = ",")
#np.savetxt("../csvs/MyHOGHys2Classes.csv",y,delimiter = ",")

#lda  = joblib.load('Trained Classifiers/MyldaHOG2.pkl')
#joblib.dump(lda, 'Trained Classifiers/MyldaHOG2.pkl')



filer = 'Trained Classifiers/My'
#X = np.loadtxt(open("../csvs/MyHOGHys2.csv","rb"),delimiter=",")
#y = np.loadtxt(open("../csvs/MyHOGHys2Classes.csv","rb"),delimiter=",")

#models = glob.glob('Trained Classifiers/*.pkl')
#print models 
#print X.shape
#print y.shape



'''

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
	for x in [-3,-2,-1,0,1,2,3,4,5,6,7]:
		c = math.pow(10,x)
		for d in [-9,-8,-7,-6,-5,-4,-3,-3,-1,0,1,2,3,4]:
			gam = math.pow(10,d)
			filed = filer + kernel + 'gamma'+str(gam) + 'C' + str(c) + '.pkl'
			print filed
			if filed not in models:
				clf = SVC(kernel=kernel, C= c,gamma= gam)
				clf.fit(X, y)
				joblib.dump(clf, filed)
				print "c: ",c
				print "gamma: ",gam



X,y = Test()
np.savetxt("../csvs/MyHOGHys2final.csv",X,delimiter = ",")
np.savetxt("../csvs/MyHOGHys2Classesfinal.csv",y,delimiter = ",")
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
	final = []
	for x in [-3,-2,-1,0,1,2,3,4,5,6,7]:
		c = math.pow(10,x)
		Clist = []
		for d in [-9,-8,-7,-6,-5,-4,-3,-3,-1,0,1,2,3,4]:
			gam = math.pow(10,d)
			filed = filer + kernel + 'gamma'+str(gam) + 'C' + str(c) + '.pkl'
			print filed
			clf = joblib.load(filed)
			this1 = clf.score(X, y)
			print this1
			Clist.append(this1)
			print "c: ",c
			print "gamma: ",gam
		final.append(Clist)
		finaler = np.array(final)
		np.savetxt("../csvs/" + kernel + '.csv',finaler,delimiter = ",")

'''


X,y = CreateResizedSamples()
print y
print X
iris = datasets.load_iris()



target_names = iris.target_names

#pca = PCA(n_components=150)
#pca  = joblib.load('Trained Classifiers/pca150.pkl')
#X_r = pca.fit(X).transform(X)
#print X_r
#joblib.dump(pca, 'Trained Classifiers/pca150.pkl')



#X,y = LoademUp()
'''
print "yes"
pca = PCA(n_components=2)
joblib.dump(pca, 'Trained Classifiers/pcahog2.pkl')
X_r = pca.fit(X).transform(X)
print X_r
'''


lda = LDA(n_components = 2)
joblib.dump(lda, 'Trained Classifiers/ldahog2.pkl')
X_r2 = pca.fit(X).transform(X)
print X_r2
#lda  = joblib.load('Trained Classifiers/MyldaHOG3.pkl')


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rgb", [[0, 1, 2,3,4,5,6,7,8],[33,34,35,36,37,38,39,40],[18,19,20,21,22,23,24,25,26,27,28,29,30,31]], ['speed','triangles','blue']):
    plt.scatter(X_r[y in i, 0], X_r[y in i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.figure()
for c, i, target_name in zip("rgb", [[0, 1, 2,3,4,5,6,7,8],[33,34,35,36,37,38,39,40],[18,19,20,21,22,23,24,25,26,27,28,29,30,31]], ['speed','triangles','blue']):
    plt.scatter(X_r2[y  in i, 0], X_r2[y in i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')


plt.show()

'''

#lda1 = LDA()
#print "lda1: ",images1
#lda1.fit(images1,y)

#lda2 = LDA()
#lda2.fit(images2,y)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

nbrsX5PCA = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X_r,y)
nbrsX5LDA = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X_r2,y)
joblib.dump(nbrsX5PCA, 'Trained Classifiers/NN5pca150.pkl')
joblib.dump(nbrsX5LDA, 'Trained Classifiers/NN5lda150.pkl')

nbrsX7PCA = neighbors.KNeighborsClassifier(n_neighbors=7).fit(X_r,y)
nbrsX7LDA = neighbors.KNeighborsClassifier(n_neighbors=7).fit(X_r2,y)
joblib.dump(nbrsX7PCA, 'Trained Classifiers/NN7pca150.pkl')
joblib.dump(nbrsX7LDA, 'Trained Classifiers/NN7lda150.pkl')

nbrsX9PCA = neighbors.KNeighborsClassifier(n_neighbors=9).fit(X_r,y)
nbrsX9LDA = neighbors.KNeighborsClassifier(n_neighbors=9).fit(X_r2,y)
joblib.dump(nbrsX9PCA, 'Trained Classifiers/NN9pca150.pkl')
joblib.dump(nbrsX9LDA, 'Trained Classifiers/NN9lda150.pkl')

nbrsX11PCA = neighbors.KNeighborsClassifier(n_neighbors=11).fit(X_r,y)
nbrsX11LDA = neighbors.KNeighborsClassifier(n_neighbors=11).fit(X_r2,y)
joblib.dump(nbrsX11PCA, 'Trained Classifiers/NN11pca150.pkl')
joblib.dump(nbrsX11LDA, 'Trained Classifiers/NN11lda150.pkl')



rfcX = RandomForestClassifier(max_depth=6, n_estimators=3000, max_features=5).fit(X,y)


svcX =   SVC(kernel="linear", C=0.025).fit(X_,y)
svcX =   SVC(gamma=2, C=1,degree = 8).fit(X,y)

nbrs1 = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X,y)
rfc1 = RandomForestClassifier(max_depth=6, n_estimators=300, max_features=1).fit(X,y)


svc11 =   SVC(kernel="linear", C=0.025).fit(X,y)
svc12 =   SVC(gamma=2, C=1,degree = 8).fit(X,y)



nbrs2 = neighbors.KNeighborsClassifier(n_neighbors=5).fit(images1,y)
rfc2 = RandomForestClassifier(max_depth=6, n_estimators=300, max_features=1).fit(images1,y)


svc21 =   SVC(kernel="linear", C=0.025).fit(images1,y)
svc22 =   SVC(gamma=2, C=1,degree = 8).fit(images1,y)





nbrs3 = neighbors.KNeighborsClassifier(n_neighbors=5).fit(images2,y)
rfc3 = RandomForestClassifier(max_depth=6, n_estimators=300, max_features=1).fit(images2,y)


svc31 =   SVC(kernel="linear", C=0.025).fit(images2,y)
svc32 =   SVC(gamma=2, C=1,degree = 8).fit(images2,y)

Fact = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#a = np.zeros((12630))
a = [[] for x in range(12630)]
#	for g in range(5):
#		for h in range(10):
#			img = cv2.imread("../000%d%d/00000_00029.ppm"%(g,h)) #specify filename and the extension
#			if not img is None:
#				print cv2.imwrite("../%d%d.ppm"%(g,h),img)



#files = glob.glob('../../../../GTSRB (3)/Final_Test/HOG/HOG_03/*')
b = np.loadtxt(open('../../../../GTSRB Test/GT-final_test.csv',"rb"),delimiter=";",usecols = (3,4,5,6,7),skiprows = 1)
files = glob.glob('../../../../GTSRB Test/Final_Test/Images/*.ppm')
#b = np.loadtxt(open('../../../../GTSRB Test/GT-final_test.csv',"rb"),delimiter=";",usecols = (3,4,5,6,7),skiprows = 1)
done = 0.0
#a = np.loadtxt(open("../csvs/hello2.csv","rb"),delimiter=",")
Folder = "Fails/"
esc = 0
ignores = 0
tests = []
for filer in files:
	print filer
	img = cv2.imread(filer) #specify filename and the extension
	thes = int(filer[-9:-4])
	print thes
	rank =  b[thes]		
	print rank
	img = img[int(rank[1]):int(rank[3]),int(rank[0]):int(rank[2])]
	x = HOG(img)
#	img1 = ReducedRun(img)
#	x = [item   for l in img1 for  sublist in l for item in sublist ]
#	x = np.loadtxt(open(filer,"rb"),delimiter=",")
	a[int(filer[-9:-4])]= x


predictions = lda.predict(a)
#corrects = [item[1] for item in b]

np.savetxt("../csvs/predictionsMyLdaHysHOG2.csv",predictions,delimiter = ",")

# Compute confusion matrix
cm = confusion_matrix(corrects, predictions)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
print "correct: ", np.trace(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()

for filer in files:
	img = cv2.imread(filer) #specify filename and the extension
	if not img is None: 
		thes = int(filer[-9:-4])
		print thes
#		rank =  b[thes]		
#		img = img[int(rank[1]):int(rank[3]),int(rank[0]):int(rank[2])]
#		img1 = ReducedRun(img)
#		x = [item   for l in img1 for  sublist in l for item in sublist ]
		theFact = rank[4]
		print "The Fact: ",theFact
#		X = lda.transform(x)
#		thePrediction = int(lda.predict(x) ) 
		thePrediction = predictions[thes] 
		print "The Prediction: ", thePrediction
		r1 = thePrediction == theFact
#		a[thes] = thePrediction
#		r2 = theFact == int(nbrsX5.predict(X)  + 48.0)
#		r3 = theFact == int(nbrsX5.predict(X)+ 48.0)
#		r4 = theFact == int(nbrsX7.predict(X) + 48.0)
#		r5 = theFact == int(nbrsX7.predict(X) + 48.0) 
#
#		r6 = theFact == int(nbrsX9.predict(X) + 48.0)
#		r7 = theFact == int(nbrsX9.predict(X) + 48.0)
#
#		r8 = theFact == int(nbrsX11.predict(X)+ 48.0) 
#		r9 = theFact == int(nbrsX11.predict(X)+ 48.0) 
#		r2 = theFact ==  int(nbrs1.predict(x))
#		r3 = theFact ==  int(rfc1.predict(x))
#		r4 = theFact == int(svc11.predict(x))
#		r5 = theFact == int(svc12.predict(x))
#		x = RunG(img)	
#		r6 = theFact == int(lda1.predict(x)+48.0)
#		r7 = theFact == int(nbrs2.predict(x))
#		r8 = theFact == int(rfc2.predict(x))
#		r9 = theFact == int(svc21.predict(x))
#		r10 = theFact == int(svc22.predict(x))
#		x = RunC(img)
#		r11 = theFact == int(lda2.predict(x))
#		r12 = theFact == int(nbrs3.predict(x))
#		r13 = theFact == int(rfc3.predict(x))
#		r14 = theFact == int(svc31.predict(x))
#		r15 = theFact == int(svc32.predict(x))	
		Fact[1] += r1
#		Fact[2] += r2
#		Fact[3] += r3
#		Fact[4] += r4
#		Fact[5] += r5
#		Fact[6] += r6
#		Fact[7] += r7
#		Fact[8] += r8
#		Fact[9] += r9
#		Fact[10] += r10
#		Fact[11] += r11
#		Fact[12] += r12
#		Fact[13] += r13
#		Fact[14] += r14
#		Fact[15] += r15	
#		if r1 == False:
#			if theFact == int(57.0):
#				ignores += 1
#			else:
#				dest = Folder + filer[-9:]
#				cv2.imwrite(dest,img)
#		done += 1
#		print "done: ", done
#		if esc != 9:
#			print "Category Lda:     ",r1
#			print "Category NN:  ", r2
#			print "Category RF:  ", r3
#			print "Category SVCL:", r4
#			print "Category SVC: ", r5
#			x = RunG(img)	
#			print "Category Lda:     ",r6
#			print "Category NN:  ", r7
#			print "Category RF:  ",r8
#			print "Category SVCL:", r9
#			print "Category SVC: ", r10
#			x = RunC(img)
#			print "Category Lda:     ",r11
#			print "Category NN:  ", r12
#			print "Category RF:  ", r13
#			print "Category SVCL:", r14
#			print "Category SVC: ", r15	
#			cv2.imshow("img",img)
#			g = cv2.waitKey(0)
#			if g == 27:
#				esc = 9
#			cv2.destroyAllWindows()
		done +=1.0 
print "Done: ",done
print "array: ",Fact[1]
#print "Ignore: ",ignores
print fact[1]/done
np.savetxt("../csvs/hello4.csv",a,delimiter = ",")

from StringIO import StringIO
out = StringIO()
out = tree.export_graphviz(clf, out_file=out)
print out.getvalue()
'''

