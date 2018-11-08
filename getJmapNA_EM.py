from scipy.signal import convolve2d as conv2
import numpy as np
from __init__ import image_data
from dct import ibdct
from dependencies import bdct,dequantize
from main_files import h1period, jpeg_rec, EMperiod, LLR
from mat_fn import hist,idct2


# [LLRmap, LLRmap_s, q1table, k1e, k2e, alphatable] = getJmapNA_EM(image,ncomp,c2)
#
# detect and localize tampered areas in doubly compressed JPEG images
#
# Pysteg are required, available at: 
# http://www.ifs.schaathun.net/pysteg/starting.html#download-and-installation
#
# image: JPEG object from image_data function which is in __init__.py file
# ncomp: index of color component (1 = Y, 2 = Cb, 3 = Cr)
# c2: number of DCT coefficients to consider (1 <= c2 <= 64)
#
# LLRmap(:,:,c): estimated likelihood of being doubly compressed for each 8x8 image block
#   using standard model and c-th DCT frequency (zig-zag order)
# LLRmap_s(:,:,c): estimated likelihood of being doubly compressed for each 8x8 image block
#   using simplified model and c-th DCT frequency (zig-zag order)
# q1table: estimated quantization table of primary compression
# k1e, k2e: estimated shift of first compression
# alphatable: mixture parameter for each DCT frequency



def getJmapNA_EM(image,ncomp,c2):
	#DCT Coefficient array of a(ncomp) channel of image
	coeffArray = image['coef_arrays'][ncomp] 
	#Quantization table of image
	qtable = image['quant_tables'][image['comp_info'][ncomp]['quant_tbl_no']]
	#8*8 matrix with all entries 1
	q1table = np.ones((8,8))
	#Quality matrix 
	Q1up = np.array(
          [ [ 16,  11,  10,  16,  24,  40,  51,  61 ],
	    [ 12,  12,  14,  19,  26,  58,  60,  55 ],
	    [ 14,  13,  16,  24,  40,  57,  69,  56 ],
	    [ 14,  17,  22,  29,  51,  87,  80,  62 ],
	    [ 18,  22,  37,  56,  68, 109, 103,  77 ],
	    [ 24,  35,  55,  64,  81, 104, 113,  92 ],
	    [ 49,  64,  78,  87, 103, 121, 120, 101 ],
	    [ 72,  92,  95,  98, 112, 100, 103,  99 ] ])

	minQ = np.floor(qtable/np.sqrt(3))
	minQ[minQ<2]=2
	maxQ = np.maximum(Q1up, qtable)
	# estimate rounding and truncation error
	I,YCbCr = jpeg_rec(image)
	
	#alternate option for uint8(I) in matlab
	t_T = np.round(I.clip(min=0))
	t_T[t_T>255] = 255
	E =  I - t_T
	Edct = bdct(0.299 * E[:,:,0] +  0.587 * E[:,:,1] + 0.114 * E[:,:,2])
	I = ibdct(dequantize(coeffArray,qtable))
	#zigzag index of 8*8 matrix
	coeff=np.array([1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64])

	binHist = np.array(range(-2**11,2**11,1))
	center = 2**11
	B = np.ones((8,8))/float(8)
	#convolution of two 2-D array
	DC = conv2(I,B)
	DC = DC[7:,7:]
	EDC = Edct[::8,::8]
	#Sample variance of EDC
	varE = np.var(EDC.flatten(1),ddof=1)
	#Mean of EDC
	bias = np.mean(EDC.flatten(1))
	sig = np.sqrt(qtable[0,0]**2 / float(12) + varE)
	#zero matrix of size 8*8
	alphatable = np.ones((8,8))
	#c2 layer of zero matrix 
	LLRmap = np.zeros([np.size(coeffArray,0)/8 ,np.size(coeffArray,1)/8,c2 ])
	LLRmap_s = np.zeros([ np.size(coeffArray,0)/8 ,np.size(coeffArray,1)/8,c2])
	k1e = 1
	k2e = 1
	#value of Lmax is -infinity
	Lmax = -np.inf

	for k1 in range(1,9):
		for k2 in range(1,9):
			if (k1>1) or (k2>1):
				DCpoly = DC[k1-1::8,k2-1::8]
				# choose shift for estimating unquantized distribution
				#through calibration
				if k1 < 5:
					k1cal = k1 + 1
				else:
					k1cal = k1 - 1
				if k2 < 5:
					k2cal = k2 + 1
				else:
					k2cal = k2 - 1
				DCcal = DC[k1cal-1::8,k2cal-1::8]
				#create a histogram array of DCcal
				hcal =hist(DCcal.flatten(1),binHist)
				hcalnorm = (hcal+1)/float(np.size(DCcal)+np.size(binHist))
				#define a function p0 which take one input (x)
				p0 = lambda x: hcalnorm[(np.round(x)).astype(int)+ center]
				#define a function p1 which take two inputs x and Q
				p1 = lambda x,Q: h1period(x, Q, hcal, binHist, center, bias, sig)
				Q, alpha, L, ii = EMperiod(DCpoly.flatten(1), minQ[0,0], maxQ[0,0], 0.95, p0, p1, 5, 20)
				if L > Lmax:
					#simplified model
					#reshape 2-d array(DCpoly) to 1-d array column-wise
					t_DCp=DCpoly.flatten(1)
					#replace each nonzero element of array(t_DCp) by 1
					t_DCp[t_DCp!=0]=1
			        	nz = np.sum(t_DCp)/float(np.size(DCpoly))
					LLRmap_s[:,:,0]=LLR(DCpoly, binHist, nz, int(Q), int(np.round(bias)), int(center), sig)
					#taking log of all elements of array
					ppu = np.log(p1(binHist, Q)/p0(binHist).astype(float))
					LLRmap[:,:,0] = ppu[(np.round(DCpoly)).astype(int) + int(center)]
					q1table[0,0] = Q
        	        		alphatable[0,0] = alpha
        	        		k1e = k1
        	        		k2e = k2
        	        		Lmax = L
	for index in range(2,c2+1):  
		coe = coeff[index-1]
		ic1 = int(np.ceil(coe/float(8))) 
		ic2 = int(coe%8) 
		if ic2==0:
			ic2=8
		#zero matrix of 8*8 size
		A = np.zeros((8,8))
		A[ic1-1,ic2-1] = 1
		#inverse DCT of A
		B = idct2(A)
		#convolution of 2D array
		AC = conv2(I,B)
		#crop AC array from (7,7) position to end
		AC = AC[7:,7:]
		#slice AC array from (k1e-1,k2e-1) position with increment of 8
		ACpoly = AC[k1e-1::8,k2e-1::8]
		if k1e < 5:
			k1cal = k1e + 1
		else:
			k1cal = k1e - 1
		if k2e < 5:
			k2cal = k2e + 1
		else:
			k2cal = k2e - 1
		#slicing array(AC) with increment of 8
		ACcal = AC[k1cal-1::8,k2cal-1::8]
		#histogram array of ACcal
		hcal =hist(ACcal.flatten(1),binHist)
		hcalnorm = (hcal+1)/float(np.size(ACcal)+np.size(binHist))
		#slice array(Edct) from (ic1-1,ic2-1) position with increment of 8
		EAC = Edct[ic1-1::8,ic2-1::8]
		#sample variance of EAC
		varE = np.var(EAC.flatten(1),ddof=1)
		if index == 1:
			bias = np.mean(EAC.flatten(1))
		else:
			bias = 0
		sig = np.sqrt(qtable[ic1-1,ic2-1]**2 / float(12) + varE)
		#define a function p0 which take one input (x)
		p0 = lambda x: hcalnorm[(np.round(x)).astype(int)+ center]
		#define a function p1 which take two inputs x and Q
		p1 = lambda x,Q: h1period(x, Q, hcal, binHist, center, bias, sig)
		Q, alpha, L, ii = EMperiod(ACpoly.flatten(1), minQ[ic1-1,ic2-1], maxQ[ic1-1,ic2-1], 0.95, p0, p1, 5, 20)
		q1table[ic1-1,ic2-1] = Q
		alphatable[ic1-1,ic2-1] = alpha
		#reshape 2-d array(ACpoly) to 1-D array column-wise
		t_ACp=ACpoly.flatten(1)
		#replace each nonzero element of array(t_ACp) by 1
		t_ACp[t_ACp!=0]=1
		#compute sum of all elements of array(t_ACp) and divide it by size of ACpoly
		nz = np.sum(t_ACp)/float(np.size(ACpoly))
		LLRmap_s[:,:,index-1]=LLR(ACpoly, binHist, nz, int(Q), int(np.round(bias)), int(center), sig)
		ppu = np.log(p1(binHist, Q)/p0(binHist).astype(float))
		LLRmap[:,:,index-1] = ppu[(np.round(ACpoly)).astype(int) + int(center)]
						
	return (LLRmap ,LLRmap_s, q1table, k1e, k2e, alphatable)




