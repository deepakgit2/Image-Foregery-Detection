import numpy as np
from dct import ibdct
from __init__ import image_data
from main_files import jpeg_rec, EMperiod, h1periodDQ, ceil2, floor2
from dependencies import dequantize, bdct
from mat_fn import hist


# [LLRmap, LLRmap_s, q1table, alphatable] = getJmap_EM(image,ncomp,c2)
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
# alphatable: mixture parameter for each DCT frequency



def getJmap_EM(image,ncomp,c2):
	#DCT Coefficient array of a(ncomp) channel of image
	coeffArray = image['coef_arrays'][ncomp]
	#Quantization table of image
	qtable = image['quant_tables'][image['comp_info'][ncomp]['quant_tbl_no']]#np
	#8*8 matrix with all entries 1
	q1table = np.ones((8,8))
	alphatable = np.ones((8,8))
	#an array which contains c2 no. of zero matrix
	LLRmap = np.zeros([np.size(coeffArray,0)/8 ,np.size(coeffArray,1)/8,c2 ])
	LLRmap_s = np.zeros([ np.size(coeffArray,0)/8 ,np.size(coeffArray,1)/8,c2])
	
	# estimate rounding and truncation error
	I,YCbCr = jpeg_rec(image)

	#alternate option for uint8(I) in matlab
	t_T = np.round(I.clip(min=0))
	t_T[t_T>255] = 255
	E =  I - t_T
	Edct = bdct(0.299 * E[:,:,0] +  0.587 * E[:,:,1] + 0.114 * E[:,:,2])
	#Sample variance of EDC
	varE = np.var(Edct.flatten(1),ddof=1)
	
	# simulate coefficients without DQ effect
	Y = ibdct(dequantize(coeffArray,qtable))
	coeffArrayS = bdct(Y[1:,1:])

	#zig-zag index of 8*8 matrix
	coeff=np.array([1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64])
	#Quality matrix 
	Q1up = np.array(
          [ [ 16,  11,  10,  16,  24,  40,  51,  61 ],
	    [ 12,  12,  14,  19,  26,  58,  60,  55 ],
	    [ 14,  13,  16,  24,  40,  57,  69,  56 ],
	    [ 14,  17,  22,  29,  51,  87,  80,  62 ],
	    [ 18,  22,  37,  56,  68, 109, 103,  77 ],
	    [ 24,  35,  55,  64,  81, 104, 113,  92 ],
	    [ 49,  64,  78,  87, 103, 121, 120, 101 ],
	    [ 72,  92,  95,  98, 112, 100, 103,  99 ] ] )

	for index in range(c2): 
		coe = coeff[index]
		ic1 = int(np.ceil(coe/float(8))) 
		ic2 = int(coe%8) 
		if ic2==0:
			ic2=8
		#slice AC array from (ic1-1,ic2-1) position with increment of 8
		AC = coeffArray[ic1-1::8,ic2-1::8]
		try:
			ACsmooth = coeffArrayS[ic1-1::8,ic2-1::8]
		except:
			ACsmooth =np.array([])
		EAC = Edct[ic1-1::8,ic2-1::8]
		Q2 = qtable[ic1-1,ic2-1]
		center = int(np.floor(2**11/float(Q2)))
	
		# get histogram of DCT coefficients
		binHist = np.array(range(-center,center,1))

		# get histogram of DCT coeffs w/o DQ effect (prior model for uncompressed image)
		hsmooth0 = hist(ACsmooth.flatten(1), np.arange(-2**11,2**11))
		hsmooth0 = np.array([hsmooth0])
		#create histogram array of ACsmooth array
		hsmooth =hist(ACsmooth.flatten(1),binHist*Q2)
		
		#conert hsmooth array into numpy array
		hsmooth = np.array([hsmooth])
		hsmoothnorm = (hsmooth+1)/float(np.size(ACsmooth)+np.size(binHist))
		hsmoothnorm = hsmoothnorm[0]

		# get estimate of rounding/truncation error
		biasE = np.mean(EAC.flatten(1)) #mean of EAC
		sig = np.sqrt(varE) / Q2
		f = int(np.ceil(6*sig))
		p = np.array(range(-f,f+1))
		#exponential value of each element of array
		g = np.exp(-p**2/float(sig**2)/float(2))
		g = g/np.sum(g)

		if index == 0:
			bias = biasE
		else:
			bias = 0
		biasest = bias
		# define mixture components
		p0 = lambda x: hsmoothnorm[np.round(x) + center] #define a function p0 which take one input (x)
		#define a function p1 which take two inputs x and Q
		p1 = lambda x,Q: h1periodDQ(x, Q, Q2, hsmooth0, np.array(range(-2**11,2**11)), 2**11, bias, sig)
		# estimate parameters of first compression
		Q1, alpha,t_z,tzz = EMperiod(AC.flatten(1), 1, Q1up[ic1-1,ic2-1], 0.95, p0, p1, 5, 20)
		q1table[ic1-1,ic2-1] = Q1
		alphatable[ic1-1,ic2-1] = alpha
		if np.mod(Q2,Q1) > 0:
			#simplified model
			nhist = Q1/float(Q2) * (floor2((Q2/float(Q1))*(binHist + biasest/float(Q2) + 0.5)) - ceil2((Q2/float(Q1))*(binHist + biasest/float(Q2) - 0.5)) + 1)
			#convolution of two array
			nhist = np.convolve(g, nhist)
			if f==0:
				nhist = nhist[f:]
			else:
		        	nhist = nhist[f:-f]
		        N = np.size(AC) / float(np.size(binHist))
		        nhist = nhist*N + 1
			#taking log value of each elment of an array
		        ppu = np.log(nhist/float(np.mean(nhist)))
			#reshape 2-d array(AC) to 1-D array column-wise
			t_AC=AC.flatten(1)
			#replace each nonzero element of array(t_AC) by 1
			t_AC[t_AC!=0]=1
			#sum of all elements of array(t_AC) and divide by size of AC
		        nz = np.sum(t_AC)/float(np.size(AC))
		        ppu[center] = nz * ppu[center]

		        LLRmap_s[:,:,index] = ppu[AC + center]

		        # standard model
			ppu = np.log(p1(binHist, Q1)/p0(binHist).astype(float))
			LLRmap[:,:,index] = ppu[AC + center ]
	return (LLRmap, LLRmap_s, q1table, alphatable)


