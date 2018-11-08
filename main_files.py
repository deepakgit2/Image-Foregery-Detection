from numpy import *
import numpy as np
from dependencies import dequantize
from mat_fn import filt,filt2
from dct import ibdct
from scipy.signal import convolve2d as conv2

def ceil2(x1):
	tol = 1e-12
	x2 = np.ceil(x1) 
	idx = np.where(np.abs(x1-x2) < tol)	
	for i in idx:
		x2[i] = x1[i] +0.5 
	return x2


def floor2(x1):
	tol = 1e-12
	x2 = np.floor(x1) 
	idx = np.where(np.abs(x1-x2) < tol)	
	for i in idx:
		x2[i] = x1[i] - 0.5 
	return x2


# [L] = LLR(x, binHist, nz, Q, phase, center, sig)
#
# compute log-likelihood map according to simplified model
#
# x: DCT values
# binHist: bin positions
# nz: fraction of nonzero coefficients
# Q: quantization step 
# phase: mean of quantization/truncation noise after dequantization
# center: index of x = 0 in binHist
# sig: std of quantization/truncation noise


def LLR(x, binHist, nz, Q, phase, center, sig):
	# define Gaussian kernel
	w = int(ceil(3*sig))
	k = array(range(-w,w+1))
	g = exp(-k**2/float(sig**2)/float(2))
	g = g/float(np.sum(g))
	N = size(x) / float(size(binHist))
	
	bppm = zeros(binHist.shape)
	bppm[center + phase::Q] = Q
	bppm[center + phase:2:-Q] = Q
	bppm = convolve(g, bppm)
	
	if w==0:
		bppm = bppm[w:]
	else:
		bppm = bppm[w:-w]
	bppm = bppm*N + 1
	LLRmap = log(bppm / float(mean(bppm)))

	LLRmap[center ] = nz * LLRmap[center]
	L = LLRmap[(np.round(x)).astype(int) + center ]
	return L

# [I,YCbCr] = jpeg_rec(image)
#
# simulate decompressed JPEG image from JPEG object 
#
# image: JPEG object from image_data function which is in __init__.py file
#
# I: decompressed image (RGB)
# YCbCr: decompressed image (YCbCr)


def jpeg_rec(image):
	Y =  ibdct(dequantize(image['coef_arrays'][0], image['quant_tables'][0]))
	Cb = ibdct(dequantize(image['coef_arrays'][1], image['quant_tables'][1]))
	Cr = ibdct(dequantize(image['coef_arrays'][2], image['quant_tables'][1]))

	Y = Y + 128
	r,c = Y.shape
	Cb = np.kron(Cb,np.ones((2,2))) + 128
	Cr = np.kron(Cr,np.ones((2,2))) + 128
	
	Cb = Cb[0:r,0:c]
	Cr = Cr[0:r,0:c]
	I = np.zeros((Y.shape[0],Y.shape[1],3))
	I[:,:,0] = Y + 1.402 * (Cr -128)
	I[:,:,1] = Y - 0.34414 *  (Cb - 128) - 0.71414 * (Cr - 128)
	I[:,:,2] = Y + 1.772 * (Cb - 128)

	YCbCr=np.zeros((Y.shape[0],Y.shape[1],3))
	YCbCr[:,:,0] = Y
	YCbCr[:,:,1] = Cb
	YCbCr[:,:,2] = Cr

	return (I,YCbCr)


# [LLRmap2] = smooth_unshift(LLRmap,k1,k2)
#
# smooth likelihood map by applying a 3x3 mean filter
# align map with the examined image
#
# LLRmap: raw likelihood map 
# k1,k2: grid shift of primary compression


def smooth_unshift(LLRmap,k1,k2):
	LLRmap = filt(LLRmap)
	LLRmap_big = np.zeros(8*np.array(LLRmap.shape))
	LLRmap_big[::8,::8] = LLRmap
	bil = conv2(np.ones((8,8)), np.ones((8,8)))
	bil = bil/float(64)
	LLRmap_big = filt2(LLRmap_big,bil)
	LLRmap2 = LLRmap_big[15-k1:-16-k1:8,15-k2:-16-k2:8]
	return LLRmap2


# function [Q, alpha, Lmax, ii] = EMperiod(x, Qmin, Qmax, alpha0, p0, p1, dLmin, maxIter)
# 
# estimate quantization factor Q and mixture parameter alpha from data x
# x are assumed distributed as h(x) = alpha * p0(x) + (1 - alpha) * p1(x,Q)
# Qmin, Qmax: range of possible Qs
# alpha0: initial guess for alpha
# dLmin, maxIter: convergence parameters
#
# alpha is estimated through the EM algorithm for every Q = Qmin:Qmax
# the optimal Q is found by exhaustive maximization over the true
# log-likelihood function L = sum(log(h(x|Q)))
# the EM algorithm is assumed to converge when the increase of L is less
# than dLmin
#
# Lmax: final value of log-likelihood function
# ii: final number of iterations


def EMperiod(x, Qmin, Qmax, alpha0, p0, p1, dLmin, maxIter):
	h0 = p0(x).astype(float)	
	Qvec = array(range(int(Qmin),int(Qmax)+1))#int changed
	alphavec = alpha0*ones(Qvec.shape)
	h1mat = zeros((max(Qvec.shape), max(x.shape) ))
	for k in range(max(Qvec.shape)):
		h1mat[k,:] = p1(x, Qvec[k])
	Lvec = -inf+zeros(Qvec.shape)
	#infinity value
	Lmax = -inf
	delta_L = inf
	ii = 0
	while delta_L > dLmin and ii < maxIter:
		ii = ii + 1
		for k in range(max(Qvec.shape)):
			# expectation
			beta0 = h0*alphavec[k] / (h0*alphavec[k] + h1mat[k,:]*(1 - alphavec[k]))
			# maximization
			alphavec[k] = mean(beta0)	#mean of an array(beta0)
			# compute true log-likelihood of mixture
			L = np.sum(log(alphavec[k]*h0 + (1-alphavec[k])*h1mat[k,:]))
			if L > Lmax:
				Lmax = L
				Q = Qvec[k]
				alpha = alphavec[k]
				if L - Lvec[k] < delta_L:
					delta_L = L - Lvec[k]
			Lvec[k] = L
	return (Q, alpha, Lmax, ii)


# p1 = h1period(x, Q, hcal, binHist, center, bias, sig)
#
# estimate probability distribution of quantized/dequantized coefficients 
# for value(s) x according to NA-DJPG model
#
# Q: quantization step 
# hcal: histogram of unquantized coefficient
# binHist: bin positions
# center: index of x = 0 in binHist
# bias: mean of quantization/truncation noise after dequantization
# sig: std of quantization/truncation noise
#
# p1: estimated pdf at binHist


def h1period(x, Q, hcal, binHist, center, bias, sig):
	N = np.sum(hcal)	
	#simulate quantization
	if mod(Q,2) == 0:
		hs = concatenate(([0.5],ones((1,Q-1))[0],[0.5]))
		ws = Q/2
	else:
		hs = ones((1,Q))[0]
		ws = (Q-1)/2

	h2 = convolve(hcal,hs)
	ws=int(ws)
	h1 = zeros(binHist.shape)
	if ws==0:
		h1[center ::Q] = h2[center + ws::Q]
	else:
		h1[center ::Q] = h2[center + ws:-ws:Q]
	h1[center:1:-Q] = h2[center + ws:1+ws:-Q]
	# simulate rounding/truncation
	w = int(ceil(3*sig))
	k = array(range(-w,w+1))
	g = exp(-(k+bias)**2/float(sig**2)/float(2))
	h1 = convolve(h1, g)
	if w==0:
		h1 = h1[w:]
	else:
		h1 = h1[w:-w]

	# normalize probability and use Laplace correction to avoid p1 = 0
	h1 = h1/float(np.sum(h1))
	h1 = (h1*N+1)/float(N+size(binHist))
	p1 = h1[(np.round(x)).astype(int) + int(center)]
	return p1

# p1 = h1periodDQ(x, Q1, Q2, hcal, binHist, center, bias, sig)
#
# estimate probability distribution of quantized/dequantized coefficients 
# for value(s) x according to A-DJPG model
#
# Q: quantization step 
# hcal: histogram of unquantized coefficient
# binHist: bin positions
# center: index of x = 0 in binHist
# bias: mean of quantization/truncation noise after dequantization
# sig: std of quantization/truncation noise
#
# p1: estimated pdf at binHist


def h1periodDQ(x, Q1, Q2, hcal, binHist, center, bias, sig):
	N = float(np.sum(hcal)	)
	#simulate quantization using Q1
	if mod(Q1,2) == 0:
		hs = concatenate(([0.5],ones((1,Q1-1))[0],[0.55]))
		ws = Q1/2
	else:
		hs = ones((1,Q1))[0]
		ws = (Q1-1)/2
	#convolution of two arrays
	h2 = convolve(hcal[0],hs)
	ws=int(ws)
	
	# simulate dequantization
	h1 = zeros(binHist.shape)
	if ws==0:
		h1[center ::Q1] = h2[center + ws::Q1]	
	else:
		h1[center ::Q1] = h2[center + ws:-ws:Q1]
	h1[center:1:-Q1] = h2[center + ws:1+ws:-Q1]    

	# simulate rounding/truncation
	w = int(ceil(5*sig))
	k = array(range(-w,w+1))
	g = exp(-(k+bias)**2/float(sig**2)/float(2))
	h1 = convolve(h1, g)
	
	if w==0:
		h1 = h1[w:]
	else:
		h1 = h1[w:-w]
	
	# simulate quantization using Q2
	if mod(Q2,2) == 0:
		hs = concatenate(([0.5],ones((1,Q2-1))[0],[0.5]))
		ws = Q2/2
	else:
		hs = ones((1,Q2))[0]
		ws = (Q2-1)/2
	h1 = convolve(h1,hs)
	ws=int(ws)
	if ws==0:
		h1 = h1[int(mod(center,Q2)) + ws::Q2]
	else:
		h1 = h1[int(mod(center,Q2)) + ws:-ws:Q2]
	h1 = h1/float(np.sum(h1))
	
	h1 = (h1*N+1)/(N+size(binHist)/Q2)
	p1 = h1[np.round(x) + int(floor(center/Q2)) ]
	return p1



