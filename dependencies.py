import numpy as np
from numpy.matlib import repmat


# IM2VEC  Reshape 2D image blocks into an array of column vectors
# V,ROWS,COLS=im2vec(IM,BSIZE,PADSIZE)

# IM is an image to be separated into non-overlapping blocks and
# reshaped into an MxN array containing N blocks reshaped into Mx1
# column vectors.  IM2VEC is designed to be the inverse of VEC2IM.
#
# BSIZE(Block Size) is a scalar or 1x2 vector indicating the size of the blocks.
#
# PADSIZE is a scalar or 1x2 vector indicating the amount of vertical
# and horizontal space to be skipped between blocks in the image.
# Default is [0,0].  If PADSIZE is a scalar, the same amount of space
# is used for both directions.  PADSIZE must be non-negative (blocks
# must be non-overlapping).
#
# ROWS indicates the number of rows of blocks found in the image.
# COLS indicates the number of columns of blocks found in the image.
#
# See also VEC2IM.
# python version of Phil Sallee 5/03

def im2vec(im,bsize,padsize=0):
	bsize = bsize + np.array([0,0])
	padsize = padsize + np.array([0,0])	

	#if padsize is negative then raise an error
	if any(padsize <0):
		raise InputException, "Pad size must be non-negative."
	
	#create a tuple which contain dimension of array(im)
	imsize = im.shape
	y = bsize[0]+padsize[0]
	x = bsize[1]+padsize[1]
	rows = int((imsize[0]+padsize[0])/float(y) )
	cols = int((imsize[1]+padsize[1])/float(x) )
	#create of zero matrix
	t = np.zeros((y*rows,x*cols))
	imy=y*rows-padsize[0] 
	imx=x*cols-padsize[1]
	#slicing
	t[:imy,:imx]=im[:imy,:imx]
	#reshape column-wise
	t = np.reshape(t.T,[cols,x,rows,y]).T
	if len(t.shape)<4:
		npt=np.transpose(np.expand_dims(t, axis=4),[0,2,1,3])
	else:
		npt=np.transpose(t,[0,2,1,3])
	t = np.reshape(npt.T, [rows*cols,x,y]).T
	v = t[0:bsize[0],0:bsize[1],0:rows*cols]
	v = np.reshape(v.T,[rows*cols,y*x]).T

	return (v,rows,cols)


# VEC2IM  Reshape and combine column vectors into a 2D image
#    IM=VEC2IM(V,PADSIZE,BSIZE,ROWS,COLS)
#
#    V is an MxN array containing N Mx1 column vectors which will be reshaped
#    and combined to form image IM. 
#
#    PADSIZE is a scalar or a 1x2 vector indicating the amount of vertical and
#    horizontal space to be added as a border between the reshaped vectors.
#    Default is [0 0].  If PADSIZE is a scalar, the same amount of space is used
#    for both directions.
#
#    BSIZE is a scalar or a 1x2 vector indicating the size of the blocks.
#    Default is sqrt(M).
#
#    ROWS indicates the number of rows of blocks in the image. Default is
#    floor(sqrt(N)).
#
#    COLS indicates the number of columns of blocks in the image.  Default
#    is ceil(N/ROWS).
#
# python version of Phil Sallee 5/03

def vec2im(v,padsize=0,bsize=None,rows=None,cols=None):
	#(m,n) is dimension of v
	try:
		m,n = v.shape
	except:
		m = 1
		n = v.shape[0]

	padsize = padsize + np.array([0,0])
	#if padsize is negative then raise input error
	if ( np.any( padsize < 0 ) ):
		raise InputError, "Pad size must be non-negative."

	if bsize==None:
		bsize=int(np.floor(np.sqrt(m)))
	bsize = bsize + np.array([0,0])

	#if product of all elements of bsize then raise an error
	if np.prod(bsize) != m:
		raise ValueError, "Block size does not match size of input vectors."
	if rows==None:
		rows=int(np.floor(np.sqrt(n)))
	if cols==None:
		cols=int(np.ceil(n/np.floor(rows)))

	# make image
	y,x = bsize+padsize
	y,x = (int(y),int(x))
	#create nd array of zero matrix
	t = np.zeros((y,x,int(rows*cols)))
	t[0:bsize[0],0:bsize[1],0:n] = np.reshape(v.T,[n,bsize[1],bsize[0]]).T
	#reshaping
	t = np.reshape(t.T,[cols,rows,x,y]).T
	#alternate of permute function in matlab
	if len(t.shape)<4: 
		npt=np.transpose(np.expand_dims(t, axis=4),[0,2,1,3])
	else:
		npt=np.transpose(t,[0,2,1,3])
	#reshape array column-wise
	t = np.reshape(npt.T, [x*cols,rows*y]).T
	im = t[0:y*rows-padsize[0],0:x*cols-padsize[1]]

	return im


# bdctmtx Blocked discrete cosine transform matrix
#    M = bdctmtx(N) generates a N^2xN^2 DCT2 transform matrix which, when
#    multiplied by NxN image blocks, shaped as N^2x1 vectors, returns the
#    2D DCT transform of each image block vector.
# python version of Phil Sallee 9/03

def bdctmtx(n):
	#create a numpy array having element (0,n-1)
	ar = np.arange(n)
	#A,B = np.meshgrid(a,b) means A is a matrix where each row is a copy of a, and B is a matrix where each column is a copy of b
	c,r = np.meshgrid(ar,ar)
	c0,r0 = np.meshgrid(c.T,c.T)
	c1,r1 = np.meshgrid(r.T,r.T)
	x = np.sqrt(2 / float(n)) * np.cos(np.pi * (2*c + 1) * r / float(2 * n))	
	x[0,:] = x[0,:] / np.sqrt(2)
	m = x.flatten(1)[r0+c0*n] * x.flatten(1)[r1+c1*n]
	return m


# bdct Blocked discrete cosine transform
#    B = bdct(A) computes DCT2 transform of A in 8x8 blocks.  B is
#    the same size as A and contains the cosine transform coefficients for
#    each block.  This transform can be inverted using IBDCT.
#    B = bdct(A,N) computes DCT2 transform of A in blocks of size n*n.
#python version of Phil Sallee 9/03

def bdct(a,n=8):
	#generate the matrix for the full 2D DCT transform (both directions)
	dctm = bdctmtx(n)
	v,r,c = im2vec(a,n)
	#reshape image into blocks, multiply, and reshape back
	b = vec2im(np.dot(dctm,v),0,n,r,c)	
	return b


# QUANTIZE  Dequantize BDCT coefficients, using center bin estimates
#
# ECOEF = DEQUANTIZE(QCOEF,QTABLE) computes a center bin estimate of the
# coefficients given the quantizer indices (quantized coefficients) and a
# quantization table QTABLE.
#
# QTABLE is applied to each coefficient block in QCOEF (shaped as an image)
# as follows: new value = old value * table value


def dequantize(qcoef,qtable):
	#create a tuple which contain dimension of array(qtable)
	blksz = qtable.shape
	v,r,c = im2vec(qcoef, blksz)
	#same as repmat function in matlab
	return vec2im(v*repmat(qtable.flatten(1),v.shape[1],1).T,0,blksz,r,c)

