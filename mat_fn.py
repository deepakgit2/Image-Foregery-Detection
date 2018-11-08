import numpy as np

def f(p,m):
	if p == 0:
		return np.sqrt(1/float(m))
	else:
		return np.sqrt(2/float(m))
#same as matlab function idct2 
def idct2(a): 
	M,N=a.shape
	t_a=np.zeros((M,N))
	for m in range(M):
		for n in range(N):
			for p in range(M):
				for q in range(N):
					t_a[m,n]+=f(p,M) * f(q,N) * a[p,q] * np.cos( np.pi*(2*m+1)*p/float(2*M) ) * np.cos( np.pi*(2*n+1)*q/float(2*N) )
	return t_a
					



#same as imfilter(a,ones(3),'symmetric','full') in matlab
def filt(a): 
	m,n=a.shape
	#create a zero matrix of size (m,n)
	t_a=np.zeros((m,n))
	a=np.append(np.array([a[0,:]]),a ,axis=0)
	a=np.append(a,np.array([a[-1,:]]),axis=0 )
	a=np.append(np.array([a[:,0]]).T,a,axis=1)
	a=np.append(a,np.array([a[:,-1]]).T,axis=1)
	for i in range(m):
		for j in range(n):
			t_a[i,j]=np.sum(a[i:i+3,j:j+3])

	t_a=np.append(np.array([t_a[0,:]]),t_a ,axis=0)
	t_a=np.append(t_a,np.array([t_a[-1,:]]),axis=0 )
	t_a=np.append(np.array([t_a[:,0]]).T,t_a,axis=1)
	t_a=np.append(t_a,np.array([t_a[:,-1]]).T,axis=1)
	return t_a


#same as imfilter(a,b,'full') in matlab
def filt2(a,b):
	m0,n0=b.shape
	fs=(m0-1)/2
	m,n=a.shape
	#create a zero matrix
	t_a=np.zeros((m+4*fs,n+4*fs))
	a=np.append(np.zeros((2*fs,n)),a ,axis=0)
	a=np.append(a,np.zeros((2*fs,n)),axis=0 )
	a=np.append(np.zeros((m+4*fs,2*fs)),a,axis=1)
	a=np.append(a,np.zeros((m+4*fs,2*fs)),axis=1)
	for i in range(m+2*fs):
		for j in range(n+2*fs):
			t_a[i+fs,j+fs]=np.sum(a[i:i+m0,j:j+n0]*b)
	return t_a[fs:-fs,fs:-fs]


#same as hist(a,b) in matlab
def hist(a,b):
	h=np.abs((b[0]-b[1])/float(2)) #absolute value
	r= np.array([np.count_nonzero(np.logical_and(i-h<a,a<=i+h)==True) for i in b])
	
	#minimum value of an array
	if np.min(a)<b[0]-h:
		r[0]=np.count_nonzero((a<=b[0]+h)==True)
	#maximum value of an array
	if np.max(a)>b[0]+h:
		r[-1]=np.count_nonzero((b[-1]-h<a)==True)
	return r



