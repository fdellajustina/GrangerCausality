import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as sig
import numpy as np 
import mne.filter
from infoPy import *
from scipy.fftpack import fft

def MapModel(seed=0, C=0.2, Tsim=100):
	
	np.random.seed(seed)

	x1   = np.random.rand(Tsim)
	x2   = np.random.rand(Tsim)

	mean = [0.0,0.0] 
	cov  = np.array([[1.0,0.0],[0.0,1.0]]) 
	eps  = np.random.multivariate_normal(mean, cov, Tsim).T 

	for t in range(2, Tsim):
		x1[t] = 0.55*x1[t-1] - 0.8*x1[t-2] + C*x2[t-1] + eps[0,t]
		x2[t] = 0.55*x2[t-1] - 0.8*x2[t-2] + eps[1,t]

	return x1, x2

def cxy(X=x, Y=[], fs):
		

N = x1.shape[0]
T = N/fs
f = np.arange(0,fs/2+1/T,1/T)

N       = 5000
c       = 0.2
seed    = 0

fs      = 200
nTrials = 50

Pxx     = np.zeros(129)+1j*np.zeros(Tsim)
Pyy     = np.zeros(129)+1j*np.zeros(Tsim)
Pxy     = np.zeros(129)
Cxy     = np.zeros(129)

X1f     = np.zeros(Tsim)+1j*np.zeros(Tsim)
X2f     = np.zeros(Tsim)+1j*np.zeros(Tsim)
CXY     = np.zeros(Tsim)+1j*np.zeros(Tsim)
CYX     = np.zeros(Tsim)+1j*np.zeros(Tsim)

for i in range(nTrials):
	if(i%500==0):
		print(i)
	seed     = i
	x1, x2   = MapModel(seed, c, Tsim) 

	fx, pxx  = sig.welch(x1, fs)
	fy, pyy  = sig.welch(x2, fs)
	fxy,cxy  = signal.coherence(x1, x2, fs)

	x1f      = fft(x1)
	x2f      = fft(x2)
	X1f      += x1f*np.conjugate(x1f)/nTrials
	X2f      += x2f*np.conjugate(x2f)/nTrials
	CXY      += x1f*np.conjugate(x2f)/nTrials
	CYX      += x2f*np.conjugate(x1f)/nTrials

	Pxx      += pxx/nTrials
	Pyy      += pyy/nTrials
	Cxy      += cxy/nTrials #np.abs(cxy)**2

#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

plt.subplot(1,2,1) 
plt.plot(fx,Pxx) 
plt.plot(fy,Pyy) 
plt.legend(['Pxx','Pyy']) 
#plt.show()

plt.subplot(1,2,2) 
plt.plot(X1f) 
plt.plot(X2f) 
plt.legend(['X1(f)','X2(f)']) 
plt.show()

plt.subplot(2,2,1) 
plt.plot(fx,Pxx) 
plt.plot(fy,Pyy) 
plt.legend(['Pxx','Pyy']) 
#plt.show()

plt.subplot(2,2,2) 
plt.plot(fxy,Cxy) 
plt.legend(['Cxy']) 
plt.show()

plt.figure(0)
plt.plot(x1)
plt.plot(x2)
plt.legend(['x1','x2'])
plt.show()

#############################
fc    = np.arange(5, 62, 2)
sigma = 1

te12 = np.zeros(fc.shape[0])
te21 = np.zeros(fc.shape[0])

for i in range(fc.shape[0]):
	print('Fc = ' + str(fc[i]) + ' Hz')

	f_low, f_high = fc[i]-sigma, fc[i]+sigma

	x1f  = mne.filter.filter_data(x1, 200, f_low, f_high, method = 'iir', verbose=False, n_jobs=40)
	x2f  = mne.filter.filter_data(x2, 200, f_low, f_high, method = 'iir', verbose=False, n_jobs=40)

	#np.savetxt('data_'+str(i)+'.dat', np.array([x1f, x2f]).T)

	#x1b    = (x1f >= 0.3*x1f.mean()).astype(int)
	#x2b    = (x2f >= 0.3*x2f.mean()).astype(int)
	#te12[i]=  binTransferEntropy(x2b, x1b, 10) 
	#te21[i]=  binTransferEntropy(x1b, x2b, 10) 

	te12[i] =  KernelEstimatorTE(x1f, x2f, bw = 0.3, kernel = 'gaussian', delay = 0, norm=False)
	te21[i] =  KernelEstimatorTE(x2f, x1f, bw = 0.3, kernel = 'gaussian', delay = 0, norm=False)

plt.plot(fc, te12)
plt.plot(fc, te21)
plt.legend(['x1->x2', 'x2->x1'])
plt.show()

