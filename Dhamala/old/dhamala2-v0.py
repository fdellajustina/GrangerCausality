import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.signal import csd
from pyGC import *

def ARmodel(N=5000, C=0.2, cov = None):

	E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))

	X = np.random.rand(N)
	Y = np.random.rand(N)

	for t in range(2, N):
		#X[t] = 0.9 * X[t-1] - 0.5 * X[t-2] + E[:,0][t]
		#Y[t] = 0.8 * Y[t-1] - 0.5 * Y[t-2] + 0.16 * X[t-1] - 0.2 * X[t-2] + E[:,1][t] 
		X[t] = 0.55*X[t-1] - 0.8*X[t-2] + C*Y[t-1] + E[t,0]
		Y[t] = 0.55*Y[t-1] - 0.8*Y[t-2] +E[t,1]

	return X, Y

def compute_freq(N, Fs):
	# Simulated time
	T = N / Fs
	# Frequency array
	f = np.arange(0,Fs/2+1/T,1/T)

	return f

def cxy(X, Y=[], Fs=1):
	# Number of data points
	N = X.shape[0]

	if len(Y) > 0:
		Xfft = np.fft.fft(X)[1:len(freq)+1]
		Yfft = np.fft.fft(Y)[1:len(freq)+1]
		Pxy  = Xfft*np.conj(Yfft) / N
		return Pxy
	else:
		Xfft = np.fft.fft(X)[1:len(freq)+1]
		Pxx  = Xfft*np.conj(Xfft) / N
		return Pxx

N      = 500
fs     = 200
C      = 0.25
Trials = 5000

#cov    = np.array([ [1.00, 0.40],[0.40, 0.70] ])
cov    = np.array([ [1.00, 0.0],[0.0, 1.0] ])

freq   = compute_freq(N, fs)

S      = np.zeros([2,2,N//2+1]) + 1j*np.zeros([2,2,N//2+1])

for i in range(Trials):
	if i%500 == 0:
		print('Trial = ' + str(i))
	x, y = ARmodel(N, C, cov)

	S[0,0,:] += cxy(X=x, Y=[], Fs=fs) / Trials
	S[0,1,:] += cxy(X=x, Y=y,  Fs=fs) / Trials
	S[1,0,:] += cxy(X=y, Y=x,  Fs=fs) / Trials
	S[1,1,:] += cxy(X=y, Y=[], Fs=fs) / Trials

#scio.savemat('spec_mat.mat', {'f':f, 'S': S})

plt.figure()
plt.plot(freq, S[0,0,:].real)
plt.plot(freq, S[1,1,:].real)
plt.plot(freq, S[0,1,:].real)
plt.legend([r'$S_{xx}$', r'$S_{yy}$', r'$S_{xy}$'])
plt.show()

Snew, Hnew, Znew = wilson_factorization(S, freq, fs, Niterations=100, tol=1e-9)

Ix2y, Iy2x, Ixy  = granger_causality(Snew, Hnew, Znew)

plt.figure()
plt.plot(freq, Ix2y)
plt.plot(freq, Iy2x)
plt.legend([r'$X->Y$', r'$Y->X$'])
plt.show()
