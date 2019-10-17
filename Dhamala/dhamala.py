import numpy        as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../pyGC/')  # Granger Causality functions
from   pySpec  import *      # spectral analisys
from   pyGC    import *      # Wilson factorisation and GC functions

# Autoregressive model from Dhamala (2008)
def ARmodel(N=5000, C=0.2, cov = None):
	# create noise using covariant matrix 
	E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))

	# initializing model's variables
	X1 = np.random.rand(N)
	X2 = np.random.rand(N)

	# creating time series from autoregressive model of order 2
	for t in range(2, N):
		X1[t] = 0.55*X1[t-1] - 0.8*X1[t-2] + C*X2[t-1] + E[t,0]
		X2[t] = 0.55*X2[t-1] - 0.8*X2[t-2] +E[t,1]

	return X1, X2

if __name__ == "__main__":
	N       = 5000                 # time serie size
	fs      = 200                  # frequency sampling
	freq    = compute_freq(N, fs)  # array with frequencis
	Trials  = 5000                 # repetitions for statistic significance
	Verbose = False

	C       = 0.25                                  # model parameter
	cov     = np.array([ [1.00, 0.0],[0.0, 1.0] ])  # covariant matrix of the system

	S       = np.zeros([2,2,N//2+1]) + 1j*np.zeros([2,2,N//2+1]) # Spectral density matrix

	for i in range(Trials):
		if (Verbose and i%500 == 0):
			print('Trial = ' + str(i))
		x1, x2    = ARmodel(N, C, cov)

		S[0,0,:] += cxy(X=x1, Y=[], Fs=fs, f=freq) / Trials    # Autospectrum Sxx
		S[0,1,:] += cxy(X=x1, Y=x2, Fs=fs, f=freq) / Trials    # Crosspectrum Sxy
		S[1,0,:] += cxy(X=x2, Y=x1, Fs=fs, f=freq) / Trials    # Crosspectrum Syx
		S[1,1,:] += cxy(X=x2, Y=[], Fs=fs, f=freq) / Trials    # Autospectrum Syy

	#scio.savemat('spec_mat.mat', {'f':f, 'S': S})

	# Auto and Cross Spectrum plots
	plt.figure()
	plt.plot(freq, S[0,0,:].real, 'b.-', ms=0.7)
	plt.plot(freq, S[1,1,:].real, 'r.-', ms=0.7)
	plt.plot(freq, S[0,1,:].real, 'k.-', ms=0.7)
	plt.legend([r'$S_{xx}$', r'$S_{yy}$', r'$S_{xy}$'])
	plt.show()

	Snew, Hnew, Znew = wilson_factorization(S, freq, fs, Niterations=100, tol=1e-9, verbose=False)

	Ix2y, Iy2x, Ixy  = granger_causality(Snew, Hnew, Znew)

	plt.figure()
	plt.plot(freq, Ix2y)
	plt.plot(freq, Iy2x)
	plt.legend([r'$X->Y$', r'$Y->X$'])
	plt.show()
