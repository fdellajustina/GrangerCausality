import numpy             as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../pyGC/')   # Granger Causality functions
from   pySpec   import *      # spectral analisys
from   pyGC     import *      # Wilson factorisation and GC functions
from   ar_model import *

if __name__ == "__main__":
	N  = 900
	fs = 200
	dt = 1.0 / fs
	C  = 0.25
	Trials = 500
	freq   = compute_freq(N, fs)  # array with frequencis

	Mor = np.zeros([2,2,N,N//2+1]) + 1j*np.zeros([2,2,N,N//2+1])

	cov     = np.array([ [1.00, 0.0],[0.0, 1.0] ])  # covariant matrix of the system
	x = ARmodel(N=N, Trials = Trials, C=C, cov=cov)

	for i in range(Trials):
		if i%50 == 0:
			print('Trial = ' + str(i))
		Wx = morlet(x[0,i,:], freq, fs)
		Wy = morlet(x[1,i,:], freq, fs)

		Mor[0,0] += Wx*np.conj(Wx) / Trials
		Mor[0,1] += Wx*np.conj(Wy) / Trials
		Mor[1,0] += Wy*np.conj(Wx) / Trials
		Mor[1,1] += Wy*np.conj(Wy) / Trials

	Ix2y = np.zeros([N//2+1, N])
	Iy2x = np.zeros([N//2+1, N])
	for i in range(N):
		if i%50 == 0:
			print('N = ' + str(i))
		_ , Hnew, Znew  = wilson_factorization(Mor[:,:,i,:], freq, fs, Niterations=30, verbose=False)
		Ix2y[:,i], Iy2x[:,i], _ = granger_causality(Mor[:,:,i,:], Hnew, Znew) 

	tlim = N/fs
	plt.subplot(2,1,1)
	plt.imshow(Iy2x, aspect='auto', cmap='jet', origin='lower', extent=[0, tlim, freq.min(), freq.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
	plt.ylabel('frequency (Hz)')
	plt.title(r'Granger causality: $Y\rightarrow X$')

	plt.subplot(2,1,2)
	plt.imshow(Ix2y, aspect='auto', cmap='jet', origin='lower', extent=[0, tlim, freq.min(), freq.max()], vmin=0, vmax=np.round(Iy2x.max(),1))
	plt.ylabel('frequency (Hz)')
	plt.xlabel('time (sec)')
	plt.title(r'Granger causality: $X\rightarrow Y$')
	plt.tight_layout()
	plt.savefig('figures/2-wavelet.png', dpi=200)
	plt.close()
