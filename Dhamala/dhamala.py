import numpy             as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../pyGC/')   # Granger Causality functions
from   pySpec   import *      # spectral analisys
from   pyGC     import *      # Wilson factorisation and GC functions
from   ar_model import *

if __name__ == "__main__":
	N       = 3000                 # time serie size
	fs      = 200                  # frequency sampling
	freq    = compute_freq(N, fs)  # array with frequencis
	Trials  = 5000                 # repetitions for statistic significance
	Verbose = False
	SaveFig = True

	C       = 0.25                                  # model parameter
	cov     = np.array([ [1.00, 0.0],[0.0, 1.0] ])  # covariant matrix of the system

	S       = np.zeros([2,2,N//2+1]) + 1j*np.zeros([2,2,N//2+1]) # Spectral density matrix

	x       = ARmodel(N, Trials, C, cov)
	for i in range(Trials):
		S[0,0,:] += cxy(X=x[0,i,:], Y=[]      , Fs=fs, f=freq) / Trials    # Autospectrum Sxx
		S[0,1,:] += cxy(X=x[0,i,:], Y=x[1,i,:], Fs=fs, f=freq) / Trials    # Crosspectrum Sxy
		S[1,0,:] += cxy(X=x[1,i,:], Y=x[0,i,:], Fs=fs, f=freq) / Trials    # Crosspectrum Syx
		S[1,1,:] += cxy(X=x[1,i,:], Y=[]      , Fs=fs, f=freq) / Trials    # Autospectrum Syy

	Snew, Hnew, Znew = wilson_factorization(S, freq, fs, Niterations=100, tol=1e-9, verbose=False)
	Ix2y, Iy2x, Ixy  = granger_causality(Snew, Hnew, Znew)

	if SaveFig:
		if not os.path.isdir("figures"):
		    os.mkdir("figures")
		    print ("Created the directory %s" % "figures")			

		# Auto and Cross Spectrum plots
		plt.figure()
		plt.plot(freq, S[0,0,:].real, 'b.-', ms=0.7)
		plt.plot(freq, S[1,1,:].real, 'r.-', ms=0.7)
		plt.plot(freq, S[0,1,:].real, 'k.-', ms=0.7)
		plt.xlabel("frequency (Hz)")
		plt.legend([r'$S_{xx}$', r'$S_{yy}$', r'$S_{xy}$'])
		plt.savefig('figures/1a-Spectrum.png', dpi=200)
		plt.close()

		# Coherency plot
		Coh = np.abs(S[0,1,:])**2/(S[0,0,:]*S[1,1,:])
		plt.figure()
		plt.plot(freq, Coh.real, 'k.-', ms=0.7)
		plt.xlabel("frequency (Hz)")
		plt.ylabel("$Coherence (x_1\;x_2)$")
		plt.savefig('figures/1b-Coherence.png', dpi=200)
		plt.close()

		# Granger Causality plot
		plt.figure()
		plt.plot(freq, Ix2y, 'k.-', ms=0.7)
		plt.plot(freq, Iy2x, 'r.-', ms=0.7)
		plt.xlabel("frequency (Hz)")
		plt.ylabel("Granger Causality")
		plt.legend([r'$X->Y$', r'$Y->X$'])
		plt.savefig('figures/1c-GrangerCausality.png', dpi=200)
		plt.close()
