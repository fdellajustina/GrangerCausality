'''
	Python module to compute information theoretical quantities
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as ss
from math import log,pi,exp
from sklearn.neighbors import NearestNeighbors

def BinNeuronEntropy(SpikeTrain):
	r'''
	Description: Computes the entropy of a binary neuron (with a response consisting of 0s and 1s).
	Inputs:
	SpikeTrain: Binary spike train of a neuron (must be composed of 0s and 1s)
	Outputs:
	Returns the entropy of the binary spike train
	'''
	T = len(SpikeTrain) # Length of the spike train
	P_firing = np.sum(SpikeTrain) / float(T) # Probability of firing (probability of the spike trai being 1)
	P_notfiring = 1.0 - P_firing # Pobability of silence (probability of the spike tain being 0)
	# Computing and returning the entropy of the binary spike train
	return -P_firing*np.log2(P_firing) - P_notfiring*np.log2(P_notfiring)


def EntropyFromProbabilities(Prob):
	r'''
	Description: Computes the entropy of given probability distribution.
	Inputs:
	Prob: Probability distribution of a random variable
	Outputs:
	H: Entropy of the probabilitie distribution.
	'''
	H = 0
	s = Prob.shape
	for p in Prob:
		if p > 0.00001:
			H -= p*np.log2(p)
	return H

def binMutualInformation(sX, sY, tau):
	r'''
	Description: Computes the delayed mutual information bewtween two binary spike trains.
	Inputs:
	sX: Binary spike train of neuron X
	sY: Binary spike train of neuron Y
	tau: Delay applied in the spike train of neuron Y
	Outputs:
	MI: Returns the mutual information MI(sX, sY)
	'''
	PX = np.zeros([2])
	PY = np.zeros([2])
	PXY = np.zeros([2,2])
	for t in range( np.maximum(0, 0-tau), np.minimum(len(sX)-tau, len(sX)) ):
		PX[1] += sX[t]
		PY[1] += sY[t+tau]
		if (sX[t]==0) and (sY[t+tau]==0):
			continue
		else:
			PXY[sX[t], sY[t+tau]] += 1

	# Estimating [0, 0] pairs for the PXY matrix
	N = len(sX)   # Number of bins in the spike train
	Np = N - tau  # Number of pairs
	PXY[0,0] = Np - np.sum(PXY)
	PX[0]    = Np - PX[1]
	PY[0]    = Np - PY[1]
	# Normalizing probabilities
	PX = PX / np.sum(PX)
	PY = PY / np.sum(PY)
	PXY = PXY / np.sum(PXY)
	HX = EntropyFromProbabilities(PX)
	HY = EntropyFromProbabilities(PY)
	HXY = EntropyFromProbabilities(np.reshape(PXY, (4)))
	MI  = HX + HY - HXY
	return MI

def binTransferEntropy(x, y, delay):
	r'''
	Description: Computes the delayed transfer entropy, from y to x, bewtween two binary spike trains.
	Inputs:
	x: Binary spike train of neuron X
	y: Binary spike train of neuron Y
	delay: Delay applied in the spike train of neuron Y
	Outputs:
	TE: Returns the Transfer Entropy TEy->x
	'''
	T = len(x)

	if delay == 1:
		delay = 0
	elif delay > 1:
		delay -= 1

	px = np.array( [T-np.sum(x), np.sum(x)] ) / float(T) # p(x)
	py = np.array( [T-np.sum(y), np.sum(y)] ) / float(T) # p(y)
	pxy = np.zeros([2, 2])                               # p(x, y)
	pxy1 = np.zeros([2, 2])                              # p(x1, y)
	pxyz = np.zeros([2, 2, 2])                           # p(x1, x, y)

	for i in range(0, T-delay):
		if (x[i+delay] == 0 and y[i] == 0):
			continue
		else:
			pxy[ x[i+delay], y[i] ] += 1.0

	pxy[0,0] = (T - delay - np.sum(pxy))
	pxy = pxy / float(T-delay)

	for i in range(0, T-1):
		if (x[i+1] == 0 and x[i] == 0):
			continue
		else:
			pxy1[ x[i+1], x[i] ] += 1.0

	for i in range(0, T-1-delay):
		if (x[i+1+delay] == 0 and x[i+delay] == 0 and y[i] == 0):
			continue
		else:
			pxyz[ x[i+1+delay], x[i+delay], y[i] ] += 1.0

	pxy1[0,0] = (T - 1 - np.sum(pxy1))
	pxyz[0,0,0] = (T - 1 - delay - np.sum(pxyz))
	# Normalizing probabilities
	pxy1 = pxy1 / float(T-1)
	pxyz = pxyz / float(T-1-delay)

	TE = 0
	for xn in [0, 1]:
		for yn in [0, 1]:
			for xn1 in [0, 1]:
				if pxy1[xn1, xn] > 0.00001 and pxy[xn, yn] > 0.00001 and pxyz[xn1, xn, yn] > 0.00001 and px[xn] > 0.00001: 
					TE += pxyz[xn1, xn, yn] * np.log2( ( pxyz[xn1, xn, yn] * px[xn] ) / ( pxy1[xn1, xn] * pxy[xn, yn] ) )
				else:
					continue
	return TE

def KernelEstimatorMI(x, y, bw = 0.3, kernel = 'tophat', delay = 0, norm = True):
	r'''
	Description: Computes the mutual information between two signals x and y.
	Inputs:
	x: Signal x.
	y: Signal y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed mutual information
	norm: Sets whether the data will be normalized or not.
	Outputs:
	MI: Returns the mutual information between x and y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Applying delays
	if delay == 0:
		x = x
		y = y
	elif delay > 0:
		x = x[:-delay]
		y = y[delay:]

	N = len(x)

	grid  = np.vstack([x, y])

	pdf_x = kde_sklearn(x, grid[0], kernel = kernel, bandwidth=bw)
	pdf_y = kde_sklearn(y, grid[1], kernel = kernel, bandwidth=bw)
	pdf_xy = kde_estimator(y, x, grid[1], grid[0], kernel = kernel, bandwidth=bw)

	MI = 0
	count = 0
	for i in range(len(pdf_x)):
		if pdf_x[i] > 0.00001 and pdf_y[i] > 0.00001 and pdf_xy[i] > 0.00001:
			MI += np.log2( pdf_xy[i] / ( pdf_x[i]*pdf_y[i] ) ) 
			count += 1.0
	return MI / float(len(pdf_x) )

def KernelEstimatorMImulti(x, y, z, bw = 0.3, kernel = 'tophat', norm = True):
	r'''
	Description: Computes the mutual information between two signals x and y.
	Inputs:
	x: Signal x.
	y: Signal y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed mutual information
	norm: Sets whether the data will be normalized or not.
	Outputs:
	MI: Returns the mutual information between x and y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)
		z = (z - np.mean(z))/np.std(z)

	N = len(x)

	grid  = np.vstack([x, y, z])

	pdf_x = kde_sklearn(x, grid[0], kernel = kernel, bandwidth=bw)
	pdf_y = kde_sklearn(y, grid[1], kernel = kernel, bandwidth=bw)
	pdf_z = kde_sklearn(z, grid[2], kernel = kernel, bandwidth=bw)
	pdf_xy = kde_estimator(x, y, grid[0], grid[1], kernel = kernel, bandwidth=bw)
	pdf_xz = kde_estimator(x, z, grid[0], grid[2], kernel = kernel, bandwidth=bw)
	pdf_yz = kde_estimator(y, z, grid[1], grid[2], kernel = kernel, bandwidth=bw)
	pdf_xyz = kde_estimator2(x, y, z, grid[0], grid[1], grid[2], kernel = kernel, bandwidth=bw)

	MI = 0
	count = 0
	for i in range(len(pdf_x)):
		if pdf_x[i] > 0.00001 and pdf_y[i] > 0.00001 and pdf_z[i] > 0.00001 and pdf_xy[i] > 0.00001 and pdf_xz[i] > 0.00001 and pdf_yz[i] > 0.00001 and pdf_xyz[i] > 0.00001:
			MI += np.log2( pdf_xy[i]*pdf_xz[i]*pdf_yz[i] / (pdf_xyz[i]*pdf_x[i]*pdf_y[i]*pdf_z[i]) ) 
			count += 1.0
	return MI / float(len(pdf_x) )

def KernelEstimatorTE(x, y, bw = 0.3, kernel = 'tophat', delay = 0, norm=True):
	r'''
	Description: Computes the transfer entropy between two signals x and y.
	Inputs:
	x: Signal x.
	y: Signal y.
	bw: bandwidth of the kernel estimator.
	kernel: Kernel used in the KDE estimator ('gaussian', 'tophat', 'cosine'; see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html)
	delay: Delay applied between x and y, for the delayed transfer entropy
	norm: Sets whether the data will be normalized or not.
	Outputs:
	TE: Returns the transfer entropy from x to y.
	'''

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Applying delays
	y1 = y[1+delay:]
	x = x[:-(1+delay)]
	y = y[delay:-1]

	N = len(x)

	grid  = np.vstack([x, y, y1])

	pdf_y   = kde_sklearn(y, grid[1], kernel = kernel, bandwidth=bw)                                 # p(i_n)
	pdf_xy  = kde_estimator(y, x, grid[1], grid[0], kernel = kernel, bandwidth=bw)                   # p(i_n, j_n)  
	pdf_yy1 = kde_estimator(y1, y, grid[2], grid[1], kernel = kernel, bandwidth=bw)                  # p(i_n+1, i_n)
	pdf_xyz = kde_estimator2(y1, y, x, grid[2], grid[1], grid[0], kernel = kernel, bandwidth=bw)     # p(i_n+1, i_n, j_n)
	
	TE = 0
	count = 0
	for i in range( len(pdf_y) ):
		if pdf_y[i]>0.00001 and pdf_xy[i]>0.00001 and pdf_yy1[i]>0.00001 and pdf_xyz[i]>0.00001:
			TE += np.log2( pdf_xyz[i]*pdf_y[i] / ( pdf_xy[i]*pdf_yy1[i] ) )
			count += 1.0
	return TE / float( len(pdf_y) )

def KSGestimatorMI(x, y, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: Computes mutual information using the KSG estimator (for more information see Kraskov et. al 2004).
	Inputs:
	x, y: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma
	from sklearn.neighbors import NearestNeighbors

	N = len(x)

	# Add noise to the data to break degeneracy
	x = x + 1e-8*np.random.rand(N)
	y = y + 1e-8*np.random.rand(N)

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	Z = np.squeeze( zip(x[:, None], y[:, None]) )
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	nx = np.zeros(N)
	ny = np.zeros(N)

	for i in range(N):
		nx[i] = np.sum( np.abs(x[i]-x) < distances[i] )
		ny[i] = np.sum( np.abs(y[i]-y) < distances[i] )
	I = digamma(k) - np.mean( digamma(nx) + digamma(ny) ) + digamma(N)
	return I

def KSGestimatorMImultivariate(X, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: Computes mutual information using the KSG estimator (for more information see Kraskov et. al 2004).
	Inputs:
	X: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma
	from sklearn.neighbors import NearestNeighbors

	N = len(X[0])
	m = len(X)

	for i in range(m):
		X[i] = X[i] + 1e-8*np.random.rand(N, 1)

	if norm == True:
		for i in range(m):
			X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])

	Z = np.squeeze( ZIP(X) )
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	n = np.zeros([N, m])

	for j in range(m):
		for i in range(N):
			n[i][j] = np.sum( np.abs(X[j][i]-X[j]) < distances[i] )

	I = digamma(k) + (m-1) * digamma(N)

	for i in range(m):
		I -= np.mean( digamma(n[:,i]) )

	return I

def delayedKSGMI(x, y, k = 3, norm = True, noiseLevel = 1e-8, delay = 0):
	'''
	Description: Computes the delayed mutual information using the KSG estimator (see method KSGestimator_Multivariate).
	Inputs:
	X: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	delay: Delay applied
	Output:
	I / log(base): Mutual information (if base=2 in bits, if base=e in nats) 
	'''
	if delay == 0:
		x = x
		y = y
	elif delay > 0:
		x = x[:-delay]
		y = y[delay:]
	return KSGestimator_Multivariate(x, y, k = k, norm = norm, noiseLevel = noiseLevel)

def KSGestimatorTE(x, y, k = 3, norm = True, noiseLevel = 1e-8):
	'''
	Description: 
	Inputs:
	x, y: Array with the signals.
	k: Number of nearest neighbors.
	base: Log base (2 for unit bits)
	norm: Whether to normalize or not the data
	noiseLevel: Level of noise added to the data to break degeneracy
	Output:
	I: Mutual information.
	'''
	from scipy.special import digamma
	from sklearn.neighbors import NearestNeighbors

	Ni = len(x)

	# Normalizing data
	if norm == True:
		x = (x - np.mean(x))/np.std(x)
		y = (y - np.mean(y))/np.std(y)

	# Add noise to the data to break degeneracy
	x = x + 1e-8*np.random.rand(Ni)
	y = y + 1e-8*np.random.rand(Ni)

	# Applying shifts
	ym = y[1:]
	x  = x[:-1]
	y  = y[:-1]		

	N = len(x)

	
	Z = np.squeeze( ZIP([x[:, None], y[:, None], ym[:, None]]) )
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='chebyshev').fit(Z)
	distances, _ = nbrs.kneighbors(Z)
	distances = distances[:, k]

	nx = np.zeros(N)
	ny = np.zeros(N)
	nym = np.zeros(N)

	for i in range(N):
		nx[i] = np.sum( np.sqrt( (y[i]-y)**2 + (x[i]-x)**2 ) < distances[i] )
		ny[i] = np.sum( np.sqrt( (y[i]-y)**2 + (ym[i]-ym)**2 ) < distances[i] )
		nym[i] = np.sum( np.abs(ym[i]-ym) < distances[i] )
		
	I = digamma(k) + np.mean( digamma(nym) - digamma(ny) - digamma(nx) )
	return I
	

##################################################################################################
# AUXILIARY FUNCTIONS                                                                            #
##################################################################################################

from sklearn.neighbors import KernelDensity

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def kde_estimator(x, y, x_grid, y_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    data       = np.concatenate( (x[:, None], y[:, None]) , axis = 1)
    data_grid  = np.concatenate( (x_grid[:, None], y_grid[:, None]) , axis = 1)

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(data_grid)
    return np.exp(log_pdf)

def kde_estimator2(x, y, z, x_grid, y_grid, z_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    data       = np.concatenate( (x[:, None], y[:, None], z[:, None]) , axis = 1)
    data_grid  = np.concatenate( (x_grid[:, None], y_grid[:, None], z_grid[:, None]) , axis = 1)

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(data_grid)
    return np.exp(log_pdf)

def ZIP(X):
	'''
	Description: Its the same as the python's zip method, but for lists of arrays.
	Inputs:
	'''
	N = len(X[0])
	C = len(X)
	zipped = []
	for i in range(N):
		aux = []
		for j in range(C):
			aux.append(X[j][i][0])
		zipped.append(aux)
	return zipped 

##########################################################################################
import sys

if sys.argv[-1] == 'ex':

	################################################################################
	# MI HEART <-> BREATH                                                          #
	################################################################################

	MI = []

	data = np.loadtxt('data.txt', delimiter = ',')
	x = data[:, 0]
	y = data[:, 1]
	X = np.array([x[:, None], y[:, None]])
	for i in range(100):
		MI.append( KSGestimatorMImultivariate(X, k = 4, norm = True, noiseLevel = 1e-8) )

	print('MI HEART;BREATH = ' + str( np.mean(MI) ) + ' nats')

	################################################################################
	# MI GAUSSIAN SIGNALS CORRELATED                                               #
	################################################################################

	MI = []

	for i in range(5000):

		numObservations = 1000;
		covariance=0.4;
		sourceArray=np.random.randn(numObservations)
		destArray = np.concatenate( ([0], covariance*sourceArray[0:numObservations-1] + (1-covariance)*np.random.randn(numObservations - 1)) )

		X = np.array([sourceArray[:, None], destArray[:, None]])

		MI.append( KSGestimatorMImultivariate(X, k = 4, norm = False, noiseLevel = 1e-8) )

	MI = np.array(MI)
	MImean = np.mean(MI[MI>0])
	
	print('MI GAUSSIAN = ' + str(MImean) + ' nats')
