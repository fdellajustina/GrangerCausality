import numpy             as np

# Autoregressive model from Dhamala (2008)
def ARmodel(N=5000, Trials = 10, C=0.25, cov = None):
	# initializing model's variables
	X = np.random.random([2,Trials, N])
	#X2 = np.random.random([Trials, N])

	for i in range(Trials):
		E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,)) # create noise using covariant matrix 
		# creating time series from autoregressive model of order 2
		for t in range(2, N):
			X[0,i,t] = 0.55*X[0,i,t-1] - 0.8*X[0,i,t-2] + C*X[1,i,t-1] + E[t,0]
			X[1,i,t] = 0.55*X[1,i,t-1] - 0.8*X[1,i,t-2] +E[t,1]
	return X