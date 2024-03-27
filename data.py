from sys import prefix
from graphical_models.classes.dags.dag import DAG
import numpy as np
from numpy.linalg import inv
import graphical_models as gm


# generate a structural causal model where you can query samples
class synthetic_instance(object):

    def __init__(self, nnodes, sigma_square, std):

        self.nnodes = nnodes
        self.sigma_square = sigma_square
        
        # generate DAG
        self.DAG = gen_dag(nnodes)

        # generate DAG weights
        self.weighted_DAG = gm.rand.rand_weights(self.DAG, weight_func)

        # create linear gaussian SCM
        self.eps_means = np.zeros(nnodes)
        self.eps_cov = self.sigma_square * np.identity(nnodes)
        self.B = np.transpose(self.weighted_DAG.to_amat())
        self.A = inv(np.identity(nnodes)- self.B)
        self.mu = np.matmul(self.A, self.eps_means.reshape(-1,1)) 

        # standardize data to make each variable equal variance
        if std:
            scale = np.matmul(np.matmul(self.A, self.eps_cov), self.A.T).diagonal().reshape(-1,1)**(-0.5)
            self.eps_means = scale.reshape(-1)*self.eps_means
            self.eps_cov = scale * self.eps_cov * (scale.reshape(-1))
            self.sigma_square = self.eps_cov.diagonal()
            self.B = scale * self.B * ((1/scale).reshape(-1))
            self.A = scale * self.A * ((1/scale).reshape(-1))
            self.mu = scale * self.mu


	# get N observational samples
    def sample(self, n):
        eps = np.random.multivariate_normal(self.eps_means, self.eps_cov, n).reshape(self.nnodes, n)
        batch = np.dot(self.A, eps)

        return batch


# generate a multi-layer star graph
def gen_dag(nnodes, layer=1):
	dag = gm.DAG(set(range(nnodes)))

	perm = np.random.permutation(nnodes)
	dag.add_arcs_from((perm[i+1], perm[0]) for i in range(nnodes-1))	

	return dag


# weight function
def weight_func(size):
	sgn = np.random.binomial(n=1, p=0.5, size=size) 
	
	rand = []
	for i in range(size):
		if sgn[i]==0:
			# recommended low & high: e^{-1/p} & e^{1/p}
			rand.append(np.random.uniform(low=-1, high=-0.25))
		else:
			rand.append(np.random.uniform(low=0.25, high=1))
		
	return rand