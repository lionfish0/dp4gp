
from dp4gp import dp4gp
import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dp4gp.utils import bin_data


class DPGP_histogram(dp4gp.DPGP):
    """Using the histogram method"""
    
    def __init__(self,sens,epsilon,delta):
        super(DPGP_histogram, self).__init__(None,sens,epsilon,delta)

    def prepare_model(self,Xtest,X,step,ys,variances=None,lengthscale=None,aggregation='mean',mechanism='laplace'):
        """
        Prepare the model, ready for making predictions
        
        The data in X and ys are the locations and values of the individual data points, respectively.
        The Xtest contains the coordinates of the corners of each bin
        step is a vector of the step sizes 
        
        Bin data X into equally sized bins defined by Xtest and step.
        step is a vector of step sizes.
        ys are the outputs (to be summed and averaged)
        
        variances and lengthscale not used.
        """
        
        bincounts, bintotals, binaverages = bin_data(Xtest,X,step,ys,aggregation=aggregation)
        if aggregation=='median':
            raise NotImplementedError
        if aggregation=='mean':            
            sens_per_bin = self.sens/bincounts
        if aggregation=='sum':
            sens_per_bin = self.sens
        if aggregation=='density':
            sens_per_bin = (self.sens/np.prod(step))
            
        if mechanism=='gaussian':
            c = np.sqrt(2*np.log(1.25/self.delta)) #1.25 or 2 over delta?
            bin_sigma = c*sens_per_bin/self.epsilon #noise standard deviation to add to each bin       
            ##add DP noise to the binaverages
            dp_binaverages=binaverages+np.random.randn(binaverages.shape[0])*bin_sigma
        if mechanism=='laplace':
            bin_sigma = np.array(sens_per_bin / self.epsilon)
            dp_binaverages=binaverages+np.random.laplace(scale=bin_sigma,size=binaverages.shape) #TODO check that scale=std so this is consistent
            
        #we need to build the input for the integral kernel
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest+step
        newXtest[:,1::2] = Xtest

        #we don't want outputs that have no training data in.
        empty = np.isnan(dp_binaverages)
        dp_binaverages[empty] = 0 #we'll make those averages zero

        self.Xtest = newXtest
        self.dp_binaverages = dp_binaverages
        return bincounts, bintotals, binaverages, sens_per_bin, bin_sigma, dp_binaverages
    
    def optimize(self,messages=False):
        pass
     
    def draw_prediction_samples(self,Xtest,N=1):
        assert N==1, "DPGP_histogram only returns one DP prediction sample (you will need to rerun prepare_model to get an additional sample)"
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest
        newXtest[:,1::2] = 0
        preds = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            v = np.ones(self.Xtest.shape[0],dtype=bool)
            for idx in range(0,Xtest.shape[1]):
                v = v & ((Xtest[i,idx] < self.Xtest[:,idx*2]) & (Xtest[i,idx] >= self.Xtest[:,idx*2+1]))        
            if not np.any(v):
                preds[i] = np.mean(self.dp_binaverages) #not sure what else to do?
            else:
                preds[i] = np.mean(self.dp_binaverages[v]) #if we have overlapping tiles!
        return preds, None

