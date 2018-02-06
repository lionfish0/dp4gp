
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


class DPGP_centroid_histogram(dp4gp.DPGP):

    def __init__(self,sens,epsilon,delta):   
        """
        DPGP_centroid_histogram(sensitivity=1.0, epsilon=1.0, delta=0.01)
        
        sensitivity=1.0 - the amount one output can change
        epsilon=1.0, delta=0.01 - DP parameters
        """
        super(DPGP_centroid_histogram, self).__init__(None,sens,epsilon,delta)

    def prepare_model(self,Xtest,X,step,ys,variances=1.0,lengthscale=1,aggregation='mean',mechanism='laplace'):
        """
        Prepare the model, ready for making predictions
        """
        bincounts, bintotals, binaverages = bin_data(Xtest,X,step,ys,aggregation=aggregation)
        if aggregation=='median':
            raise NotImplementedError
        if aggregation=='mean':            
            sens_per_bin = self.sens/bincounts
        if aggregation=='sum':
            sens_per_bin = self.sens*np.ones_like(bincounts)
        if aggregation=='density':
            sens_per_bin = (self.sens/np.prod(step))*np.ones_like(bincounts)

        if mechanism=='gaussian':
            c = np.sqrt(2*np.log(1.25/self.delta)) #1.25 or 2 over delta?
            bin_sigma = c*sens_per_bin/self.epsilon #noise standard deviation to add to each bin       
            ##add DP noise to the binaverages
            dp_binaverages=binaverages+np.random.randn(binaverages.shape[0])*bin_sigma
        if mechanism=='laplace':
            #note the standard deviation is np.sqrt(2)*the scale parameter
            bin_sigma = np.array(sens_per_bin / self.epsilon) * np.sqrt(2)
            dp_binaverages=binaverages+np.random.laplace(scale=bin_sigma/np.sqrt(2),size=binaverages.shape)
        
        newXtest = Xtest+np.repeat(np.array(step)[None,:],len(Xtest),0)/2 #move the X values to the centres of the bins

        #we don't want outputs that have no training data in.
        keep = ~np.isnan(dp_binaverages)
        finalXtest = newXtest[keep,:]
        final_dp_binaverages = dp_binaverages[keep]
        bin_sigma = bin_sigma[keep]

        #the integral kernel takes as y the integral... 
        #eg. if there's one dimension we're integrating over, km
        #then we need to give y in pound.km
        self.meanoffset = np.mean(final_dp_binaverages)
        final_dp_binaverages-= self.meanoffset
        finalintegralbinaverages = final_dp_binaverages

        finalintegralsigma = bin_sigma
        
        #generate the integral model
        kernel = GPy.kern.RBF(input_dim=newXtest.shape[1], variance=variances, lengthscale=lengthscale)
        #we add a kernel to describe the DP noise added
        kernel = kernel + GPy.kern.WhiteHeteroscedastic(input_dim=newXtest.shape[1], num_data=len(finalintegralsigma), variance=finalintegralsigma**2)
        
        self.model = GPy.models.GPRegression(finalXtest,finalintegralbinaverages[:,None],kernel)
        self.model.sum.white_hetero.variance.fix() #fix the DP noise
        self.model.Gaussian_noise = 0.1 # seems to need starting at a low value!        
        return bincounts, bintotals, binaverages, sens_per_bin, bin_sigma, dp_binaverages
    
    
    def optimize(self,messages=True):
        self.model.optimize(messages=messages)
     
    def draw_prediction_samples(self,Xtest,N=1):
        assert N==1, "DPGP_histogram only returns one DP prediction sample (you will need to rerun prepare_model to get an additional sample)"
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest
        newXtest[:,1::2] = 0
        mean, cov = self.model.predict(newXtest)
        return mean+self.meanoffset, cov
