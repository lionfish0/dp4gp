# Differential Privacy for Gaussian Processes

## Installation

Clone the repo and use pip to install;

<pre>
git clone https://github.com/lionfish0/dp4gp.git
cd dp4gp
pip install -e .
</pre>

## Usage

Fairly simple to use. Simply build a normal Gaussian Process regression model in GPy, using any kernel you would like. Then pass it to the dp4gp method of your choice.

<pre>
from dp4gp import dp4gp
import numpy as np
import GPy
import matplotlib.pyplot as plt
%matplotlib inline

X = np.arange(0,10,0.1)[:,None]
Y = np.sin(X)+np.random.randn(len(X),1)*0.4
kern = GPy.kern.RBF(1.0,lengthscale=2.0,variance=1.0)
model = GPy.models.GPRegression(X,Y,kern,normalizer=None)
#model = GPy.models.SparseGPRegression(X,Y,kern,normalizer=None)
model.Gaussian_noise = 0.3

dpgp = dp4gp.DPGP_cloaking(model,ac_sens,epsilon=1.0,delta=0.01)
#dpgp = dp4gp.DPGP_inducing_cloaking(model,ac_sens,epsilon=1.0,delta=0.01)

dpgp.plot(plot_data = True,extent_lower={0:0},extent_upper={0:20},Nits=100,confidencescale=[1.0]);
plt.ylim([-5,5])
</pre>

