import numpy as np

def dp_unnormalise(y,normalisation_parameters):
    """
    new_y = dp_unnormalise(y, normalisation_parameters)
    
    Unnormalises the data, y, using the parameters passed in normalisation_parameters.
    """    
    y = y * normalisation_parameters['std']
    y = y + normalisation_parameters['mean']
    return y
    
def dp_normalise(y, sensitivity, clip='midpoint'):
    """new_y,actual_sensitivity,normalisation_parameters = dp_normalise(y, sensitivity)
    
    Normalises the data to have outputs mean zero, std one.
    It also clips the data to lie within half the sensitivity
    of the data's mid point*, thus inforcing the DP assumptions
    for the sensitivity.
    
    *This behaviour can be modified or disabled by setting the clip parameter:
      - None                = don't clip
      - 'midpoint' (default) = a point halfway between the max and min values
      - 'mean'               = use the mean
      - 'median'             = use the median
    
    The method returns the new y values, the new sensitivity (in the now
    normalised range), and a dictionary of the mean and std to allow future
    unnormalisation"""
    
    if clip is not None:
        middley = None
        if clip=='midpoint': middley = (np.max(y)+np.min(y))/2
        if clip=='mean': middley = np.mean(y)
        if clip=='median': middley = np.median(y)
        assert middley is not None, "clip option invalid"
        
        y[y>middley+sensitivity/2] = middley+sensitivity/2
        y[y<middley-sensitivity/2] = middley-sensitivity/2

    #normalise...
    normalisation_parameters = {}
    normalisation_parameters['mean'] = np.mean(y)
    #ysub = (max(y)+min(y))/2.0 #todo decide what's best to use here...
    new_y = y - normalisation_parameters['mean']
    normalisation_parameters['std'] = np.std(y)
    new_y = new_y / normalisation_parameters['std']
    actual_sensitivity = sensitivity/normalisation_parameters['std']
    return new_y,actual_sensitivity,normalisation_parameters

def compute_Xtest(X,fixed_inputs=[],extent_lower={},extent_upper={},percent_extra=0.1,steps=10):
    """
    Produce a matrix of test points, does roughly what meshgrid does, but for
    arbitrary numbers of dimensions, and handles fixed_inputs, etc etc.
        - Pass X (training data)
        - can also specify extent
            - extent_lower/upper are dictionaries, e.g. {0:5.2,1:4.3}, with the
              index of the dimension and the start value. Note that if you
              specify fixed_inputs then the extents will be overridden.
        - if extend is not specified, the method will use the data's extent
              it adds an additional "percent_extra" to each dimension.
        - steps = number of steps, either an integer or a list of integers (one
              for each dimension)
              
    Example:
            X is 4d, we fix dimensions 0 and 1. We make dimension 2 start at zero
            and have 3 steps in that dimension and 10 in the last dimension, giving
            us 30 points in our output.
        Xtest = compute_Xtest(X, [(1,180e3),(0,528e3)], extent_lower={2:0},steps=[1,1,3,10])   
    """
    rangelist = []
    lower = np.zeros(X.shape[1])
    upper = lower.copy()
    step = lower.copy()
    
    free_inputs = []
    
    if type(steps)==int:
        steps = np.ones(X.shape[1])*steps
        
    for i,(start,finish) in enumerate(np.array([np.min(X,0),np.max(X,0)]).T):
        extra = (finish-start)*percent_extra
        if i not in extent_lower:
            lower[i] = start-extra
        else:
            lower[i] = extent_lower[i]
        if i not in extent_upper:
            upper[i] = finish+extra
        else:
            upper[i] = extent_upper[i]

        step[i] = (upper[i]-lower[i])/steps[i]
        rangelist.append('lower[%d]:upper[%d]:step[%d]'%(i,i,i))
        if i not in [f[0] for f in fixed_inputs]:
            free_inputs.append(i)
        else:
            lower[i] = [f[1] for f in fixed_inputs if f[0]==i][0]
            upper[i] = lower[i]+0.1
            step[i] = 1 #just ensure one item is added

    evalstr = 'np.mgrid[%s]'%(','.join(rangelist))
    res = eval(evalstr)
    
    #handles special case, when ndim=1, mgrid doesn't have an outer array
    if np.ndim(res)==1: 
        res = res[None,:]
    
    res_flat = []
    for i in range(len(res)):
        res_flat.append(res[i].flatten())
    Xtest = np.zeros([len(res_flat[0]),X.shape[1]])
   
    for i, r in enumerate(res_flat):
        Xtest[:,i] = r
        
    return Xtest, free_inputs, step
