import numpy as np
import math
import time
import datetime
from fastpredict_noise_func import fast_LOOerror

def stochastic_gradient_LOOcovparamestim(X, Y, n, d, gp, gpsize, N, q, var_noise, y_noise,
                                         covtype, var, niter, linit, linf, lsup, alpha, gamma, a, A, c,
                                         periodmessage, nuggetfactor=1):
    """X: n*d matrix of input points
    Y: n-dimensional vector of scalar output
    n, d: see X, Y
    gp: n-dimensional vector containing the group number of all n points
    gpsize: vector with the sizes of each group
    N: total number of groups (fixed throughout the gradient descent)
    q: total number of points where LOO predictions are computed (fixed as a parameter of the gradient descent).
    covtype: the covariance name
    var: variance
    niter: number of iterations of the algorithm
    linit: d-dimensional vector. Initial vector of correlation lengths (starting point of the gradient descent)
    linf: d-dimensional vector. Minimum value for the correlation lengths (algorithms will refuse a move that goes below these)
    lsup: d-dimensional vector. same with maximum values
    alpha, gamma, a, A, c: scalar parameters of the gradient descent: cf book Bhatnagar et al. chapter 5
    periodmessage: The gradient descent prints it status every periodmessage iterations"""

    """RETURN a list with the following fields:
    mvl: niter*d matrix of all the investigated correlation lengths
    mLOOMSE: niter-dimensional vector of all the (stochastically) evaluated LOO-MSE
    vhatl: d-dimensional vector of the correlation lengths at the end of the descent"""

    "Initialize lcurrent"
    lcurrent = linit

    "Initialize the matrix which will contain the parameter estimation at each iteration"
    mvl = np.matrix(np.zeros((niter, d)), dtype='float64')
    mLOOMSE = np.zeros(niter)

    for i in range(niter):
        if ((i+1) / periodmessage == math.floor((i+1) / periodmessage)):
            ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            print("##################################################################")
            print("iteration: ", i + 1, "\n")
            print("current parameter vector: ", lcurrent, "\n")
            print("System time", theTime, "\n")
            print("##################################################################")

        ai = a / ((A + i + 1) ** alpha)
        deltai = c / ((i + 1) ** gamma)
        np.random.seed(int(time.time()))
        Deltai = 2 * np.random.binomial(d, 0.5) - 1

        "The LOO error is not computed for all n points bur for q << n"
        indices = np.zeros(n, dtype='int32')
        indices[np.random.choice(n, q, replace=False)] = 1

        "computation of the LOO errors and extracts the LOO-MSE"
        lplus = np.exp(np.log(lcurrent) + deltai * Deltai)
        resplus = fast_LOOerror(X=X, Y=Y, n=n, d=d, gp=gp, gpsize=gpsize, N=N, indices=indices,
                                q=sum(indices), var_noise=var_noise, y_noise=y_noise, covtype=covtype,
                                var=var, param=lplus, nuggetfactor=nuggetfactor)
        LOOMSEplus = np.average(resplus['LOOerror'])

        lminus = np.exp(np.log(lcurrent) - deltai * Deltai)
        resminus = fast_LOOerror(X=X, Y=Y, n=n, d=d, gp=gp, gpsize=gpsize, N=N, indices=indices,
                                 q=sum(indices), var_noise=var_noise, y_noise=y_noise, covtype=covtype,
                                 var=var, param=lminus, nuggetfactor=nuggetfactor)
        LOOMSEminus = np.average(resminus['LOOerror'])

        "we obtain a proposal which is accepted if within the bounds"
        lproposal = np.exp(np.log(lcurrent) - ai * (LOOMSEplus - LOOMSEminus) / (2 * deltai * Deltai))

        if sum(lproposal > linf) == d and sum(lproposal < lsup) == d:
            lcurrent = lproposal

        "we keep track of the current proposal and its LOO-MSE"
        mvl[i, :] = lcurrent
        mLOOMSE[i] = 0.5 * (LOOMSEplus + LOOMSEminus)

    "The final estimate is the last proposal"
    vhatl = mvl[niter-1, :]

    "return it together with the previous proposals"
    result = {}
    result['mvl'] = mvl
    result['mLOOMSE'] = mLOOMSE
    result['vhatl'] = vhatl

    return result


"""compute the variance"""
def estim_sigma2(X, Y, var_noise, y_noise, n, d, gp, gpsize, N, covtype, l, nuggetfactor=1):
    "l: d-dimension vector of length-scale"
    
    "return estimate of the variance"
    
    res = fast_LOOerror(X=X, Y=Y, n=n, d=d, gp=gp, gpsize=gpsize, N=N,
                       indices=indices, q=q, var_noise=var_noise, y_noise=y_noise, covtype=covtype, var=1,
                       param=l, nuggetfactor=nuggetfactor)

    return np.average(((Y - res['m_A']) ** 2) / res['v_A'])
