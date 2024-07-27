import numpy as np
from LDA_discrimination import LDA

def main(m=1000,n=2000,mu1=np.array([1,2]),
         mu2=np.array([3,4]),sigma=np.array([[3,1],[1,3]])):
    """
    This function gives a portable API for several LDA models

    Parameters
    ----------
    m : int, optional
        The first sample size. The default is 1000.
    n : int, optional
        The second sample size. The default is 2000.
    mu1 : numpy array (p*1), optional
        The mean parameter of the first sample.
        The default is np.array([1,2]).
    mu2 : numpy array (p*1), optional
        The mean parameter of the second sample.
        The default is np.array([3,4]).
    sigma : numpy array (p*p), optional
        The covariance parameter of the two samples.
        The default is np.array([[3,1],[1,3]]).

    Returns
    -------
    fun1 : function
        Discriminate a data using sample lda.
    fun2 : function
        Discriminate a data using lda package in sklearn.
    fun3 : function
        Discriminate a data using modified sample lda 1.
    fun4 : function
        Discriminate a data using modified sample lda 2.

    """
    discriminator = LDA(m,n)
    discriminator.generate_data(mu1,mu2,sigma)
    discriminator.estimate()
    err_rate1, fun1 = discriminator.sample_lda()
    err_rate2, fun2 = discriminator.sample_lda()
    err_rate3, fun3 = discriminator.sample_lda()
    err_rate4, fun4 = discriminator.sample_lda()
    
    print(f'The error rate of several models are:')
    print(f'The error rate of original model is {err_rate1}')
    print(f'The error rate of lda package in sklearn is {err_rate2}')
    print(f'The error rate of modified model 1 is {err_rate3}')
    print(f'The error rate of modified model 2 is {err_rate4}')
    
    return fun1,fun2,fun3,fun4