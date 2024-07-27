import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA():
    
    def __init__(self,m=1000,n=2000):
        self.m = m
        self.n = n
    def generate_data(self,mu1=np.array([1,1]),
                      mu2=np.array([3,3]),
                      sigma=np.array([[2,1],[1,2]])):
        self.p = mu1.shape[0]
        self.data_train1 = np.random.multivariate_normal(mu1,sigma,
                                                         size=(self.m,))
        self.data_train2 = np.random.multivariate_normal(mu2,sigma,
                                                          size=(self.n,))
        
        self.data_test1 = np.random.multivariate_normal(mu1,
                                                        sigma,
                                                        size=(self.m,))
        self.data_test2 = np.random.multivariate_normal(mu2,
                                                        sigma,
                                                        size=(self.n,))
        self.data_test = np.concatenate((self.data_test1,self.data_test2))
        self.label_test = np.concatenate((np.ones((self.m,1)),\
                                         np.zeros((self.n,1))))

    
    def estimate(self):
        self.mu1_hat = np.sum(self.data_train1,axis=0) / self.m
        self.mu2_hat = np.sum(self.data_train2,axis=0) / self.n
        self.mu1_hat = self.mu1_hat.reshape(-1,1)
        self.mu2_hat = self.mu2_hat.reshape(-1,1)
        
        self.sigma1_hat = (self.data_train1 - self.mu1_hat.T).T @\
            (self.data_train1 - self.mu1_hat.T) / (self.m - 1)
        self.sigma2_hat = (self.data_train2 - self.mu2_hat.T).T @\
            (self.data_train2 - self.mu2_hat.T) / (self.n - 1)     
        self.sigma_hat = (self.m-1) / (self.m+self.n-2) * self.sigma1_hat\
            + (self.n-1) / (self.m+self.n-2) * self.sigma2_hat
    
    def sample_lda(self):
        dis_fun = lambda x: ((x-(self.mu1_hat+self.mu2_hat)/2).T @ \
            np.linalg.inv(self.sigma_hat) @ (self.mu1_hat-self.mu2_hat) + \
                np.log(self.m/self.n)>=0)+0
        error_rate = np.sum(abs(dis_fun(self.data_test.T) - self.label_test))\
            / (self.m+self.n)
        return error_rate, dis_fun
    
    def sklearn_lda(self):
        clf = LinearDiscriminantAnalysis()
        X = np.concatenate((self.data_train1,self.data_train2))
        y = np.concatenate((np.ones((self.m,1)),np.zeros((self.n,1))))
        clf.fit(X,y.reshape(-1))
        error_rate = np.mean(abs(clf.predict(self.data_test)-\
                                 self.label_test.reshape(self.m+self.n)))
        return error_rate,clf.predict
    
    def modified_lda1(self):
        dis_fun = lambda x: ((x-(self.mu1_hat+self.mu2_hat)/2).T @ \
            np.linalg.inv(self.sigma_hat) @ (self.mu1_hat-self.mu2_hat) + \
                np.log(self.m/self.n)*(self.n+self.m-2)/\
                    (self.n+self.m-self.p-3)>=0)+0
        error_rate = np.sum(abs(dis_fun(self.data_test.T) - self.label_test))\
            / (self.m+self.n)
        return error_rate, dis_fun
    
    def modified_lda2(self):
        dis_fun = lambda x: ((x-(self.mu1_hat+self.mu2_hat)/2).T @ \
            np.linalg.inv(self.sigma_hat) @ (self.mu1_hat-self.mu2_hat) +\
            (self.n+self.m-2)/(self.n+self.m-self.p-3)/2*self.p*\
                (1/self.m-1/(self.n))+np.log(self.m/self.n)*(self.n+self.m-2)/\
                    (self.n+self.m-self.p-3)>=0)+0
        error_rate = np.sum(abs(dis_fun(self.data_test.T) - self.label_test))\
            / (self.m+self.n)
        return error_rate, dis_fun
    

    






























