
import numpy as np
from main import main
from LDA_discrimination import LDA

##################### demo for main function ###############################
discriminator1,discriminator2,discriminator3,discriminator4 = main()
test_num = 3
test_data = 5 * np.random.rand(test_num,2)
for i in range(test_num):
    for j in range(4):
        if discriminator1(test_data[i,:].reshape(2,1))[0][0] == 1:
            print(f'This data is from the first distribution under model {j+1}')
        else:
            print(f'This data is from the second ditribution under model {j+1}')


#### loop several times to obtain mean and std for mu1 and mu2 are close ####
m=1000
n=2000
mu1=np.array([1,2])
mu2=np.array([1.5,2.5])
sigma=np.array([[3,1],[1,3]])
iter_num = 100
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())


### loop several times to obtain mean and std for mu1 and mu2 are not close ###
mu1=np.array([1,2])
mu2=np.array([5,7])
sigma=np.array([[3,1],[1,3]])
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())



########################### high dimension ####################################
########################### p/(m+n) = 0.25 ####################################
p = 500 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 1
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())

p = 500 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 0.3
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())

########################### p/(m+n) = 0.40 ####################################
p = 800 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 1
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())

p = 800 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 0.3
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())

########################### p/(m+n) = 0.60 ####################################
p = 1200 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 1
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())

p = 1200 
iter_num = 50
sigma = 2*np.eye(p) + 0.1
mu1 = np.random.rand(p)
mu2 = mu1 + 0.3
sample_re = np.zeros((iter_num))
sklearn_re = np.zeros((iter_num))
modified1_re = np.zeros((iter_num))
modified2_re = np.zeros((iter_num))
for k in range(iter_num):
    l = LDA(m,n)
    l.generate_data(mu1=mu1,mu2=mu2,sigma=sigma)
    l.estimate()
    sample_re[k],_ = l.sample_lda()
    sklearn_re[k],_ = l.sklearn_lda()
    modified1_re[k],_ = l.modified_lda1()
    modified2_re[k],_ = l.modified_lda2()
print(sample_re.mean(),sample_re.std())
print(sklearn_re.mean(),sklearn_re.std())
print(modified1_re.mean(),modified1_re.std())
print(modified2_re.mean(),modified2_re.std())