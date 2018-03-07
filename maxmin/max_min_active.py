import scipy.sparse.csr_matrix
'''
Created on 12-Dec-2017

@author: paramita
'''
import sys
import csv
import scipy
import sklearn 
from sklearn.neighbors import BallTree
import numpy as np  
from numpy import genfromtxt
import matplotlib
import pandas
from scipy.sparse import find 
import random
from array import array
from random import randint
import subprocess
from StringIO import _complain_ifclosed
from numpy import linalg as LA
# load data . 
# Load CSV
    
                
   
# check whether we can pass sparse data to ball tree

# create random sparse matrix
#X=scipy.sparse.rand(m=80,n=10,density=.25)
# call ball tree
#tree = BallTree(X,leaf_size=10)




"""
def main():
    ftrain=''
    ftest=''
    leaf_size=40
    
    
    almn=active_learner_max_min(ftrain, ftrain_src, ftest, fheldout,leaf_size)
    active_learner_general(almn)
    
if __name__ == '__main__':
    main()
"""


"""
buffer=0
buffer_len=10
filename = 'pima-indians-diabetes.data'
data = genfromtxt(filename, delimiter=',')
X , Y = data[:,:-1],data[:,-1]
#print X.shape
#print Y.shape

split=int(X.shape[0]*.6);

Xtrain , Xtest = X[:split,:],X[split:,:]
Ytrain , Ytest = Y[:split],Y[split:]

#print Xtrain.shape
#print Xtest.shape
#print Ytrain.shape
#print Ytest.shape


test=np.array([[1, 2],[3, 4], [5, 6]])
test_part=test[[0, 2],:]
#print test_part
# split data in test train 
# put points in ball tree
tree = BallTree(Xtrain, leaf_size=10)
acc=[] 
acc_val=[]
while True:
    #print 'you are inside while loop'
    # select sample 
    length_it=460;
    index_to_sel_from=list(xrange(length_it))
    
    k=2
    index_sel=random.sample(index_to_sel_from,k)
    #print index_sel
    
    

    # write to file
    ftrain='testfile'
    #print type(index_sel)
    # remove elements from ball tree
    
    #nsamples=len(index_sel)
    wlist=''
    for i in range(k):
        data_label ,data_val=Ytrain[index_sel[i]], Xtrain[index_sel[i],:]
        #list.append(object)
        dummy,key,value= find(data_val)
        
        data_label=np.array([1,2])
        data_label_list=data_label.tolist()
        
        wlist+= ','.join(map(str, data_label_list))
        wlist+=' '
        for kv in zip(key,value):
             wlist += ":".join(map(str, kv))
             wlist+=' '
        wlist += '\n'
    # print wlist
    
    fptrain = open(ftrain,'a')
    fptrain.writelines(wlist)
    fptrain.close() 

    fheldout=''
    ftrain=''
    fmodel=''
    """
"""
    try: 
        out=subprocess.check_output(['./multiTrain', '-h', fheldout, ftrain, fmodel])
    # need to check exitstatus by subprocess.returncode attribute
        t=0, out
    except subprocess.CalledProcessError as e:
        t = e.returncode , e.message
        print t
        
       
    out=subprocess.check_output(['./multiPred',ftest,fmodel,1,'-p',5,fout])
    """
        # learn pdsparse
"""
    current_acc=0    
    acc_val.append(current_acc)
    
    buffer+=1
    fsave='testfile'
    if buffer > buffer_len:
        buffer=1
        with open(fsave,'a') as f :
            writer=csv.writer(f)
            writer.writerow(acc_val)
            acc.extend(acc_val)
            acc_val=[];
    break
    """  
"""          
fsave_final=''               
with open(fsave_final,'w') as f:
    writer=csv.writer(f)
    writer.writerow(acc)
    
class test:
    a=1
    def f1(self,v):
        self.b=v
        a=v
ob1=test()
ob2=test()
ob1.f1(3)
print ob1.b
print ob1.a
print test.a

print ob2.a



max min code summary

starts with data and limit 
creates the ball tree
    whenever number of points are more than specified in limit,
    take the largest dimension, split it by taking mean of two middle points, 
    split the whole data around this split point, recursively call ball tree to create ball tree further.  
    
   """ 
    

# get acc
# keep on appending data after some time  
# end while 

# save data 


"""
Doubts 
1. multiple run provision should be there or not 

"""


"""
Doubt 
1. how to get seed
2. problem with output - In the paper, should I use test accuracy reported by multiPred or from outputs I should computes. 
As I found, it will give scores for K top classes. In that case how should I distinguish positive from negative labels?
"""
