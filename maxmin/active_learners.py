from learner import learner
from sklearn.preprocessing import normalize
#from sklearn.neighbors import BallTree
import numpy as np  
from numpy import genfromtxt
import matplotlib
#import pandas
import scipy.sparse
from scipy.sparse import find 
import random
from array import array
from random import randint
import subprocess
import datetime
#from StringIO import _complain_ifclosed
from numpy import linalg as LA
from learner import learner
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt


"""class random_learner(learner):
    def random_select(self):
        pass
    
class active_learner_max_sum(learner):
    def active_select(self):
        pass
class active_learner_max_max(learner):
    def active_select(self):
        pass
    """
class margin_based_active_learner(learner):
    def __init__(self,fp):
        learner.__init__(self, fp)
        self.active_method='minmax_inner_prod'
        self.mask=[]
        
    def read_seed(self,fseed):
        idx=learner.read_seed(self,fseed)
        self.mask=idx
        return idx
        
    def active_select(self,score):
        score[self.mask]=float('inf')
        ind=np.argmin(score)
        #print score[ind]
        #print ind
        self.mask.append(ind)
        return [ind]


class min_max_inner_prod(margin_based_active_learner):
    def __init__(self,fp):
        margin_based_active_learner.__init__(self, fp)
        self.active_method='minmax_inner_prod'
        
    def active_select(self):
        #print self.Q.shape 
        #print self.Xtrain.shape
        self.Q=normalize(self.Q, axis=0, norm='l2')
        score=np.absolute(self.Xtrain*self.Q).max(1).toarray()
        
        return margin_based_active_learner.active_select(self, score)
        
class min_sum_inner_prod(margin_based_active_learner):
    def __init__(self, fp):
        margin_based_active_learner.__init__(self, fp)
        self.active_method='minsum_inner_prod'
    def active_select(self):
        self.Q=normalize(self.Q, axis=0, norm='l2')
        score=np.absolute(self.Xtrain*self.Q).sum(1)
        
        return margin_based_active_learner.active_select(self, score)
class min_min_inner_prod(margin_based_active_learner):
    def __init__(self, fp):
        learner.__init__(self, fp)
        self.active_method='minmin_inner_prod'
    def active_select(self):
        self.Q=normalize(self.Q, axis=0, norm='l2')
        score=np.absolute(self.Xtrain*self.Q).min(1).toarray()
        return margin_based_active_learner.active_select(self, score)
class max_query_active_learner(margin_based_active_learner):
    def __init__(self, fp, delta, save_fig_interval=None):
        margin_based_active_learner.__init__(self, fp)
        self.active_method='maxquery'
        self.delta=delta
        if save_fig_interval!=None:
            self.save_fig_interval=save_fig_interval
            print(self.save_fig_interval)
        else:
            self.save_fig_interval=10000
    def active_select(self, save_fig_count=None):
        # get score
        # for each x, get higest
        # find number of dist from highest to highest -1 
        # take argmax of this count array
        """
        #d start
        nsamples=6
        nlabels=4
        nfeatures=10
        density=.8
        self.delta=.2
        X=csr_matrix(scipy.sparse.random(nsamples, nfeatures, density=density))
        self.Q=csr_matrix(scipy.sparse.random(nlabels, nfeatures, density=density))
        self.Xtrain=sklearn.preprocessing.normalize(X, norm='l2')
        #d end
        """
        #print self.Q.shape
        self.Q=normalize(self.Q, axis=0, norm='l2')
        score_matrix=self.Xtrain*self.Q
        #print score_matrix.shape
        neg_score=[]
        for list_of_score,ind in zip(score_matrix,range(self.Xtrain.shape[0])):
            score_list=list_of_score.toarray()[0].tolist()
            max_score = max(score_list)
            count=0
            for d in score_list: 
                if d > max_score-self.delta: 
                    count+=1
            neg_score.append(float(-count))
        if save_fig_count!=None:
            if save_fig_count%self.save_fig_interval==0:
                plt.plot(np.sort(np.negative(neg_score)))
                plt.xlabel('index of classes')
                plt.ylabel('number of labels which are within threshold')
                plt.grid()
                plt.xlim(1,len(neg_score))
                plt.savefig(self.fp+'.'+self.active_method+'.count.'+str(int(save_fig_count/self.save_fig_interval))+'.png')
                #plt.show()
                
        return margin_based_active_learner.active_select(self, np.array(neg_score))
class min_max_variance_active_learner(margin_based_active_learner):
    def __init__(self, fp, count):
        margin_based_active_learner.__init__(self, fp)
        self.active_method='minmaxdev'
        self.count=count
    def active_select(self):
        
        
        self.Q=normalize(self.Q, axis=0, norm='l2')
        score_matrix=self.Xtrain*self.Q
        dev_list_all=[]
        #score_matrix=np.array([[1,2,4],[3,1,7],[2,8,4],[4,2,7],[3,1,1]])
        for score,i in zip(score_matrix,range(score_matrix.shape[0])):
            if i in self.mask:
                dev_list_all.append([float('inf') for i in range(self.count)])
            else:
                sorted_score = np.sort(score.toarray())[::-1]
                max_dev=0
                dev_list=[]
                for i in range(self.count):
                    dev = abs(sorted_score[0,i+1]-sorted_score[0,i])
                    
                    if dev > max_dev :
                        max_dev = dev
                    dev_list.append(max_dev)
                dev_list_all.append(dev_list)
        final_score = np.array(dev_list_all)
        #print final_score
        #print score_matrix
        idx=[]
        for i in range(self.count):
            idx.append(np.argmin(final_score[:,i]))
        #print final_score
        #print list(set(idx))
        
        self.mask.extend([ i for i in list(set(idx))])
        print('idx selected')
        print(list(set(idx)))
        print('currently mask')
        print(self.mask)
        return list(set(idx)) 
class random_active_learner(learner):
    def __init__(self, fp):
        learner.__init__(self, fp)
        self.active_method='random'
    def load_data(self):
        learner.load_data(self)
        self.population=range(self.Xtrain.shape[0])
    def read_seed(self,fseed):
        idx=learner.read_seed(self,fseed)
        for s in idx: self.population.remove(s)
        return idx
    
    def active_select(self):
        ind = random.sample(self.population, 1)
        self.population.remove(ind[0])
        print('selected '+str(ind[0]))
        return [ind[0]]
        
        
        
        