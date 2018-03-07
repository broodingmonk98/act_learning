'''
Created on 12-Dec-2017

@author: paramita
'''
from plots import *
from checks import *
import sys
import os
import csv
import scipy
import sklearn 
#from sklearn.neighbors import BallTree
import numpy as np  
from numpy import genfromtxt
import matplotlib
#import pandas
from scipy.sparse import find 
import random
from random import randint

import datetime
#from StringIO import _complain_ifclosed
from numpy import linalg as LA
from learner import learner
from data_process import data_process
#from data_process import data_process
from min_max_l2distance import min_max_l2distance
from active_learners import *
from scipy.sparse import csr_matrix
#from numpy import linalg as LA
from sklearn import preprocessing
import time
#from pip.utils.logging import indent_log
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import pylab
from OpenSSL.rand import seed
def active_learner_general(al,fseed, fsave_ext=None, max_iter=None):
    seed_idx=fseed.split('.seed.')[-1]
    fsave_acc=al.fsave+'acc.'+al.active_method+'.'+seed_idx
    #fsave_time=al.fsave+'time.'+al.active_method+'.'+seed_idx
    if fsave_ext!=None:
        fsave_acc=fsave_acc+fsave_ext
    #data load
    al.load_data() # read train and test data # * 
    # read seed data from a file
    idx=al.read_seed(fseed)
    # write corresponding train data
    al.write_train_data(idx)
    l=al.tune() # lambda
    print(l)
    #acc = al.train_test(acc_method='popular_n_rare')
    acc=[al.train_test(acc_method='popular_n_rare',l=l)]
    print(acc) # d 
    #with open(fsave_acc,'a') as fp : 
    #    fp.write(' '.join([str(acc_val) for acc_val in acc])+'\n')
    if max_iter==None:
        max_iter=(al.Xtrain.shape[0]- len(idx))
    
    #st=time.time()
    #time_required=[]
    for iter in range(max_iter): 
        #start_al=time.time()
        # from the pool, select point to label
        if al.active_method=='maxquery':
            curr_id = al.active_select(save_fig_count=iter) # d
        else:
            curr_id = al.active_select()
        #time_required.append(time.time()-start_al)
        # write the labeled point to the end of train file
        al.write_train_data(curr_id)
        # check accuracy of learnt model
        #acc= al.train_test(acc_method='popular_n_rare')
        acc.append(al.train_test(acc_method='popular_n_rare', l=l))
        print(acc) # d 
        #print 'next'
        #with open(fsave_acc,'a') as fp :
        #    fp.write(' '.join([str(acc_val) for acc_val in acc])+'\n')
        # check whether method is running
        #if time.time()-st > 1800 :
        #    with open('check','a') as fp:
        #        fp.write(datetime.datetime.now().strftime('%d/%m/%Y %H:%M')+'\n')
        #    st=time.time()
    #with open(fsave_time,'w') as fp:
    #    fp.write('\n'.join(str(t) for t in time_required))
    #for a in acc:
        #print a
    with open(fsave_acc,'w') as fp: 
        fp.write('\n'.join([' '.join(str(a) for a in acc_val) for acc_val in acc]))
def run_tradeoff_exp(al,seed_files,max_iter=None):
    al.remove_files(['status_iter']) 
    for fseed in seed_files:
        if max_iter==None:
            active_learner_general(al,fseed)
        else:
            active_learner_general(al,fseed,max_iter=max_iter)
        with open('status_iter','a') as fp:
            fp.write('Learning complete with seed file as  '+fseed+'\n')
            
    
def main():
    filename='Eur-Lex'
    path='../'+filename+'/' + filename 
    # small checks
    # al=min_max_variance_active_learner(path)
    # al.active_select()
    # return
    """with open(path+'.seed.0','r') as fp:
        line=fp.readline()
    print len(line.strip().split(' '))
    return
    l=learner(path)
    l.read_sparse_data(path+'.train.norm',nfeatures=1837)
    
    sparse_list_data=[1,2,3,4,5,6,7,8]
    sparse_list_row=[0,1,2,3,4,5,6,7]
    sparse_list_col=[0,1,0,1,0,1,0,1]
    Q=csr_matrix((np.array(sparse_list_data),(np.array(sparse_list_row),np.array(sparse_list_col))),shape=(8,2))
    print Q.todense()
    Q=normalize(Q, axis=0, norm='l2')
    print Q.todense()
    return
    """
    # change tradeoff function calls 
    if len(sys.argv)>1:
        input_text = sys.argv[1]
        if input_text in ['tradeoff','check']:
            input_method = sys.argv[2]
        
    else:
        input_text='generate_seed'#tradeoff'#''#'random'#'random'#'check_minmax_l2distance'#'preprocess'
        input_method='random'#'multiple_seed'#'maxquery_vary_delta'
    
    #**********************
    #***Tradeoff Section***
    #**********************
    if input_text in 'tradeoff':
        seed_file_idx=['0']#[str(i) for i in range(5)]
        seed_files=[path+'.seed.'+idx for idx in seed_file_idx]
        max_iter=2
        #**********************************
        if input_method in 'max_query_vary_delta':
            set_of_delta=list(np.arange(.1,.2,.01))
            for fseed in seed_files:
                for delta in set_of_delta:
                    active_learner_general(max_query_active_learner(path,delta=delta), fseed, fsave_ext='.delta_'+str(delta), max_iter=max_iter)
            return
        #*********************************
        if input_method in 'minmax':
            al= min_max_inner_prod(path)
        if input_method in 'minsum':#
            al= min_sum_inner_prod(path)
        if input_method in 'minmin':#
            al= min_min_inner_prod(path)
        if input_method in 'maxquery':#
            delta=.1
            save_fig_interval=50
            al=max_query_active_learner(path,delta=delta, save_fig_interval=save_fig_interval)
        if input_method in 'minmaxvar':
            al=min_max_variance_active_learner(path, count=5)
        if input_method in 'random':#
            al= random_active_learner(path)
        run_tradeoff_exp(al,seed_files,max_iter=max_iter)
    #**********************************************************
    #****************CHECK*************************************
    #**********************************************************
    if input_text in 'check':
        if input_method in 'tuning':
            l=learner(path)
            l.tune()
        if input_method in 'popular_n_rare':
            l=learner(' ')
            acc = l.train_test()
            with open('acc','a') as fp : 
                fp.write(' '.join([str(acc_val) for acc_val in acc])+'\n')
            #d=data_process()
            #d.get_seed(frac=.5,ftrain='train')
            #d.get_label()
            #l.read_out_file()
            #l.decide_popular_n_rare_labels()
            #l.get_acc()
            return
            """
            X,Y,Q=generate_sparse_data(nsamples=5,nlabels=2,nfeatures=2,density=.75)
            al=min_max_inner_prod(path)
            al.Xtrain=X 
            al.Ytrain=Y 
            al.Q=Q 
            for i in range(5):
                al.active_select()
            return"""
        if input_method in 'min_max_l2distance':
            check_min_max_dist_outer(path)    
        if input_method in 'ball_tree':
            X,Y,Q = generate_sparse_data(nsamples=8,nlabels=2,nfeatures=2,density=.75)
            al=min_max_l2distance_lower(path,leaf_size=2)
            al.Xtrain=X 
            al.Ytrain=Y 
            al.Q=Q
            al.create_ball_tree()
            al.brute_force_l2_norm(X,np.array(range(X.shape[0])))
            print('************\n\ndoing active select\n\n *******************')
            al.active_select()
            #al.show_ball_tree_n_points()
        if input_method in 'inner_prod':# checking is left
            k=1
            l=learner(path)
            l.read_metadata()
            # change ftrain  to ftrain_src
            l.ftrain=l.ftrain_src
            # run train and test * create Q, create out file , decide a rank value
            l.train_test(k=k,acc_method='popular_n_rare')# define rank, default k=1
            # read test file, 
            l.Ytest,l.Xtest=l.read_sparse_data(l.ftest, l.nfeatures)
            # compute out file values
            pred_computed = l.compute_k_scores(l.Xtest, l.Q, k=k) # change the output if required later
            Ytest_comp=[]
            score_comp=[]
            for sample_score in pred_computed:
                Ytest_comp.append(sample_score[0][0])
                score_comp.append(sample_score[0][1])
            # compare and report the result
            Ytest_pred, score_pred = l.read_out_file()
            for label1,label2,score1,score2 in zip(Ytest_comp,Ytest_pred, score_comp, score_pred):
                print('label1:'+str(label1)+',label2:'+str(label2)+',score1:'+str(score1)+',score2:'+str(score2)+'\n')
    #**********************************************************
    #****************PLOT*************************************
    #**********************************************************
    if input_text in 'plot':
        if input_method in 'active_score_per_iter':
            path='../results/exp7rep/bibtex.maxquery.count.'
            nfiles=40
            plot_active_score_per_iter(path, nfiles)
        if input_method in 'samples per labels':
            l=learner(path)
            l.read_metadata()
            Ytrain=l.read_sparse_data(l.ftrain_src,l.nfeatures)[0]
            l.find_samples_per_label(l.nlabels,Ytrain)
        if input_method in 'single_seed':
            plot_single_seed()
        if input_method in 'multiple_seed':
            seed_set=[str(i) for i in range(5) ]
            #seed_set=['']
            path='../results/testing/bibtex.acc.'  
            method_list=['minmaxdev','random']
            dp = data_process()
            dp.readfiles_outer(path, method_list, seed_set, num_of_acc=2,fsave=path) # modify
        if input_method in 'ad-hoc':
            path='../results/exp5/e/bibtex.acc.'
            method_list=['maxquery','random']
            file_list = [path+m for m in method_list]
            plot_your_choice(2 , file_list, method_list)
    #**********************************************************
    #****************OTHER*************************************
    #**********************************************************
    if input_text in 'preprocess':
        l = learner(path)
        list_of_files=[path+ext for ext in ['.train', '.test', '.heldout']]
        l.read_metadata()
        l.read_normalize_write_sparse_files(list_of_files,l.nfeatures)
    if input_text in 'min_max_l2distance':
        print('minmax l2 distance')
        leaf_size=20
        bound='lower'
        minmaxl2=min_max_l2distance(path,leaf_size,bound)
        fseed=path+'.seed.0'
        active_learner_general(minmaxl2,fseed)
        #active_learner_general(random_al,frac_seed)
    if input_text in 'generate_seed':
        #print 'seed generation'
        start_idx=0
        nseed_required=1#10
        gen_seed=data_process()
        frac_seed=.1
        ftrain=path+'.train.norm'
        for i in range(nseed_required):
            fseed=path+'.seed.'+str(i+start_idx)
            #print fseed
            #list_of_labels,nlabels =gen_seed.get_label(ftrain)
            seeds=gen_seed.get_seed(frac_seed,ftrain)# how to get nlabels
            #print len(seeds)
            
            #with open(fseed,'w') as fp:
            #    fp.write(' '.join(str(s) for s in seeds))
                
        #seed_generator.get_seed(frac_seed,fseed) 
    if input_text in 'changing_seed':# incomplete
        frac_l=0
        frac_u=0
        frac_diff=0
        gen_seed=data_process()
        for frac in range(frac_l,frac_u, frac_diff):
            seeds=gen_seed.get_seed(frac_seed,ftrain)
            with open(fseed,'w') as fp:
                fp.write(' '.join(str(s) for s in seeds))
if __name__ == '__main__':
    main()



"""
max min code summary

starts with data and limit 
creates the ball tree
    whenever number of points are more than specified in limit,
    take the largest dimension, split it by taking mean of two middle points, 
    split the whole data around this split point, recursively call ball tree to create ball tree further.  
    
   """ 
"""
Doubts 
1. multiple run provision should be there or not 

"""


"""

things to do 
    copy
        maxmin
        bibtex
        make pdsparse
    
"""
         
"""
    X=csr_matrix(scipy.sparse.random(5,5,.75))
    Y=np.array([0,1,2,3,4])
    
    for x , y in zip(X,Y):
        print x.todense()
        print y
    X=csr_matrix([[0,0,0],[0,0,1],[0,0,0],[0,1,1]])
    X=X[X.getnnz(1)>0]
    print X.todense()
    x=X.toarray()[:,0]
    y=X.toarray()[:,1]
        #print color("#%06x" % random.randint(0, 0xFFFFFF))
        
    plt.scatter(x,y, s=9)
    
    plt.scatter(Q.toarray()[:,0],Q.toarray()[:,1], s=9,c='r')
    plt.show()
    
    if input_text in 'minmax':
        print 'minmax'
        minmax = min_max_inner_prod(path)
        #minmax.create_data(nsamples=10, nfeatures=10, xdensity=.5, ydensity=.5, nlabels=10)
        #l = learner(path)
        #list_of_files=[path+ext for ext in ['.train']]
        #l.read_metadata()
        #l.read_normalize_write_sparse_files(list_of_files,l.nfeatures)
        
    if input_text in 'minsum':
        
        print 'minsum'
        minsum = min_sum_inner_prod(path)
        #minmax.create_data(nsamples=10, nfeatures=10, xdensity=.5, ydensity=.5, nlabels=10)
        #l = learner(path)
        #list_of_files=[path+ext for ext in ['.train']]
        #l.read_metadata()
        #l.read_normalize_write_sparse_files(list_of_files,l.nfeatures)
        niter=1
        minsum.remove_files(['status_iter'])
        for i in range(niter):
            fseed=path+'.seed.'+str(i)
            active_learner_general(minsum,fseed)
            with open('status_iter','a') as fp:
                fp.write(str(i)+' iter complete\n')
        
        
    if input_text in 'random':
        print 'random'
        random_al = random_active_learner(path )
        niter=1
        random_al.remove_files(['status_iter'])
        for i in range(niter):
            fseed=path+'.seed.'+str(i)
            active_learner_general(random_al,fseed)
            with open('status_iter','a') as fp:
                fp.write(str(i)+' iter complete\n')
        

    """
        
