import numpy as np
import time
import scipy
import csv
from scipy.sparse import csr_matrix
import random
from sklearn import preprocessing
import subprocess
import datetime
import os
import matplotlib.pyplot as plt
from gtk._gtk import accel_group_from_accel_closure

class learner:
    #buffer_len=20
    
    def __init__(self,fp): # done
        self.fp=fp
        self.ftrain_src=fp+'.train.norm'
        self.ftest=fp+'.test.norm'
        self.fheldout=fp+'.heldout.norm'
        self.ftrain=self.ftrain_src+'.current'
        self.fmetadata=fp+'.metadata'
        #time_str=str(datetime.datetime.now().strftime('%d-%m.%H-%M.'))
        #self.fsave=fp+'.'+time_str
        self.fsave=fp+'.'
        self.fmodel=fp+'.model'
        self.fout=fp+'.out'
        #self.max_col=0
        #self.min_col=1000
    def read_metadata(self):
        dim = np.genfromtxt(self.fmetadata, delimiter=' ')
        self.nfeatures=int(dim[1])
        self.nlabels=int(dim[2])
    def find_max_nlabels(self,curr_list):
        max_nlabels=0
        for item in curr_list:
            if len(item)>max_nlabels:
                max_nlabels=len(item)
        return max_nlabels
    def set_max_nlabels(self):
        self.n_top_labels=self.find_max_nlabels(self.Ytest)
    def remove_files(self,list_of_files):
        for file in list_of_files:
            if os.path.exists(file):
                os.remove(file)
    def normalize_data(self,X):
        return preprocessing.normalize(X, norm='l2')
    def write_sparse_data(self,file,Y,X):
        with open(file,'w') as fp:
            wlist=''
            for i in range(X.shape[0]):
                wlist+=Y[i]
                val = X.getrow(i)
                for j in range(val.data.size):
                    wlist +=' '+":".join(map(str, [val.indices[j]+1,val.data[j]])) 
                    
                wlist += '\n'
                #fp.write(wlist)
                
            fp.writelines(wlist)
    def read_sparse_data(self,file,nfeatures):
        with open(file,'r') as fp:
            data_list = fp.readlines()
        #print 'the number of datapoints are %d ' % (len(data_list))
        Y=[]
        sparse_list=[]
        row=0
        max_col=0
        min_col=1000
        #max_label=0
        #print data_list[0]
        for line in data_list:
            part_of_lines=line.strip().split(' ')
            Y.append(part_of_lines[0]) 
            # get max label
            """
            label=part_of_lines[0]
            l = [int(str_l) for str_l in label.split(',')]
            if max_label<max(l):
                max_label=max(l)
            #
            """
            for point in part_of_lines[1:]:
                ind , val = map(float,point.split(':'))
                
                sparse_list.append([row,int(ind-1),val])
                if ind>max_col: max_col=ind
                if ind<min_col: min_col=ind
            row+=1
        #print 'maximum label value is %d ' % (max_label)
        #print 'max index  is %d \nmin index is %d ' % (max_col,min_col)
        
        sparse_list_array = np.asarray(sparse_list)
        
        if nfeatures != max_col:
            print('number of feature mismatch: nfeatures = '+str(nfeatures)+' , nfeatures_here ='+str(max_col) ) 
        X = csr_matrix( (sparse_list_array[:,2],(sparse_list_array[:,0],sparse_list_array[:,1])), shape=(row,nfeatures) )
        return Y,X    
    def create_data(self,nsamples, nfeatures, xdensity, ydensity, nlabels):
        X=csr_matrix(scipy.sparse.random(nsamples, nfeatures, density=xdensity))
        Y_init=scipy.sparse.random(nsamples, nlabels, density=ydensity).todense()
        Y=[]
        for i in range(Y_init.shape[0]):
            Y.append(','.join(map(str,np.nonzero(Y_init[i,:])[1])))
        self.write_sparse_data(self.fp+'.train',Y,X)
        with open(self.fp+'.metadata','w') as fp:
            fp.write(str(nsamples)+' '+str(nfeatures-1)+' '+str(nlabels-1))
    def read_normalize_write_sparse_files(self,list_of_files,nfeatures):
        for file in list_of_files:
            write_file = file+'.norm'
            #print write_file
            Y,X=self.read_sparse_data(file,nfeatures) # 
            self.write_sparse_data(write_file,Y,self.normalize_data(X))
    def load_data(self):
        self.read_metadata()
        self.Ytrain,self.Xtrain=self.read_sparse_data(self.ftrain_src,self.nfeatures) # change Ytrain to list of int
        #self.set_max_nlabels() 
        # read test labels
        self.Ytest=self.read_sparse_data(self.ftest,self.nfeatures)[0]
        # create list of integers from list of strings 
        self.Ytest_int=[]
        for label in self.Ytest: 
            self.Ytest_int.append([int(l) for l in label.split(',')])
        # from train labels, identify rare and sparse labels
        self.rare_labels, self.popular_labels=self.find_popular_n_rare_labels(self.nlabels,self.Ytrain)
        # delete existing train model out file etc
        self.remove_files([self.ftrain])
        
    def read_seed(self,file):
        with open(file,'r') as fp:
            line=fp.readline()
        seed=[int(s) for s in line.strip().split(' ')]
        return seed
    def add_points_select(self,label,val): 
        wlist=label
    
        for i in range(val.data.size):
            wlist +=' '+ ":".join(map(str, [val.indices[i]+1,val.data[i]]))
            #if self.max_col<  val.indices[i]+1: self.max_col=val.indices[i]+1
            #if self.min_col > val.indices[i]+1: self.min_col=val.indices[i]+1
        wlist += '\n'
        with open(self.ftrain,'a') as fp: 
            fp.writelines(wlist)    
        #print self.max_col
        #print self.min_col
    def write_train_data(self,idx):
        for i in idx:
            self.add_points_select(label=self.Ytrain[i], val=self.Xtrain.getrow(i))
    def get_queries(self,fmodel):#  
        # from model file , create Q array and assign to self 
        # put the model file inside init also
        with open(fmodel) as fp:
            model_list = fp.readlines()
        model_list = [x.strip() for x in model_list]
        nclass=int(model_list[0].split(' ')[1])
        labels = map(int,model_list[1].split(' ')[1:])
        nfeatures = int(model_list[2].split(' ')[1])
        
        #if nfeatures!=self.nfeatures:
        #    print 'number of feature mismatch'
        if nclass != self.nlabels:
            print( 'number of label mismatch')
        
        sparse_list_data=[]
        sparse_list_row=[]
        sparse_list_col=[]
        row=0
        nnz_w = int(model_list[3].split(' ')[0])
        if nnz_w !=0:
            print( 'first coef not equal zero')
        
        for line in model_list[4:]:
            part_of_lines=line.split(' ')
            nnz_w=int(part_of_lines[0])
            if nnz_w > 0:
                for point in part_of_lines[1:]:
                    ind , val = map(float,point.split(':'))
                    sparse_list_data.append(val)
                    sparse_list_row.append(row)
                    sparse_list_col.append(int(ind))
            row+=1
        if row!=self.nfeatures:
            print( 'row '+str(row)+', nfeature '+str(nfeature))
        self.Q=csr_matrix((np.array(sparse_list_data),(np.array(sparse_list_row),np.array(sparse_list_col))),shape=(self.nfeatures,self.nlabels))
        #print self.Q.shape    
    def compute_k_scores(self, X, Q, k=None):
        if k==None:
            k=1
        #k=1
        #X=csr_matrix([[1,2],[4,5],[6,2]])
        #Q=csr_matrix([[1,2,3,1],[0,1,3,2]])
        scores = X*Q
        #print X.todense()
        #print Q.todense()
        #print scores.todense()
        #print type(scores)
        predict_list=[]
        
        for sample_score in scores:
            #print sample_score.todense()
            top_indices=sample_score.toarray()[0].argsort()[:-k-1:-1]
            #print top_indices
            top_scores=sample_score.toarray()[0][top_indices]
            #print( sample_score.toarray())
            #print( top_indices)
            #print( top_scores)
            #print('next')
            predict_list.append([[i,s] for i,s in zip(top_indices,top_scores)])
            #print(predict_list)
        return predict_list
    def read_out_file(self): 
        with open(self.fout,'r') as fp:
            lines=fp.readlines()
        list_label=[]
        list_score=[]
        for line in lines:
            list_label.append([int(l.split(':')[0]) for l in line.split(' ')])
            list_score.append([float(l.split(':')[1]) for l in line.split(' ')])
        #print list_label
        #print list_score
        return list_label, list_score
    def compare_lists(self, list1, list2): # check ******************
        for predict,output in zip(list1,list2):# both predict and output are list of integers
            for p,o in zip(predict,output):
                print('modify here')
                #print() 'label %d with score %f where true label %d with true score %f' % (p[0],p[1],o[0],o[1])
            
    def get_acc(self): # checked but Ytest changed later # put get acc inside train and test and decide how to save the values
        # read the out file
        Ytest_predict=self.read_out_file()[0]
        popular_pred, corr_popular_pred, rare_pred, corr_rare_pred, acc_popular, acc_rare = 0,0,0,0,0,0
        for label_true,label_predict_list in zip(self.Ytest_int,Ytest_predict): 
            # if it is popular,put in one bag and if it is rare, put in another bag
            # if it is popular and correct, increment popular_acc
            # if it is rare and correct , increment accordingly            
            label_predict=label_predict_list[0]
            if label_predict in self.popular_labels:  
                popular_pred+=1
                if label_predict in label_true:
                    corr_popular_pred+=1
            if label_predict in self.rare_labels:
                rare_pred+=1
                if label_predict in label_true:
                    corr_rare_pred+=1
        if popular_pred>0:  acc_popular = float(corr_popular_pred)/popular_pred
        if rare_pred>0: acc_rare=float(corr_rare_pred)/rare_pred
        return [acc_popular, acc_rare]
    def train_test(self,k=None,acc_method=None, ftrain=None, ftest=None, l=None):
        if l==None:
            l=.1
        if ftrain==None:
            ftrain=self.ftrain
        if ftest==None:
            ftest=self.ftest
        self.remove_files([ self.fmodel, self.fout])
        # training
        proc = subprocess.Popen(['./PDSparse/multiTrain -l '+str(l)+' -h '+self.fheldout+' '+ ftrain+' '+ self.fmodel],stderr=subprocess.PIPE, shell=True)
        (out,err)=proc.communicate()
        print(err) # d 
        # initialize self.Q from model file
        self.get_queries(self.fmodel) 
        # if acc_method is None, use prediction by multiPred, do not produce out file
        if acc_method==None:
            if k==None:
                proc = subprocess.Popen(['./PDSparse/multiPred'+' '+ftest+' '+self.fmodel+' 1'],stderr=subprocess.PIPE, shell=True)
            else:
                proc = subprocess.Popen(['./PDSparse/multiPred'+' '+ftest+' '+self.fmodel+' '+str(k)],stderr=subprocess.PIPE, shell=True)
            (out,err)=proc.communicate()
            list_of_lines = err.split('\n')
            acc = [float(list_of_lines[1].strip().split('=')[1])]
        else:# acc_method in 'popular_n_rare': produce out file and print rank 1 label 
            proc = subprocess.Popen(['./PDSparse/multiPred'+' '+ftest+' '+self.fmodel+' -p 1 '+self.fout+ ' 1'],stderr=subprocess.PIPE, shell=True)
            (out,err)=proc.communicate()
            print(err) # d 
            # read the out file, check with Ytest to produce popular and rare accuracy
            acc=self.get_acc()
             
        # extra codes
        #subprocess.call(['./PDSparse/multiTrain', '-h', path+self.fheldout, path+self.ftrain, path+self.fmodel])
        #subprocess.check_call(['./PDSparse/multiTrain', '-h',self.fheldout, self.ftrain, self.fmodel])
        #out = subprocess.check_output(['./PDSparse/multiPred &>>',self.ftest,self.fmodel,'1'],stdout=subprocess.PIPE, shell=True)#,'-p','1',self.fout,'1'])
        return acc
    def find_samples_per_label(self,nlabels,Ytrain):
        counter=[0]*nlabels
        for sample in Ytrain: 
            sample_int=[int(s) for s in sample.split(',')]
            for label in sample_int: 
                counter[label]+=1 # create numpy array
        # create histogram
        plt.plot(sorted(counter))
        plt.title('number of samples in different classes')
        plt.xlabel('index of classes after sorting')
        plt.ylabel('number of samples ')
        plt.grid()
        plt.savefig(self.fsave+'sample_per_class.png')
        plt.show()
    def find_popular_n_rare_labels(self,nlabels,Ytrain):
        counter=[0]*nlabels
        for sample in Ytrain: 
            sample_int=[int(s) for s in sample.split(',')]
            for label in sample_int: 
                counter[label]+=1 # create numpy array
            
        #print sorted(counter)
        sorted_labels=np.argsort(np.array(counter))
        # decide threshold
        #threshold=counter[sorted_labels[int(nlabels/2)]]
        with open(self.fp+'.threshold','r') as fp:
            threshold=int(fp.readline().strip())
        # once threshold decided 
        # create two lists named self.popular_labels self.rare_labels
        # find label index after sorting 
        rare_labels=[i for i in range(nlabels) if counter[i] < threshold]
        popular_labels=[i for i in range(nlabels) if counter[i] >= threshold]
        return rare_labels, popular_labels
    def tune(self):
        #self.load_data() 
        
        set_of_lambda=[10**i for i in range(-4,4,1)]
        acc=[]
        #acc_p,acc_r=[],[]
        for l_val in set_of_lambda: 
            acc.append(self.train_test(acc_method='popular_n_rare', l=l_val)[1])
            #acc_p.append(acc[0]) # 
            #acc_r.append(acc[1]) #
        #with open('savep','w') as fp:
            #fp.write(' '.join([str(f) for f in acc_p]))
        #with open('saver','w') as fp:
            #fp.write(' '.join([str(f) for f in acc_r]))
             
        """
        plt.plot(set_of_lambda,acc_p)
        plt.title('popular')
        plt.xlabel('lambda')
        plt.ylabel('accuracy')
        plt.show()
        plt.plot(set_of_lambda,acc_r)
        plt.title('rare')
        plt.xlabel('lambda')
        plt.ylabel('accuracy')
        
        plt.show()
        """
        return set_of_lambda[np.argmax(np.array(acc))]
        
    
    def get_accuracy_old(self,fout): 
        Ytest_predict=[]
        with open(fout,'r') as fp:
            list_of_pred=fp.readlines()
        #print list_of_pred
        for line in list_of_pred:
            labels=[]
            list_words=line.strip().split(' ')
            for words in list_words:
                label,score = words.split(':')
                labels.append(int(label))
            Ytest_predict.append(labels)
            
        #print Ytest_predict
        acc=0
        for i in range(len(self.Ytest)):
            labels_pred = Ytest_predict[i][0]#[:len(labels_actual)]
            #print 'predicted label is '+str(labels_pred)+' where actual label set is '+self.Ytest[i]
            labels_actual = map(int , self.Ytest[i].split(','))
            if labels_pred in labels_actual:
                acc+=1
        #print acc
        acc/=float(len(self.Ytest))
        return acc
    def generate_dummy_query(self):
        nclass=5
        nfeatures=5
        self.Q=csr_matrix(scipy.sparse.rand(m=nclass,n=nfeatures,density=.25))
        self.Q=preprocessing.normalize(self.Q, norm='l2')
        #print self.Q.todense()
    
    
