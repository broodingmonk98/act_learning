import random
#from sets import Set
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
#from matplotlib import pyplot as plt
import pylab as plt
from symbol import testlist1
class data_process:
	def __init__(self):
		pass
	def get_index(self,file):
		with open(file,'r') as fp:
			data_list=fp.readlines()
		idx=[]
		for line in data_list:
			idx.extend(map(int, line.strip().split(' ')))
		return idx
	def write_indexed_list(self,list,idx,file):
		with open(file,'w') as fp:
			for i in idx:
				fp.write(list[i])
	def split_data(self,fdata, ftrain_sp, ftest_sp, ftrain, ftest):
		with open(fdata,'r') as fp:
			data_list=fp.readlines()
			
		tr_idx = self.get_index(ftrain_sp)
		te_idx = self.get_index(ftest_sp)
		self.write_indexed_list(data_list, tr_idx, ftrain)
		self.write_indexed_list(data_list, te_idx, ftest)
	def get_label(self,ftrain):
		
		list_of_labels=[]
		nlabels=0
		with open(ftrain,'r') as fp:
			lines=fp.readlines()
		#print lines
		for line in lines:
			set_of_labels=map(int,line.strip().split(' ')[0].split(','))
			#print max(set_of_labels)
			if nlabels < max(set_of_labels):
				nlabels=max(set_of_labels)
			list_of_labels.append(set_of_labels)
		nlabels+=1
		return list_of_labels,nlabels 
	def get_seed(self,frac,ftrain):
		list_of_labels,nlabels=self.get_label(ftrain)
		# initialize sample_of_classes
		sample_of_classes=[]
		
		for i in range(nlabels):
			sample_of_classes.append([])
		# it contains list where each element contains index of samples per label
		row=0
		for label_of_sample in list_of_labels:
			for label in label_of_sample:
				sample_of_classes[label].append(row)
			row+=1
		#print sample_of_classes
		
		# create an array of population of each label
		label_population=np.array([len(l) for l in sample_of_classes])
		
		plt.plot(np.sort(label_population))
		#plt.ylim([1,200])
		plt.grid(True)
		plt.xlabel('class labels')
		plt.ylabel('number of samples')
		plt.show()
		
		#print label_population
		# sort it to get labels in order of population
		labels_sorted  = np.argsort(label_population)
		#print labels_sorted
		
		# collects sample of labels in order
		set_of_seed=set([])
		# for each label, randomly sample some fraction of total samples in that label and add to previous set
		for l in labels_sorted: 
			# find samples in label
			current_samples=set(sample_of_classes[l])
			#print current_samples
			# find how many of them are there in seed set
			common_set=set_of_seed.intersection(current_samples)
			#print common_set
			nsample_required=int(len(current_samples)*frac)-len(common_set)
			if nsample_required > 0 :
				# find the rest from the rest of the set
				rest_item=random.sample(current_samples.difference(common_set),k=nsample_required)
				#print type(rest_item)
				#print set(rest_item)
				# add to seed set 
				set_of_seed=set_of_seed.union(set(rest_item))
				#print set_of_seed
			#print 'selected %s, so now set %s ' % (','.join(str(e) for e in np.array(samples)[ind]),','.join(str(e) for e in set_of_seed))
			#print 'current set %s , add %s , now it is %s' % (','.join(str(e) for e in set_of_seed_old),','.join(str(e) for e in samples[ind]),','.join(str(e) for e in set_of_seed))
		return set_of_seed
	def readfile(self,file,array_length=None,num_of_acc=None):
		print file
		with open(file,'r') as fp:
			data = fp.readlines()
		if array_length!=None:
			data=data[:array_length]
		if num_of_acc==None:
			num_of_acc = 1
		data_list=[]
		for i in range(num_of_acc):
			data_list.append([float(line.strip().split(' ')[i])  for line in data ])
		return data_list # contains two list for popular and rare 
	
	def readfiles(self, file, seed_set, array_length=None,num_of_acc=None):
		if num_of_acc==None:
			num_of_acc=1
		data_list_all=[[] for i in range(num_of_acc)]
		for seed in seed_set:
			curr_file = file+'.'+seed
			#print curr_file #d
			#return [] #d
			
			data_list =self.readfile(curr_file,array_length,num_of_acc)
			for acc_no in range(num_of_acc):
				data_list_all[acc_no].append(data_list[acc_no])
		#print data_list_all[0]
		#print data_list_all[1]
		# data_list_all[0] list with popular acc for all seed 
		# data_list_all[1] list with rare acc for all seed
		data_array_avg=[]
		for i in range(num_of_acc):
			data_array = np.array([np.array(dl) for dl in data_list_all[i]])
			data_array_avg.append(list(np.average(data_array,axis=0)))
		#print data_array_avg
		return data_array_avg
	def readfiles_outer(self, path, method_list, seed_set,num_of_acc, fsave=None):
		avg = [self.readfiles(path+method, seed_set, num_of_acc=num_of_acc) for method in method_list ] 
		# return #d
		# method_list =[time_str.split('.')[1] for time_str in time_str_set]#[time_str.split('.')[3] for time_str in time_str_set]#d
		# print method
		
		min_ind = min([len(l1) for l in avg for l1 in l ])
		#print min_ind
		#print [l.shape for l in avg]
		#plot
		color=['r','g','b','c','m','y','k','w']
		for acc_no in range(num_of_acc):
			
			for i,method,c in zip(range(len(avg)),method_list,color[:len(avg)]):
				plt.plot(avg[i][acc_no][:min_ind],c,label=method)
			if acc_no==None:
				plt.title('accuracy')
			else:
				if acc_no==0:
					plt.title('accuracy of popular labels ')
				else:
					plt.title('accuracy of rare labels ')	
			plt.legend(loc='upper left')
			plt.ylabel('accuracy of active methods ')
			plt.xlabel('samples')
			#plt.xlim(1,100)
			#plt.ylim(0,1)
			
			plt.savefig(fsave+str(acc_no)+'.png')
			plt.show()
			
		
	def plot_ratio(self,file,xlabel_string=None, ylabel_string=None):
		
		with open(file,'r') as fp:
			lines=fp.readlines()
		data = [float(line.strip()) for line in lines]
		plt.plot(data)
		plt.ylabel(xlabel_string)
		plt.xlabel(ylabel_string)
		plt.show()
	def generate_synthetic_data(self):
		# generate Q
		# generate X
		# compute Y
		pass
	