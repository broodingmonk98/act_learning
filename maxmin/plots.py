from matplotlib import pyplot as plt
import numpy as np
def plot_it():
    file_list=['../results/exp5/d/bibtex.acc.maxquery.0.delta_0.0'+str(i+1) for i in range(10)]
    path=''
    save_ext=''# modify
    title_list=['a']
    for file,plot_title in zip(file_list,title_list):
        with open(file,'r') as fp:
            line=fp.readline()
        
        acc=[float(s) for s in line.strip().split(' ')]
        plt.plot(acc)
        plt.title(plot_title)
        plt.xlabel('lambda')
        plt.ylabel('accuracy')
        plt.show()
        plt.savefig(path+save_ext)
            
            
def plot_your_choice(nacc , file_list, method_list):
    #nacc= 2
    nmethod=len(file_list)
    color=['r','g','b','c','m','y','k','w']
    #method_list=[]
    for i in range(nacc):
        for file,c,m in zip(file_list, color[:nmethod] , method_list ):
            with open(file,'r') as fp:
                lines=fp.readlines()
            acc=[float(line.strip().split(' ')[i]) for line in lines]
            
            plt.plot(acc,c,label=m)
        plt.grid()
        plt.xlabel('samples')
        plt.ylabel('acc')
        plt.legend()
        plt.show()
def plot_active_score_per_iter(path, nfiles):
    path='../results/exp7rep/bibtex.maxquery.count.'
    nfiles=40
    for i in range(nfiles):
        file=path+str(i)
        with open(file,'r') as fp:
            line=fp.readline()
        score=[int(float(s[1:])) for s in line.strip().split(' ')]
        plt.plot(sorted(score))
        # plot
        plt.xlabel('samples after sorting based on score')
        plt.ylabel('active score given by count of labels inside threshold')
        plt.grid()
        plt.ylim(1,159)
        plt.title('count of labels inside threshold')
        plt.savefig(path+str(i)+'.png')
        plt.close()
        
def plot_single_seed():
    """
    nacc=2
    #path='../results/exp5/b/bibtex.acc.maxquery.0.delta_0.1'
    path = '../results/exp7rep/bibtex.acc.'
    seed_list=['.0']
    methods=['maxquery','random']
    methods_legend=methods
    # option 1 methods=['minmax_inner_prod','minmin_inner_prod','minsum_inner_prod','maxquery','random']
    #methods_legend=['delta=0.1'+str(i) for i in range(1,10,1)]
    #methods=[str(m) for m in range(1,10,1)]
    acc_list_outer=[[] for i in range(nacc)]
    color=['r','g','b','c','m','y','k','w','w']
    color = color[:len(methods)]
    acc_method=[[] for m in methods]
    for m in methods:
        acc_seed=[[] for i in seed_list]
        for seed in seed_list:
            acc_inner=[]
            file=path+m+seed
            
            with open(file,'r') as fp:
                lines=fp.readlines()
                acc_list = [[float(s) for s in line.strip().split(' ')] for line in lines]
            for i in range(nacc):
                acc_inner.append(np.array(acc_list)[:][i])
            acc_seed.append(acc_inner)
            
            
            
        
    for  i in range(nacc):
        for acc,c,m in zip(acc_list[i],color,methods_legend):
            # option 1 plt.plot(acc_list[i][j],color[j],label=methods[j]
            plt.plot(acc,c,label=m)
        if nacc==1:
            plt.title('accuracy')
        else:
            if i==0:
                plt.title('accuracy of popular labels ')
            else:
                plt.title('accuracy of rare labels ')    
        plt.legend(loc='upper left')
        plt.ylabel('accuracy of active methods ')
        plt.xlabel('samples')
        #plt.xlim(600,700)
        #plt.ylim(0,1)
        if i==0:
            plt.savefig(path+'.popular.1.png')
        else:
            plt.savefig(path+'.rare.1.png')
        plt.show()
    """

def get_average_over_seed():
    nacc=2
    path='../results/test/a'
    seed_list=[0,1]
    file_list=[path+'.'+str(seed) for seed in seed_list]
    acc=[ [] for i in range(nacc) ]
    
    for file in file_list:
        with open(file,'r') as fp:
            lines=fp.readlines()
        acc_list=[[float( s ) for s in line.strip().split(' ') ] for line in lines ]
        #print acc_list
        acc_array = np.array(acc_list)
        #print acc_array[:,0]
        #print acc_array[:,1]
        for i in range(nacc): # d nacc
            acc[i].append(acc_array[:,i])
    #print acc
    avg_acc=[]
    for i in range(nacc):
        #print np.array(acc[i])
        avg_acc.append(np.average(np.array(acc[i] ),axis=0))
        #print avg_acc
    
    print avg_acc
    
    with open(path,'w') as fp:
        for i in range(len(avg_acc[0])):
            for j in range(nacc):
                fp.write(str(avg_acc[j][i])+' ')
            fp.write('\n')
         
        
        
        
        
        
        
        
        
        
        
        
        
        