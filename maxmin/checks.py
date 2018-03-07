def check_min_max_basic():
    ncluster=16
    nsample_in_cluster=4
    nlabels=16
    minmaxl2=min_max_l2distance(path, leaf_size)
    X,Q=generate_gaussian_data(ncluster, nsample_in_cluster,nlabels)
    minmaxl2.ntotalsamples=X.shape[0]
    minmaxl2.nfeatures=X.shape[1]
    minmaxl2.nlabels=Q.shape[0]
    minmaxl2.Xtrain=X
    minmaxl2.Ytrain=[]
    minmaxl2.Q=Q
    return minmaxl2

def generate_sparse_data(nsamples,nlabels,nfeatures,density):
    X=csr_matrix(scipy.sparse.random(nsamples, nfeatures, density=density))
    Q=csr_matrix(scipy.sparse.random(nlabels, nfeatures, density=density))
    
    X=X[X.getnnz(1)>0]
    
    Q=Q[Q.getnnz(1)>0]
    
    Y_init=scipy.sparse.random(nsamples, nlabels, density=0.25).todense()
    Y=[]
    for i in range(Y_init.shape[0]):
        Y.append(','.join(map(str,np.nonzero(Y_init[i,:])[1])))
    
    return X,Y,Q
def generate_gaussian_data(ncluster, nsample_in_cluster,nlabels):
    a=-1
    b=1
    X=[]
    ac=0
    bc=.2
    for i in range(ncluster):
        mean=[random.uniform(a,b), random.uniform(a,b) ]
        cov_diag = [random.uniform(ac,bc), random.uniform(ac,bc) ]
        cov= np.diag(cov_diag)
        mat= np.random.multivariate_normal(mean, cov, nsample_in_cluster)
        X.extend(mat)
    X=csr_matrix(X)
    
    mean=[random.uniform(a,b), random.uniform(a,b) ]
    cov= np.diag(np.array(.2*np.random.rand(1,2))[0])
    Q=csr_matrix(np.random.multivariate_normal(mean, cov, nlabels))
    return X,Q
    
def check_min_max_dist_outer(path):
    for nfeatures in range(176,177):
        for nlabels in range(16,96,16):
            #print 'inside'
            
            check_min_max_dist(nfeatures,nlabels,path)
            
        
def check_min_max_dist(nfeatures, nlabels, path):
    count_mistake=0
    leaf_size=8
    nsamples_l=256
    nsamples_u=512#256#9#10#2048
    diff=256
    density=.5
        
    list_dist=[]
    list_ratio=[]
    
    for nsamples in range(nsamples_l,nsamples_u,diff):
        #print '%d th test starts' % (i)
        X,Y,Q=generate_sparse_data(nsamples,nlabels,nfeatures,density)
        X=preprocessing.normalize(X, norm='l2')
        Q=preprocessing.normalize(Q, norm='l2')
        nsamples=X.shape[0]
        nlabels=Q.shape[0]
        """
        print 'shape of x'
        print X.shape
        print 'shape of Q'
        print Q.shape
        """
        # step 1
        al=min_max_l2distance(path,leaf_size)
        al.Xtrain=X
        al.Q=Q
        dist_brute_force,winner_x_exact,id,complexity_brute_force = al.brute_force_l2_norm(al.Xtrain,np.array(range(X.shape[0])))
        
        al.create_ball_tree()
        al.active_select()
        dist=al.curr_minmax
        complexity=al.complexity 
        if dist > dist_brute_force:
            count_mistake+=1
        ratio=float(complexity)/complexity_brute_force
        
        # listing
        list_dist.append([nfeatures, nlabels, nsamples, dist_brute_force, dist])
        list_ratio.append([nfeatures, nlabels, nsamples, ratio])
        
        
        #printing 
        print 'Distance: BF : %f , minmax:%f\n' % (dist_brute_force, dist)
        print 'Ratio:%f\n' % ( ratio)
        
    #writing to file
    with open('distance','a') as fp:
        fp.write('\n'.join(' '.join(str(s) for s in list_elm) for list_elm in list_dist))
        fp.write('\n')
    with open('ratio','a') as fp:
        fp.write('\n'.join(' '.join(str(s) for s in list_elm) for list_elm in list_ratio))
        fp.write('\n')
    with open('mistake','a') as fp:
        fp.write(str(nfeatures)+' '+str(nlabels)+' '+str(count_mistake))
        fp.write('\n')
    
    """
    ncheck=(nsamples_u-nsamples_l)/diff
    print 'number of mistakes %d/%d' % ( count_mistake,ncheck)
    fwrite='ratio_dim_'+str(nfeatures)
    with open(fwrite,'w') as fp:
        fp.write(' '.join(str(r) for r in list_complexity))
    #list_complexity_ratio=np.array(list_complexity)
    #plt.plot(list_complexity_ratio)
    #plt.show()
    plt.plot(list_complexity)
    plt.show()
    """
    # step 2 
    """
    al_origin = min_max_l2distance_original(path,leaf_size)
    al_origin.Xtrain=X
    al_origin.Q=Q
    al_origin.load_data()# will create ball tree 
    dist_origin,winner_ball_tree,id,complexity_origin= al_origin.active_select()
    if dist_origin > dist_brute_force:
        count_mistake_origin+=1
    ratio_origin=float(complexity_origin)/complexity_brute_force"""
