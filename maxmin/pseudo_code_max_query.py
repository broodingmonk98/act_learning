from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
from ball_tree import BallTree
class max_query_active_learner():
    def get_bound(self,x, BT):
        min_d=LA.norm((x.toarray()-BT.loc).toarray(),2)-BT.radius
        upper_b=LA.norm((x.toarray()-BT.loc).toarray(),2)+self.delta
        return min_d,upper_b
    def prune_child(self,x,BT):
        # if child 1 min > child 2 center + delta 
        c1_min_d,c1_upper_b=self.get_bound(x, BT.child1)
        c2_min_d,c2_upper_b=self.get_bound(x, BT.child2)
        if c1_min_d > c2_upper_b:
            return [1,0] 
        if c2_min_d > c1_upper_b:
            return [0,1]
        return [0,0]
                
    def brute_force(self,x,BT):
        
        """
        # check start
        x=    X=csr_matrix(scipy.sparse.random(nsamples, nfeatures, density=density))
        Q=csr_matrix(scipy.sparse.random(nlabels, nfeatures, density=density))
        # check end
        """
        dist_list = [LA.norm((x-q).toarray(),2) for q in BT]
        true_min_dist=min(dist_list)
        dist_list=[d for d in dist_list if d < true_min_dist+self.delta]
        return true_min_dist,dist_list
    def max_query(self, x, BT, depth):
        # find minimum distance of set of query from x 
        min_d=self.get_bound(x,BT)[0]
        
        # if lower bound of minimum of current node is greater than existing upper bound, we may exit the node without visiting 
        if self.upper_b<min_d:
            return
        # Leaf Node        
        if BT.child1 is None:
            print 'I am in child node at depth %d' % (depth)
            # return minimum distance with query node as well as a list of distances from x where distance is less than minimum + delta
            true_min_d,node_dist_list=self.brute_force(x,BT) 
            if true_min_d< self.upper_b:
                if true_min_d>self.min_dist:
                    # few correct distances from node_dist_list adds to self.dist_list
                    self.dist_list.extend([l for l in node_dist_list if l < self.upper_b ])
                else: # this is global minimum
                    self.min_dist=true_min_d # set new global min
                    self.upper_b=self.min_dist+self.delta
                    #node_dist_list becomes new self.dist_list and few distances from old self.dist_list adds to this one
                    node_dist_list.extend([d for d in self.dist_list if d < self.upper_b ])
                    self.dist_list=node_dist_list
        # Internal node    
        else:
            #print 'now in internal node at depth %d' % (depth)
            # check if any child can be prunned 
            prune_flags=self.prune_child(x, BT) # to be defined
            if any(prune_flags): # when all flags are zero
                if prune_flags[0]==0: # child 2 is prunned 
                    self.max_query( x, BT.child1, depth+1)
                else: # child 1 is prunned
                    self.max_query( x, BT.child2, depth+1)
            else: # one of them has been prunned 
                c1_min_d=self.get_bound(x, BT.child1)[0]
                c2_min_d=self.get_bound(x, BT.child2)[0]
                if c1_min_d < c2_min_d: # decide the order of visiting nodes
                    self.max_query( x, BT.child1, depth+1)
                    self.max_query( x, BT.child2, depth+1)
                else:
                    self.max_query( x, BT.child2, depth+1)
                    self.max_query( x, BT.child1, depth+1)
    def active_select(self):
        # generate ball tree for query variables 
        idx=np.array(range(self.Q.shape[0]))
        Qtree=BallTree(self.Q, self.leaf_size, idx)
        # for each data point x , 
        # find minimum distance of set of query as min_dist(x)
        # count the number of query which has l2 distance from x  <= min_dist(x)+self.delta
        max_count_global=0 # contains globally maximum number of query within specified range over all data points
        max_query_x_id=0 # id of datapoint which posses maximum number of query within bound as specified 
        # iterate over all data points
        for x,id in zip(self.Xtrain,range(self.Xtrain.shape[0])):
            # each data point maintains following list of distances of querypoints where the distance values are within 
            # a threshold of minimum distance
            self.dist_list=[]
            self.min_dist=float('inf')
            self.upper_b=self.min_dist+self.delta
            # updates above two variables
            self.max_query(x,Qtree,depth=0)
            # Count number of query within bound for x 
            count=len(self.dist_list) 
            if count > max_count_global:
                max_count_global=count
                max_query_x_id=id
        return max_query_x_id
#------------------------------------------------------------------------------
# This function generates random data points and also random query 
# points, normalizes and returns them.
#------------------------------------------------------------------------------
def gen_random_points(dim, num_X, num_Q):
    #--------------------------------------------------------------------------
    # Create a set of structured random points in two dimensions
    np.random.seed(0)
    X = np.random.random((num_X, dim)) * 2 - 1
    X[:, 1] *= 0.1
    X[:, 1] += X[:, 0] ** 2
    #X = np.random.multivariate_normal([0.5, 0.5], [[0.1, 0.5],[0.5, 0.6]], 30)
    X_normalized = preprocessing.normalize(X, norm='l2')


    # We hardcode the query points so they cluster close to one another
    # on the surface of the sphere!
    Q = [[2,3], [2.5,2.8], [3,3.5], [2.2,3.3] ]
    #Q = np.random.random((num_Q, dim)) * 2 - 1
    #Q[:, 1] *= 0.1
    #Q[:, 1] += Q[:, 0] ** 2
    Q_normalized = preprocessing.normalize(Q, norm='l2')

    return X_normalized, Q_normalized;

def main():
    # generate X and Q
    X,Q=gen_random_points(dim=2, num_X=40, num_Q=8)
    # initialize class variable
    al= max_query_active_learner()
    # load train data X and query variables Q
    al.Xtrain=X 
    al.Q=Q
    al.leaf_size=2
    # initialize threshold
    al.delta=1
    # call active selection algorithm
    al.active_select()