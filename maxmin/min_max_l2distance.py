from learner import learner
from numpy import linalg as LA
from scipy.sparse import csr_matrix
import numpy as np
from ball_tree import BallTree
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
class min_max_l2distance(learner):
    #curr_minmax=float('inf')
    #curr_winner=0
    
    def __init__(self,fp,leaf_size): 
        #print 'inside'
        learner.__init__(self,fp)
        self.active_method='minmax_l2distance'
        self.leaf_size=leaf_size
        self.max_min_dist=0
        self.max_min_point=0
        self.curr_minmax=float('inf')
        self.curr_winner=0
        self.curr_id=0
        self.complexity=0
        self.fcomplexity=self.fsave+'complex'
        #self.bound=bound
    def create_ball_tree(self): # done
        idx=np.array(range(self.Xtrain.shape[0]))
        self.tree= BallTree(self.Xtrain, self.leaf_size, idx)
    
    def show_ball_tree_n_points(self):
        #------------------------------------------------------------
        # Plot four different levels of the Ball tree
        X=self.Xtrain.toarray()
        fig = plt.figure(figsize=(5, 5))
        fig.subplots_adjust(wspace=0.1, hspace=0.15,
                            left=0.1, right=0.9,
                            bottom=0.05, top=0.9)
        
        
        for level in range(4):
            ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
            
            #ax.scatter(X[:, 0], X[:, 1], s=9)
            self.tree.draw_circle(ax, depth=level)
            
            #ax.scatter(Q[:, 0], Q[:, 1], s=9, color='r')
            #BT.draw_circle(ax, depth=None)
            #ax.set_xlim(-1.35, 1.35)
            #ax.set_ylim(-1.0, 1.7)
            ax.set_title('level %i' % level)
        
        # suptitle() adds a title to the entire figure
        fig.suptitle('Ball-tree Example')
        plt.show()
        
    def load_data(self):
        learner.load_data(self)
        #print 'not calling learner load data'
        #print 'creating ball tree'
        self.create_ball_tree()
        
     
    def create_query_ball(self): # done 
        
        q_center = np.array(self.Q.mean(0))
        #print q_center
        """
        plt.scatter(self.Q.toarray()[:,0], self.Q.toarray()[:,1], s=2)
        plt.scatter(q_center[0,0],q_center[0,1],c='r')
        plt.show()
        """   
        q_radius=0  
        for i in range(self.Q.shape[0]):
            
            #print type(a)
            
            norm_val=LA.norm(self.Q.getrow(i).toarray()-q_center,2) 
            #print norm_val
            if norm_val > q_radius:
                q_radius = norm_val
            
        #print (self.Q-q_center)**2
        #q_radius = np.sqrt(np.max(np.sum((self.Q - q_center) ** 2, 1)))
        return q_center,q_radius
        
    def get_bounds(self,q_center, q_radius,BT): 
        
        
        #print BT.loc.shape 
        #print q_center.shape
        min_dist = LA.norm(BT.loc - q_center) - (BT.radius + q_radius ) 
        max_dist = LA.norm(BT.loc - q_center) + (BT.radius + q_radius )
        
        maxmin_dist = min_dist#max( 0, LA.norm(BT.loc - q_center) - min(BT.radius, q_radius))
        minmax_dist =  max( 0, LA.norm(BT.loc - q_center) -BT.radius)
        
        #print "min %f\nmax %f\nmaxmin %f\nminmax %f\n" % (min_dist, max_dist, maxmin_dist, minmax_dist) 
        return min_dist, max_dist, maxmin_dist, minmax_dist
    

    
    def prune_child_level1(self,q_center, q_radius, BT):# left
        # check how ball tree implementation refer their children as 
        # Compute the bounds for BOTH the children
        c1_min_d, c1_max_d, c1_maxmin_d, c1_minmax_d = self.get_bounds(q_center,q_radius,BT.child1)
        c2_min_d, c2_max_d, c2_maxmin_d, c2_minmax_d = self.get_bounds(q_center,q_radius,BT.child2)

        # If the lower bound (c1_maxmin_d) of child1 is higher 
        # than upper bound (c2_minmax_d) of child2, prune child1
        #if( c1_maxmin_d > c2_minmax_d ):
        #if( c1_maxmin_d > c2_max_d ):
        if( c1_maxmin_d > c2_minmax_d):
            return 1,0

        # If the lower bound (c2_maxmin_d) of child2 is higher 
        # than upper bound (c1_minmax_d) of child1, prune child2
        #if( c2_maxmin_d > c1_minmax_d ):
        #if( c2_maxmin_d > c1_max_d ):
        if( c2_maxmin_d > c1_minmax_d ):
            return 0,1

        # Nothing to prune! 
        return 0,0  
    
    def brute_force_l2_norm(self,X,idx): 
        # Compute the minmax l2-norm distance now.
        minmax_eu = float("inf")
        #print X.getrow(0)
        for x,id in zip(X,idx):
            #print x.todense()
            #print id
            
            max_eu = float("-inf")
            for q in self.Q:
                #print type(x-q)
                
                eu = LA.norm((x - q).toarray())
                if eu > max_eu:
                    max_eu = eu
                    max_x  = x # unnecessary
    
            if max_eu < minmax_eu:
                minmax_eu = max_eu
                minmax_x  = max_x
                minmax_id = id
            #print max_eu
            #print "for x=(%f,%f) , max_eu is %f" % (x.toarray()[0,0], x.toarray()[0,1],max_eu)
        #print "for x =(%f,%f) minmax dist is obtained as %f with idx as %d " % (minmax_x.toarray()[0,0],minmax_x.toarray()[0,1],minmax_eu,minmax_id)
        """
        plt.scatter(X.toarray()[:,0],X.toarray()[:,1], s=9 , c='b')
        
        plt.scatter(self.Q.toarray()[:,0],self.Q.toarray()[:,1], s=9,c='r')
        plt.scatter(minmax_x.toarray()[0,0],minmax_x.toarray()[0,1],s=9,c='g')
        plt.show()
        """
        complexity = X.shape[0]*self.Q.shape[0]
        return minmax_eu, minmax_x, minmax_id, complexity
        #print 'minmax_eu : {0}, winning data point : {1}'.format(minmax_eu, minmax_x)        
        
    def minmaxdist(self,BT, q_center, q_radius, depth):
        
        #global curr_minmax
        #global curr_winner
        
        # Leaf Node        
        if BT.child1 is None:
            print 'I am in child node at depth %d' % (depth)
            
            # We shouldn't be dumping a lot of data here as 
            # we hope to prune more branches and hit the leaf
            # nodes less number of times.
            #dump_ball_contents(depth,BT)
            
            min_d, max_d, maxmin_d, minmax_d = self.get_bounds(q_center,q_radius,BT)
            print 'Current actual minmax = {0}, Ball minmax = {1}'.format(self.curr_minmax,minmax_d)
            #print 'Current actual minmax = {0}, Ball minmax bound = {1}'.format(self.curr_minmax,minmax_d)
            if(minmax_d < self.curr_minmax):
                #print 'brute force check '
                # Now just do a brute force computation
                win_dist, win_x, win_id, curr_complexity = self.brute_force_l2_norm(BT.data, BT.idx)
                self.complexity += curr_complexity
                if(win_dist < self.curr_minmax):
                    #print win_dist
                    #print 'previous minmax was %f where curr minmax is %f' % (self.curr_minmax,win_dist)
                    self.curr_minmax = win_dist
                    self.curr_winner = win_x
                    self.curr_id = win_id
                    #print '----- Current minmax_euclidean, minmax data point = <{0},{1}>'.format(win_dist,win_id)
                    #print '----- Number of points processed = {0}\n'.format(BT.data.shape[0])
            else:
                #print  'therefore not checking'
                return
            
        # Internal Node    
        else:
            #print 'now in internal node at depth %d' % (depth)
            #dump_ball_contents(depth,BT) 
            # Compute the bounds for BOTH the children
            c1_min_d, c1_max_d, c1_maxmin_d, c1_minmax_d = self.get_bounds(q_center,q_radius,BT.child1)
            c2_min_d, c2_max_d, c2_maxmin_d, c2_minmax_d = self.get_bounds(q_center,q_radius,BT.child2)
    
            # Work out what nodes to prune and what to leave!
            #print 'check which child to prune '
            c1_prune, c2_prune = self.prune_child_level1(q_center,q_radius, BT)
            print '---- pruning flags after level 1 = ({0},{1})'.format(c1_prune,c2_prune)
            
            if(c1_prune==0 and c2_prune==1):
                #print '--- pruned child2! ---'
                print 'Current minmax = {0} where child 1 minmax bound = {1}'.format(self.curr_minmax,c1_minmax_d)
                if(c1_minmax_d < self.curr_minmax):
                    #print 'going in child 1 '
                    self.minmaxdist(BT.child1, q_center, q_radius, depth+1 )    
            
            if(c2_prune==0 and c1_prune==1):
                #print '--- pruned child1! ---'
                print 'Current minmax = {0} where child 2 minmax bound = {1}'.format(self.curr_minmax,c2_minmax_d)
                if(c2_minmax_d < self.curr_minmax):
                    #print 'going in child 2'
                    self.minmaxdist(BT.child2, q_center, q_radius, depth+1 )
    
            if(c1_prune==0 and c2_prune==0):
                #print '--- No child pruned, so we order them! ---'            
    
                # First descend down child 1
                if( c1_minmax_d < c2_minmax_d ):
                    #print 'c1 before c2'
                    print 'Current minmax = {0} where child 1 minmax bound = {1}'.format(self.curr_minmax,c1_minmax_d)
                    if(c1_minmax_d < self.curr_minmax):
                        #print 'going in first child  %d' % (depth)
                        self.minmaxdist(BT.child1, q_center, q_radius, depth+1 )
                    print 'Current minmax = {0} where child 2 minmax bound = {1}'.format(self.curr_minmax,c2_minmax_d)
                    if(c2_minmax_d < self.curr_minmax):
                        #print 'going in second child  %d' % (depth)
                        self.minmaxdist(BT.child2, q_center, q_radius, depth+1 )
                else:
                    #print 'c2 before c1'
                    print 'Current minmax = {0} where child 2 minmax bound = {1}'.format(self.curr_minmax,c2_minmax_d)
                    if(c2_minmax_d < self.curr_minmax):
                        #print 'going in second child  %d' % (depth)
                        self.minmaxdist(BT.child2, q_center, q_radius, depth+1 )
                    print 'Current minmax = {0} where child 1 minmax bound = {1}'.format(self.curr_minmax,c1_minmax_d)
                    if(c1_minmax_d < self.curr_minmax):
                        #print 'going in first child  %d' % (depth)
                        self.minmaxdist(BT.child1, q_center, q_radius, depth+1 )
    def write_complexity_ratio(self):
        max_complexity=self.Xtrain.shape[0]*self.Q.shape[0]
        with open(self.fcomplexity,'a') as fp:
            fp.write('search complexity '+str(self.complexity)+' out of total '+str(max_complexity) +' nodes\n')
                
    def active_select(self):
        self.complexity=0
        q_center,q_radius=self.create_query_ball()
        #print 'query ball created'
        #self.prune_child_level1(q_center, q_radius, self.tree)
        #self.get_bounds(q_center, q_radius,self.tree)
        self.minmaxdist(self.tree,q_center,q_radius,0)
        
        #self.write_complexity_ratio()
        #print 'min max dist'
        #print 'current minmax distance %f and winner sample is %s' % (self.curr_minmax,','.join(str(e) for e in list(self.curr_winner.toarray())))
        #idx=0 # to be found 
        #self.brute_force_l2_norm(self.Xtrain)
        #self.minmaxdist(self.tree, q_center, q_radius, depth)
        #print self.curr_id
        #return self.curr_minmax,self.curr_winner,self.curr_id, self.complexity
        return self.curr_id
        
    def check_recursion(self):
        if self.ck_rec==0:
            print 'ck rec %d ' % (self.ck_rec)
            return
        print 'ck rec %d' % (self.ck_rec)
        self.ck_rec-=1
        self.check_recursion()
        return
    
    """Doubt
    should we return only nearest points? How to deal with repeatation?
    """
    # print the ball at level 4 
    # whenever getting dist for a leaf  draw the circle with red
    
    """
    Here BOTH data and query trees should have the same height!
We want to descend both the data and query ball trees simultaneously level-wise
and as we are doing this we are pruning away the balls on either trees that won't 
give us a minmax. Again, I suggest you run this on a very small dataset with just
1 or 2 dimensional data and see if the given algo works... if not, we have to again
tweak it. In theory it should be fine. Big difference is that we must at every level 
check for each combination of q and d balls, which isn't much since we just have
2 balls on each side to check.

#
# Q,D: Root node of (Q)uery tree and (D)ata tree.
# depth: level of tree
#
minmax (Q, D, depth):
    // 1. Pruning phase:
    for each child node of Q do {
         for each child node of D do {
               prune(); // Remove from both Q and D children using our bounds.   
         }
     }
    // 2. Processing phase:
    for each remaining child of Q do {
         for each remaining child of D do {
               if (D is a leaf page) then {  // Q is also a leaf at this point...
                   compute new actual bound.
                   if bound is ok, compute minmax using brute force
               }

               else{ // interior node
                    // Do the same as your single q-ball case! 
                    // Make a recursive call to minmax with correct children for Q and D and depth+1
               }
           }
       } 
       Regarding the above guideline, I have one doubt at the prunning phase. In the processing phase, 
       it has been written that "for each remaining child of Q". But at the prunning phase, for each child of Q, 
       we only prune if any child of D can be prunned. There it is not clear how I can prune any node of Q.
       
       I assume , for each child of Q, wherever possible, we prune the child of D. Then for each child of Q, we run the minmax 
       algorithm over the remaining child of D. In above way, I face following contradiction . 
       
       Let a query node Q has two child Q1 and Q2. Let running minmax over Q1 and Q2  produce d1 and d2. From this two distances,
       we will not be able to accurately find minmax over (union of Q1 and Q2). Because d1 is min(over X) max(over Q1) d(x,q) and 
       d2 is min(over X) max(over Q2) d(x,q). Whenever we consider Q(= union of Q1 and Q2) maximum distances of Q from each datapoint x changes.
       Therefore, it may happen that true minmax distance is neither d1 nor d2. 
       
       The above arguement can be shown by a simple example. 
       Let us consider all points are there on a one dimensional line. 
       X = range(-10,10)
       Q1  = range(-20,-15)
       Q2 = range(15,20)
       d1=distance(-20,-10)=10
       d2=distance(10,20)=10
       true minmax = d= 20
       
       In that case, it shows spliting Q into nodes and finding minmax over child nodes of Q does not help to find true minmax.    
"""
    
"""
change self.tree to self.data_tree
create ball tree for query in active select

"""