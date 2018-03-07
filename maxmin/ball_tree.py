from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix
import random
class BallTree:
    def __init__(self,data,limit,idx):
        self.data=data
        self.limit=limit
        self.idx=idx
        #print limit
        #print 'data'
        #print data.todense()
        #print 'idx'
        #print idx
        
        self.loc = data.mean(0).A
        """
        print 'check type and value of ball tree center'
        print type(self.loc)
        print self.loc
        """
        self.radius=0
        #print data.shape
        for i in range(data.shape[0]):
            norm_val=LA.norm(data.getrow(i)-self.loc,2) 
            #print norm_val
            if norm_val > self.radius:
                self.radius = norm_val
        #print self.radius
        self.child1 = None
        self.child2 = None
        
        if data.shape[0] > limit:
            #print 'parent node' 
            #print data.max(0).toarray()
            #print data.min(0).toarray()
            #print data.max(0)-data.min(0)
            #a=data.max(0)-data.min(0)
            #a=csr_matrix(np.array([1, 2, 3, 2.3, .1, 3.6]))
            largest_dim=np.argmax(data.max(0)-data.min(0))
            #print 'splitting on dim '+ str(largest_dim)
            #print type(data.getcol(largest_dim))
            max_feature = data.getcol(largest_dim).toarray().flatten()
            #print max_feature
             
            i_sort=np.argsort(max_feature)
            #print type(i_sort)
            #print i_sort
            
            N=data.shape[0]
            
            split_point=.5*(max_feature[i_sort[int(N/2)]]+max_feature[i_sort[int(N/2)-1]])
            
            
            #max_feature=np.sort(max_feature)
            #split_point2=.5*(max_feature[int(N/2)]+max_feature[int(N/2)-1])
            #print split_point1
            #print split_point2
            self.child1 = BallTree(self.data[i_sort[:int(N/2)],:],self.limit,self.idx[i_sort[:int(N/2)]])
            self.child2 = BallTree(self.data[i_sort[int(N/2):],:],self.limit,self.idx[i_sort[int(N/2):]])
        """
        else:
            print 'child node'
            print data.todense()"""
    def draw_circle(self, ax, depth=None):
        """Recursively plot a visualization of the Ball tree region"""
        if depth is None or depth == 0:
            r = lambda: random.randint(0,255)
            #print self.loc
            col='#{:02X}{:02X}{:02X}'.format(r(),r(),r())
            circ = Circle((self.loc[0,0],self.loc[0,1]), self.radius, ec=col, fc='none')
            ax.add_patch(circ)
            
            x=self.data.toarray()[:,0]
            y=self.data.toarray()[:,1]
            #print color("#%06x" % random.randint(0, 0xFFFFFF))
            
            ax.scatter(x,y, s=9,c=col)
        
        if self.child1 is not None:
            if depth is None:
                self.child1.draw_circle(ax)
                self.child2.draw_circle(ax)
            elif depth > 0:
                self.child1.draw_circle(ax, depth - 1)
                self.child2.draw_circle(ax, depth - 1)

            
# things to do
# remove node from ball tree 
# query tree will also be sparse 
# should have changed corresponding functions 


# remove nodes
# things to do
# check in leaf node 
# change location and radius 
# GO TO PARENT node
# change loc and radius 
# check whether split changes , 
# if it does, change the rest of the tree ,
# else change only that thread
