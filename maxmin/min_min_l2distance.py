class min_min_l2distance(learner):
    
    def __init__(self,fp,leaf_size): 
        #print 'inside'
        learner.__init__(self,fp)
        self.active_method='minmin_l2distance'
        self.leaf_size=leaf_size
        
        self.curr_minmax=float('inf')
        self.curr_winner=0
        self.curr_id=0
        self.complexity=0
        self.fcomplexity=self.fsave+'complex'
    def active_select(self):
        for q in self.Q:
            self.mindist(q)
    