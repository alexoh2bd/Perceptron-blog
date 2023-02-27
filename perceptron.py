import numpy as np


class Perceptron:
    def __init__(self, num_iters= 1000):
        self.max_steps = num_iters
        self.w = None
        self.x = None
        self.bias = None
        self.history = None
        
        
    
    #def predict(self, X, W, y):
        
       # linear_output = y * (X@W)
        #y_predicted = np.where(linear_output > 0 , 1, 0)
        
        #return y_predicted   
    
    def score(self, X, W, y):
        output =  X@W * y
        classifier = np.where(output > 0, 1, 0)
        accuracy = 0
        
        # sum classifiers
        for i in range(len(y)):
            accuracy += classifier[i]
       
        # find average
        accuracy = accuracy/len(y)
        
        return accuracy
    
    def fit(self, X, y): 
       
        # num of observations, num of features
        num_obs, num_features = X.shape
        
         # init weight and bias parameters
        W_ = np.random.rand(num_features+1)
        
        
        # modify X_
        X_ = np.append(X, np.ones((num_obs, 1)), 1)
        
        self.history = []
        
        y_ = 2*y-1
        
        for _ in range(self.max_steps):
                
            # w dot x + bias
            linear_output = y_ * (X_@W_)

            
            # 1 (yi <wt, xi< 0 )
            classifier = np.where(linear_output < 0 , 1, 0)

            # activation, updates if classifier = 1  
            update = np.dot(y_ * classifier, X_ )
            W_ +=   update 
            self.history.append(self.score(X_, W_, y_))
            
        self.w = W_
  
    
    

    
    
    
         