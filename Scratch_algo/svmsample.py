import matplotlib.pyplot as plt 
from matplotlib import style 
import numpy as np 
style.use('ggplot')

data_dict_val = {-1:np.array([[1,7],[2,8],[3,8]]) ,1:np.array([[5,1],[6,-1],[7,3]])}


class SupportScatchr:
    def __init__(self,visualization = True):
        self.visula  = visualization
        self.b = None 
        self.w = None
        self.colors = {1:'r',-1:'b'}
        if self.visula:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            pass

    def fit(self,data):
        self.data = data
        opt_dict = {} #mag of w : w and b
        transforms = [[1,1],
                        [-1,1],
                        [-1,-1],
                        [1,-1],
                        ]

        all_data =[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
                    pass
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        step_sizes = [self.max_feature_value *0.01]

        b_range_multiple = 5 #not as precise 
         
        #
        b_multiple = 5

        latest_optimum = self.max_feature_value*10

        for stp in step_sizes:
            w = np.array([latest_optimum,latest_optimum])

            optimized = False
            while not optimized:
                # min w and max b 
                for b in np.arrange(-1*(self.max_feature_value*b_range_multiple) ,self.max_feature_value*b_range_multiple,stp*b_multiple):
                    
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option1 = True 
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i 
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option1 = False 
                            if found_option1:
                                opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                        
                        
                if w[0]<0:
                    optimized = True 
                    print('optimised step')
                else:
                    w = w-stp
                        

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+stp*2

             




    def predict(self,features):
        classi_fication = np.sign(np.dot(np.array(features),self.w)+self.b) #sign (x.w+b)
        
        return classi_fication
        
        
   










