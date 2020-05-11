import matplotlib.pyplot as plt 
from matplotlib import style 
import numpy as np 
style.use('ggplot')



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
        opt_dict = {}
        transforms = [[1,1],
                        [-1,1],
                        [-1,-1],
                        [1,-1],
                        ]

        all_data =[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append()
                    pass



    def predict(self,features):
        classi_fication = np.sign(np.dot(np.array(features),self.w)+self.b)
        
        return classi_fication
        pass
        
   






data_dict_val = {-1:np.array([[1,7],[2,8],[3,8]]) ,1:np.array([[5,1],[6,-1],[7,3]])}




