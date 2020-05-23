import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 

style.use('ggplot')
data_dick1 = {-1:np.array([[1,7],
                            [2,8],
                            [3,8],

                        ]),

              1:np.array([[5,1],
                            [6,-1],
                            [7,3],

                        ])
                                 
            }


class SVM:
    def __init__(self,visualise = True):
        # super().__init__()
        self.visualise = visualise 
        self.colors = {1:'r',-1:'b'}
        if self.visualise:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        # self.w = None 
        # self.b = None


    def fit(self,data):
        self.data = data  
        # save all magnitudeas as key and fin the minimum one 
        opt_dict = {}

        transofrm = [[1,1],
                        [-1,1],
                        [-1,-1],
                        [1,-1] ]
        all_data = []
        for yi in self.data:
        # for yi in range(len(self.data)):
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
            
        # print(all_data)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        # print(self.max_feature_value)
        # print(self.min_feature_value)

        # step_sizes = [self.max_feature_value * 0.1,self.max_feature_value * 0.01,self.max_feature_value * 0.001]
                        # self.max_feature_value * 0.0001,
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

                        
        #extremely expensive
        b_range_multiple = 5
        # b_range_multiple = 2


        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes :
            w = np.array([latest_optimum,latest_optimum])
            #we can do this
            optimised = False 

            # while not optimised:
            #     for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
            #         for transofrvar in transofrm:
            #             w_t = w*transofrvar
            #             found_option = True 
            #             for i in self.data:
            #                 for xi in self.data[i]:
            #                     yi = i 
            #                     if not((yi*(np.dot(w_t,xi)))+b)>=1:
            #                         found_option = False 
            #                         # break 
            #                 # if found_option == False:
            #                 #     break 
            #             if found_option:
            #                 opt_dict[np.linalg.norm(w_t)] = [w_t,b]
            #     if w[0]<0:
            #         optimised = True
            #     else :
            #         w= w-step #vector - scalar each value will be scaled down


            while not optimised:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transofrm:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimised = True
                else:
                    w = w - step


            norms = sorted([n for n in opt_dict])

            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

    


    def predict(self,features):
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0 and self.visualise:
            self.ax.scatter(features[0],features[1],s = 200,marker = '*', c = self.colors[classification])

        return classification

    def visualifucse(self):
        [[self.ax.scatter(x[0],x[1],s= 100,color = self.colors[i]) for x in data_dick1[i]] for i in data_dick1]

        # v = x.w+b 
        # psv = 1 
        # nsv = -1
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9 , self.max_feature_value*1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]


        #postive support vector
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],"k")


        #negative support vector
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],"k")


        zer1 = hyperplane(hyp_x_min,self.w,self.b,0)
        zer2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[zer1,zer2])


        plt.show()


# data_dict = {-1:np.array([[1,7],
#                           [2,8],
#                           [3,8],]),
             
#              1:np.array([[5,1],
#                          [6,-1],
#                          [7,3],])}



svmobj = SVM()
svmobj.fit(data_dick1)
# svmobj.visualifucse()
# svmobj.visualifucse()


predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]



for p in predict_us:
    svmobj.predict(p)

svmobj.visualifucse()























