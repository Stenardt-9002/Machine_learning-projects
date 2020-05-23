self.data = data  
        # save all magnitudeas as key and fin the minimum one 
opt_dict = {}

transofrm = [[1,1],
                [-1,1],
                [-1,-1],
                [1,-1],
]
all_data = []
for yi in self.data:
# for yi in range(len(self.data)):
    for featureset in self.data[yi]:
        for feature in featureset:
            all_data.append(feature)
    pass
# print(all_data)

self.max_feature_value = max(all_data)
self.min_feature_value = min(all_data)
all_data = None


b_range_multiple = 5


b_multiple = 5

latest_optimum = self.max_feature_value * 10