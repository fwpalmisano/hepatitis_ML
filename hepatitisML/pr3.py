import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from collections import defaultdict
import pandas as pd

"""
I chose to work with a dataset from the UCI machine learning repository. This dataset contained information on 
patientes with hepatitis. I chose this dataset because I was recently diagnosed with a rare genetic mutation that affects
my liver and initially presented with hepatitis-like symptoms. Most of the attributes were either Boolean classifiers 
or floating point values. The attributes contained information on symptoms displayed by patients (fatigue, malaise, liver 
palpable, spiders, etc), treatments pursued (steroids, antivirals, etc.), levels of different blood markers (bilirubin, 
albumin, etc), and survival. 

For my first experiment I used decision trees and random forest algorithms to predict survival based on 
symptoms displayed and treatments pursued. I was able to predict survival with 85%+ accuracy with DT and 90%+ with
Random Forests, but I believe that the accuracy of these models are overstated. In this dataset the VAST majority 
of individuals died (an algorithm could just say that every single person in my test group died and probably get 
80%+ accuracy). I would be more impressed with the accuracy of the models if the datasets were more balanced 
in terms of survival. 

For my second experiment I decided to use DT and RF Regressions to predict levels of albumin (protein in blood plasma) 
in the blood of individuals). For whatever reason, it was much easier to predict albumin levels than other 
floating point levels like bilirubin. I was able to predict albumin levels with a 50%+ accuracy with DT and 
70%+ accuracy with RF. 

Example results are listed below.

I have included example decision tree visualizations for both experiments.
"""

df = pd.read_csv('hep.csv', header=0)

# df.head()
df.info()

df = df.drop('protime', axis=1)

df = df[np.isfinite(df['steroids'])]
df = df[np.isfinite(df['fatigue'])]
df = df[np.isfinite(df['malaise'])]
df = df[np.isfinite(df['anorexia'])]
df = df[np.isfinite(df['liver_big'])]
df = df[np.isfinite(df['liver_firm'])]
df = df[np.isfinite(df['spleen_palpable'])]
df = df[np.isfinite(df['spiders'])]
df = df[np.isfinite(df['ascites'])]
df = df[np.isfinite(df['varices'])]
df = df[np.isfinite(df['bilirubin'])]
df = df[np.isfinite(df['alk_phosphate'])]
df = df[np.isfinite(df['sgot'])]
df = df[np.isfinite(df['albumin'])]

df.dropna()

# df.head()
df.info()

feature_names = df.columns.values

X_data = df.drop('albumin', axis=1).values
y_data = df['albumin'].values


indices = np.random.permutation(len(X_data))  # this scrambles the data each time
X_data = X_data[indices]
y_data = y_data[indices]


X_test = X_data[0:30, ]
X_train = X_data[30:, ]

y_test = y_data[0:30]
y_train = y_data[30:]



# depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# best_score = [0.0, 0]
# score_list = []
# MC = range(100)
# MC_result_dict = defaultdict(int)
# #
# for x in MC:
#     print(x)
#     best_depth = [0.0, 0]

#     for y in depth:
#         print(y)

#         dtree = tree.DecisionTreeRegressor(max_depth=y)

#         num_trials = 10
#         sum_total = 0.0
#         for i in range(num_trials):  # run at least 10 times.... take the average cv testing score
#             print(i)
#             #
#             # split into our cross-validation sets...
#             #
#             cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#             cross_validation.train_test_split(X_train, y_train, test_size=0.2) 
#             random_state=0

#             # fit the model using the cross-validation data
#             #   typically cross-validation is used to get a sense of how well it works
#             #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
#             dtree = dtree.fit(cv_data_train, cv_target_train)
#             # print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
#             sum_total += dtree.score(cv_data_test,cv_target_test)
#         avg_score = sum_total/num_trials
#         # print(avg_score)
#         # print(best_depth)
#         # print(best_depth[0])
#         if avg_score > best_depth[0]:
#             best_depth[0] = avg_score
#             best_depth[1] = y
#         if avg_score >= best_score[0]:
#             best_score[0] = avg_score
#             best_score[1] = y

#         # print("Depth", x, ":", avg_score)

#     print("BEST DEPTH: ", best_depth)
#     best_depth_level = best_depth[1]
#     MC_result_dict[best_depth_level] += 1
# print(MC_result_dict)
# print("Best Depth:", max(MC_result_dict.items(), key=lambda a: a[1]))
# print()
# print()
# print(best_score)

# max_depth = 2
# dtree = tree.DecisionTreeRegressor(max_depth=max_depth)
# dtree = dtree.fit(X_train, y_train)
# x_score = dtree.score(X_train,y_train)
# x_results = dtree.predict(X_test)
# print('Predicted with Decision Tree Algorithm: ')
# print()
# print('Projected Model Accuracy: ', x_score)
# print()
# print('Predicted    Actual')
# print()
# for x in range(len(x_results)):
#     print('', x_results[x], '           ', y_test[x])
# print()
# print()

# tree.export_graphviz(dtree, out_file='albuminDT',   # constructed filename!
#     feature_names=feature_names,  filled=True, rotate=False, # LR vs UD
#     leaves_parallel=True)

###################################################################################################################

# max_forest_depth = range(1, 10)
# num_trees = range(20, 200, 10)
# best_depth = [0, 0]
# best_forest_size = 0
# num_trials = 10

# for depth in max_forest_depth:
#     print(depth)
#     for size in num_trees:
#         print(size)
#         sum_total = 0
#         for i in range(num_trials):
#             #
#             # split into our cross-validation sets...
#             #
#             cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#                 cross_validation.train_test_split(
#                     X_train, y_train, test_size=0.2)  # random_state=0

#             rforest = ensemble.RandomForestRegressor(
#                 max_depth=depth, n_estimators=size)

#             # fit the model using the cross-validation data
#             #   typically cross-validation is used to get a sense of how well it works
#             # and tune any parameters, such as the k in kNN (3? 5? 7? 41?,
#             # etc.)
#             rforest = rforest.fit(cv_data_train, cv_target_train)
#             # print("CV training-data score:",
#             #       dtree.score(cv_data_train, cv_target_train))
#             # print("CV testing-data score:",
#             #       dtree.score(cv_data_test, cv_target_test))
#             sum_total += rforest.score(cv_data_test, cv_target_test)
#         avg_score = sum_total / num_trials
#         if avg_score >= best_depth[1]:
#             best_depth[0] = depth
#             best_depth[1] = avg_score
#             best_forest_size = size

# print('Best Depth: ', best_depth)
# print('Best Forest Size: ', best_forest_size)

# max_depth = 3
# rforest = ensemble.RandomForestRegressor(max_depth=max_depth, n_estimators=130)
# rforest = rforest.fit(X_train, y_train)
# x_score = rforest.score(X_train,y_train)
# x_values = rforest.predict(X_test)
# print('Predicted with Random Forest Algorithm:')
# print()
# print('Projected Model Accuracy: ', x_score)
# print()
# print('Predicted    Actual')
# print()
# for x in range(len(x_values)):
#     print('', x_values[x], '           ', y_test[x])

##################################################################
# Experiment 1:

# Looking at hepatitis dataset. Contains variables detailing survival/treatment/symptoms.

# First experiment uses treatment/symptom data to predict survival.

# DTree:
# Best Depth == 1

# Predicted with Decision Tree Algorithm: 

# Projected Model Accuracy:  0.853658536585

# Predicted    Actual

#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  1             2
#  2             2
#  2             2
#  2             2
#  2             2
#  1             1
#  2             2
#  1             1
#  2             1
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  1             2
#  2             2
#  2             2
#  2             2

# RForest:

# Best Depth:  [3, 0.91176470588235292]
# Best Forest Size:  130

# Projected Model Accuracy:  0.926829268293

# Predicted    Actual

#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             1
#  2             2
#  2             2
#  2             1
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  1             2
#  2             2
#  2             2
#  2             2
#  2             2
#  2             2
#  1             1
#  2             2
#  2             2
#  1             2
#  2             2
#  2             2
#  2             2

# Random Forest was better for predicting survival

# In retrospect, this dataset isn't the most impressive application of these tools because almost
# every individual in the data ends up dying. Makes predicting survivors harder.2

####################################################################################

# Experiment 2:

# this experiment uses symptoms/treatment/survival info to predict levels of albumin in the blood
# albumin is the most prevalent protein of the blood plasma

# Projected Model Accuracy:  0.562770829915

# Predicted                 Actual

#  4.06734693878             4.9
#  3.56666666667             4.2
#  4.06734693878             4.2
#  4.06734693878             4.2
#  4.06734693878             4.0
#  4.06734693878             4.7
#  2.80909090909             2.9
#  3.56666666667             3.4
#  2.80909090909             2.8
#  3.56666666667             3.3
#  4.06734693878             4.9
#  3.56666666667             4.4
#  3.56666666667             3.9
#  4.06734693878             3.8
#  4.06734693878             3.9
#  3.56666666667             3.5
#  3.56666666667             3.9
#  4.06734693878             4.1
#  3.56666666667             4.2
#  3.56666666667             3.8
#  4.06734693878             4.0
#  3.56666666667             4.0
#  4.06734693878             4.0
#  4.06734693878             4.0
#  4.06734693878             4.4
#  4.06734693878             4.1
#  4.06734693878             4.5
#  4.06734693878             4.6
#  3.56666666667             3.3
#  4.06734693878             4.0

#RForest:

# Best Depth:  [5, 0.51062037115960845]
# Best Forest Size:  180

# Projected Model Accuracy:  0.714659464785

# Predicted    Actual

#  3.84347930019             4.3
#  4.25020635305             4.2
#  4.15406027439             4.4
#  3.63570554415             3.6
#  2.66946214896             2.4
#  2.81793223443             4.5
#  3.47109130817             3.4
#  4.22742670415             4.0
#  3.71205233878             3.5
#  4.08288120354             3.8
#  4.10548052535             3.8
#  4.17241383628             4.0
#  4.23246421421             4.0
#  3.0691379257              2.9
#  4.24958099065             3.5
#  2.57663425756             2.8
#  4.18244069264             4.3
#  4.15881824782             4.1
#  4.16276996869             3.0
#  4.14516751738             4.0
#  4.22510657983             4.0
#  4.04092005896             4.2
#  4.24178013091             4.3
#  3.37597057286             3.8
#  2.96145554446             3.3
#  4.27581600521             4.0
#  4.15829969526             4.3
#  4.03511841576             3.5
#  3.00588302461             3.0
#  4.10552329118             3.0

# Random Forest made better predictions!
