# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
@author:  Kevin Tranchina ,Filippo Maria Casula ,Giulia Mura , Enrico Ragusa
"""
import pandas as pd
import numpy as np
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import methodsForAnalysisTrainingTest



#read file csv using PANDAS
training = pd.read_csv('training.csv', sep=';')  
# drop Nan value in training.csv
training = training.dropna()
# transformation TS in datetime
training['TS'] = pd.to_datetime(training['TS'])

training['VAR_CLASS'].value_counts()
#16521526 items with VAR_CLASS = 0 disservice 's absence 
#36 items with VAR_CLASS = 1 potential presence of disservice
#472 items with VAR_CLASS =2 disservice 
# The dataset is heavily imbalanced. The approaches analysis for imbalanced dataset are 2
# - approach cost-sensitive
# - approach SMOTE (Synthetic Minority Over-sampling Technique), the dataset is rebalanced with synthetic values
#In this analysis we use a SMOTE approach, because for the realization of a cost-matrix we needed un expert of this sector

#Analysis column TS remove outliers
training['TS'] = pd.to_datetime(training['TS'])
training['TS'].dt.year.unique()#2018
training['TS'].dt.month.unique()#11
training['TS'].dt.day.unique()#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
training['VAR_CLASS'].unique()#1,2,3
training[training['NUM_CLI'] >= 100]

#check column KIT_ID using describe()
training['KIT_ID'].describe()

#count    1.652203e+07
#mean     2.160265e+09
#std      1.224466e+09
#min      1.512491e+06
#25%      1.127272e+09
#50%      2.135241e+09
#75%      3.242074e+09
#max      4.292018e+09

############# Binary Classification ########################################################################
############# TRY PREDICTION WITH ONLY KIT_ID WITH 1 and 2 #################################################
# training e test
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
# overlapping 2 with 1 in order to trasform problem in binary problem
kitWith1or2.loc[:,'VAR_CLASS'] = kitWith1or2.loc[:,'VAR_CLASS'].replace(2,1)

###### HOLD-OUT  #####################################################################################################################
resultRandomForest = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(AdaBoostClassifier(),kitWith1or2)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(DecisionTreeClassifier(),kitWith1or2)

###### Cross Validation ##############################################################################################################
resultRandomForest = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(AdaBoostClassifier(),kitWith1or2)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(DecisionTreeClassifier(),kitWith1or2) 


################ more data ############################################################################################################
kitNot1or2 = training.loc[((training['KIT_ID']!= 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]

########## sampling ###################################################################################################################
kitNot1or21HalfMilion =  kitNot1or2.sample(n=500000,replace=False)

frames = [kitNot1or21HalfMilion, kitWith1or2]
result = pd.concat(frames)

########## predicition with more data #################################################################################################
resultRandomForest = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),result)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(AdaBoostClassifier(),result)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(DecisionTreeClassifier(),result)


    
######################### Multiclass Classification############################################
#Multiclass classification: classification task with more than two classes. Each sample can only be labelled as one class.
#The implementation of these algorithms in the scikit-learn library implements the OvR strategy 
#by default when using these algorithms for multi-class classification.

################ re-read kit with disservice ############################################################################################################
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
frames = [kitNot1or21HalfMilion, kitWith1or2]
result = pd.concat(frames)

########## predicition with more data #################################################################################################
resultRandomForest = methodsForAnalysisTrainingTest.multinominalHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),result)
adaBoostClassifier = methodsForAnalysisTrainingTest.multinominalHoldOutClassifierSmote(AdaBoostClassifier(),result)
resultDecisionTree = methodsForAnalysisTrainingTest.multinominalHoldOutClassifierSmote(DecisionTreeClassifier(),result)


########### DECISION TREE CLASSIFIER #######################################################################
#print(classification_report(y_test, y_pred_DT, target_names=['Var_Class 0', 'Var_Class 1', 'Var_Class 2']))
#              precision    recall  f1-score   support

# Var_Class 0       1.00      1.00      1.00      5952
# Var_Class 1       1.00      1.00      1.00      5950
# Var_Class 2       1.00      1.00      1.00      6033

#    accuracy                           1.00     17935
#   macro avg       1.00      1.00      1.00     17935
#weighted avg       1.00      1.00      1.00     17935


#intero dataset e oversample
#              precision    recall  f1-score   support

# Var_Class 0       1.00      1.00      1.00   4954812
# Var_Class 1       1.00      1.00      1.00   4957736
# Var_Class 2       1.00      1.00      1.00   4956826

#    accuracy                           1.00  14869374
#   macro avg       1.00      1.00      1.00  14869374
#weighted avg       1.00      1.00      1.00  14869374

########### RANDOM FOREST ########################################################################################
#print(classification_report(y_test, y_pred_RF, target_names=['Var_Class 0', 'Var_Class 1', 'Var_Class 2']))

#report
#              precision    recall  f1-score   support

# Var_Class 0       1.00      1.00      1.00      5952
# Var_Class 1       1.00      1.00      1.00      5950
# Var_Class 2       1.00      1.00      1.00      6033

#    accuracy                           1.00     17935
#   macro avg       1.00      1.00      1.00     17935
#weighted avg       1.00      1.00      1.00     17935

#Classification report 's full dataset training
#              precision    recall  f1-score   support

# Var_Class 0       1.00      1.00      1.00   4954812
# Var_Class 1       1.00      1.00      1.00   4957736
# Var_Class 2       1.00      1.00      1.00   4956826

#    accuracy                           1.00  14869374
#   macro avg       1.00      1.00      1.00  14869374
#weighted avg       1.00      1.00      1.00  14869374


################################# LOGISTIC REGRESSION 

#intero dataset e oversample accuracy 0.5359

#Confusion Matrix with dataset with only disservice
#[[2265 1669 2018]
#[ 385 3635 1930]
#[ 992 1686 3355]]

#Confusion Matrix with dataset with full dataset
#[[2562618 1167981 1224213]
# [ 769271 3235253  953212]
# [1105486 1680838 2170502]]

#report
#              precision    recall  f1-score   support

# Var_Class 0       0.33      1.00      0.50      5952
# Var_Class 1       0.00      0.00      0.00      5950
# Var_Class 2       0.00      0.00      0.00      6033

#    accuracy                           0.33     17935
#   macro avg       0.11      0.33      0.17     17935
#weighted avg       0.11      0.33      0.17     17935


#####################################   SVC

#svc with confusion matrix
#[[1504 2017 2431]
# [ 124 3896 1930]
# [ 126 1897 4010]]

#accuracy 0.5247

#report
#              precision    recall  f1-score   support

# Var_Class 0       0.86      0.25      0.39      5952
# Var_Class 1       0.50      0.65      0.57      5950
# Var_Class 2       0.48      0.66      0.56      6033

#    accuracy                           0.52     17935
#   macro avg       0.61      0.52      0.50     17935
#weighted avg       0.61      0.52      0.50     17935





############ PLOT of 3 KIT_ID whose var_char is 1 or 2 #############################################################
kit3409364152 = training.loc[training['KIT_ID'] == 3409364152]
kit1629361016 = training.loc[training['KIT_ID']== 1629361016]
kit2487219358 = training.loc[training['KIT_ID']== 2487219358]

###### Random Forest ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 

###### Adaptive Boosting 1 ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
 
###### Decison Tree Classifier 1 ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 





#############            APPLICATION TO TEST.CSV                          ##################
test = pd.read_csv('test.csv', sep=';')

#prepare test to analisys
T= methodsForAnalysisTrainingTest.prepareTest(test)

######## PREDICTION ADAPTIVE BOOSTING
y_pred = adaBoostClassifier.get_clf().predict(T)
np.unique(y_pred)
test.loc[:,'VAR_CLASS'] = pd.Series(y_pred)

test.loc[:,'VAR_CLASS'].value_counts()

len(test[(test['VAR_CLASS']== 1) | (test['VAR_CLASS']== 2)]['KIT_ID'].unique())## 377
len(test['KIT_ID'].unique())## 2121

test = methodsForAnalysisTrainingTest.fromSecondToDate(test)
kit1629361016 = test.loc[test.loc[:,'KIT_ID'] == 1629361016]# present var_class=0 even when usage is different to zero 
kit1970831019 = test[test['KIT_ID'] == 1970831019]
kit2709380104 = test[test['KIT_ID'] == 2709380104]
kit2130171679 = test[test['KIT_ID'] == 2130171679]
kit1824349749 = test[test['KIT_ID'] == 1824349749]# present var_class=0 even when usage is different to zero 


######## PREDICTION RANDOM FOREST
y_pred = resultRandomForest.get_clf().predict(T)
np.unique(y_pred)
test.loc[:,'VAR_CLASS'] = pd.Series(y_pred)

test.loc[:,'VAR_CLASS'].value_counts()

len(test[(test['VAR_CLASS']== 1) | (test['VAR_CLASS']== 2)]['KIT_ID'].unique())## 377
len(test['KIT_ID'].unique())## 2121

test = methodsForAnalysisTrainingTest.fromSecondToDate(test)
kit1629361016 = test.loc[test.loc[:,'KIT_ID'] == 1629361016]# present var_class=0 even when usage is different to zero 
kit1970831019 = test[test['KIT_ID'] == 1970831019]
kit2709380104 = test[test['KIT_ID'] == 2709380104]
kit2130171679 = test[test['KIT_ID'] == 2130171679]
kit1824349749 = test[test['KIT_ID'] == 1824349749]# present var_class=0 even when usage is different to zero 
########################## END ####################################################################################


"""
Spyder Editor Kevin Tranchina ,Filippo Maria Casula ,Giulia , Enrico Ragusa

This is a temporary script file.
"""