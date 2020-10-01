# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:34:29 2020

@author: Kevin Tranchina ,Filippo Maria Casula ,Giulia Mura, Enrico Ragusa
"""


import pandas as pd
from sklearn.metrics import  confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import datetime
from sklearn.model_selection import cross_val_predict
from plotDSLab import plotUsageAndNumcliAndVarClassByTS
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

class Results:

    def __init__(self, confusion_matrix,accuracy_score,clf,dataframe):
        ## private varibale or property in Python
        self.__confusion_matrix = confusion_matrix
        self.__accuracy_score = accuracy_score
        self.__clf = clf
        self.__dataframe = dataframe
        
        #In terms of machine learning, Clf is an estimator instance, which is used to store model.
        #We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.
    def __init__(self):
        ## private varibale or property in Python
        pass
    ## getter method to get the properties using an object
    def get_confusion_matrix(self):
        return self.__confusion_matrix

    ## setter method 
    def set_confusion_matrix(self, confusion_matrix):
        self.__confusion_matrix = confusion_matrix
        
    ## getter method to get the properties using an object
    def get_accuracy_score(self):
        return self.__accuracy_score

    ## setter method
    def set_accuracy_score(self, accuracy_score):
        self.__accuracy_score = accuracy_score
   
    ## getter method to get the properties using an object
    def get_clf(self):
        return self.clf

    ## setter method
    def set_clf(self, clf):
        self.clf = clf
        
    def get_dataframe(self):
        return self.clf

    ## setter method
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

#restituisce in outPut l'accuracy e la confusionMatrix per il classifier in input


def prepareTraining2(dateframe):
    epoch = datetime.utcfromtimestamp(0)
    dateframe.loc[:,'TS'] = pd.to_datetime(dateframe.loc[:,'TS'])
    dateframe.loc[:,'TS'] = dateframe.loc[:,'TS'] - epoch
    dateframe.loc[:,'TS'] = dateframe.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = dateframe.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y =  dateframe.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)


def binaryHoldOutClassifierSmote(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    results = Results()
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    return results

def multinominalHoldOutClassifierSmote(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    results = Results()
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    return results

def binaryCrossValidationClassifierSmote(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    results = Results()
    clf = classifier
    #clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X, y, cv=10)
    score = accuracy_score(y, y_pred)
    confusionMatrix = confusion_matrix(y, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    results.set_dataframe(dataframe)
    return results

def binaryCrossValidationClassifierSmote2(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    distributions = dict(C=uniform(loc=0, scale=4),
                      penalty=['l2', 'l1'])
    clf = classifier
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    search = RandomizedSearchCV(clf, scoring='average_precision', cv=10,
                            n_iter=10, param_distributions=distributions,
                            refit=True, n_jobs=-1)
    search.fit(X, y)
    clf = search.best_estimator_
    results = Results()
    #clf.fit(X_train, y_train)
    y_pred = clf.predict(X)
    dataframe.loc[:,'VAR_CLASS_PRED'] = y_pred
    dataframe = fromSecondToDate(dataframe)
    score = accuracy_score(y, y_pred)
    confusionMatrix = confusion_matrix(y, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    results.set_dataframe(dataframe)
    return results

def testClassifier(clf1,dataframe):
    X,y = prepareTraining2(dataframe)
    results = Results()
    y_pred = clf1.predict(X)
    score = accuracy_score(y, y_pred)
    confusionMatrix = confusion_matrix(y, y_pred, labels=[0, 1])
#   confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf1)
    return results

def prepareTest(test):
    epoch = datetime.datetime.utcfromtimestamp(0)
    test.loc[:,'TS'] = pd.to_datetime(test.loc[:,'TS'])
    test.loc[:,'TS'] = test.loc[:,'TS'] - epoch
    test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()
    test = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    test = test.dropna()
    
    t = test.to_numpy()
    return t

def plotPredKitID(dataframe,clf):
    Z,w = prepareTraining2(dataframe)
    y_pred = clf.predict(Z)
    dataframe.loc[:,'VAR_CLASS_PRED'] = y_pred
    dataframe = fromSecondToDate(dataframe)
    plotUsageAndNumcliAndVarClassByTS(dataframe,True)
    
def plotPredKitIDCrossValidation(dataframe,clf):
    Z,w = prepareTraining2(dataframe)
    y_pred = cross_val_predict(clf, Z, w, cv=5)
    dataframe.loc[:,'VAR_CLASS_PRED'] = y_pred
    dataframe = fromSecondToDate(dataframe)
    plotUsageAndNumcliAndVarClassByTS(dataframe,True)
    
def plotPredKitIDCrossValidation2(dataframe,clf):
    Z,w = prepareTraining2(dataframe)
    y_pred = cross_val_predict(clf, Z, w, cv=5)
    dataframe.loc[:,'VAR_CLASS_PRED'] = y_pred
    dataframe = fromSecondToDate(dataframe)
    plotUsageAndNumcliAndVarClassByTS(dataframe,True)
    
def plotKitID(dataframe,clf):
    plotUsageAndNumcliAndVarClassByTS(dataframe,False)
        
def fromSecondToDate(dataframe):
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataframe.loc[:,'TS'] = pd.to_timedelta(dataframe.loc[:,'TS'],unit='s')+ epoch
    return dataframe
