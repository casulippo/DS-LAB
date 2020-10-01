# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:41:32 2020

@author: Kevin Tranchina ,Filippo Maria Casula ,Giulia Mura , Enrico Ragusa
"""
import numpy as np
import matplotlib.pyplot as pyplot

#normalize function prerequisite to overlapping plots of usage , num_cli, var_class
def normalizeSeries(seriesInDatframe):
    if(seriesInDatframe.max()-seriesInDatframe.min() ==0):
        seriesInDatframe=0
    else:
        seriesInDatframe = (seriesInDatframe-seriesInDatframe.min())/(
        seriesInDatframe.max()-seriesInDatframe.min())
    return seriesInDatframe

def plotUsageAndNumcliAndVarClassByTS(dataFrame,pred):
    if(pred == False):
        #Normalize 
        dataFrame.loc[:,'USAGE']= normalizeSeries(dataFrame.loc[:,'USAGE'])
        dataFrame.loc[:,'VAR_CLASS']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS'])
        dataFrame.loc[:,'NUM_CLI']= normalizeSeries(dataFrame.loc[:,'NUM_CLI'])
        
        #plot sixe colors
        pyplot.figure(figsize=(15,2))
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'USAGE'],linewidth=1)
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS'], color='darkblue',linewidth=None,edgecolors=None , marker='o')
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'NUM_CLI'], color='aqua',linewidth=3)
        pyplot.xticks(np.arange(min(dataFrame['TS']), max(dataFrame['TS'])+timedelta(days=1), timedelta(days=1)),rotation=70)
        pyplot.legend(('USAGE', 'NUM_CLI', 'VAR_CLASS'))
        pyplot.show()
    else:
        #Normalize 
        dataFrame.loc[:,'USAGE']= normalizeSeries(dataFrame.loc[:,'USAGE'])
        dataFrame.loc[:,'VAR_CLASS']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS'])
        dataFrame.loc[:,'VAR_CLASS_PRED']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS_PRED'])
        dataFrame.loc[:,'NUM_CLI']= normalizeSeries(dataFrame.loc[:,'NUM_CLI'])

        
        #plot sixe colors
        pyplot.figure(figsize=(15,2))
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'USAGE'],linewidth=1)
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS'],linewidth=0.25,color='darkblue',edgecolors=None , marker='.')
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS_PRED'] + 0.25, color='c',edgecolors=None , marker='.')
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'NUM_CLI'],  color='aqua',linewidth=3)
        pyplot.xticks(np.arange(min(dataFrame['TS']), max(dataFrame['TS'])+timedelta(days=1), timedelta(days=1)),rotation=70)
        pyplot.legend((
                'USAGE', 
                       'NUM_CLI', 
                       'VAR_CLASS',
                       'VAR_CLASS_PRED'))
        pyplot.show()
    return pyplot.show()
