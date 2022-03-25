# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:42:23 2022

@author: james
"""

import pandas
import numpy as np
import random
import math
from matplotlib import pyplot
import statsmodels
import PIL
import csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf



def CreateDictionaryAndExport(InputData, InputLabels, ECGLength, FileName):
    
    TextLabels = []
    for Label in InputLabels:
        TextToAdd = None
        
        if Label == 0:
            TextToAdd = "Normal"
        elif Label == 1:
            TextToAdd = "Supraventricular"
        elif Label == 2:
            TextToAdd = "Ventricular"
        elif Label == 3:
            TextToAdd = "Fusion"
        elif Label == 4:
            TextToAdd = "Unclassifiable"
        
        if TextToAdd != None:
            for i in range(ECGLength):
                TextLabels.append(TextToAdd)
        else:
            assert False
            
    FlattenedData = InputData.flatten()
    
    TheDict = {"Residual": FlattenedData, "Label": TextLabels}
    TheDataFrame = pandas.DataFrame(data=TheDict)
    
    TheDirectoryToSave = r"E:\Python\Diss" + "\\" + FileName + ".csv"
    TheDataFrame.to_csv(TheDirectoryToSave, index=False)



if __name__ == "__main__":
    ECGDirectoryOne = r"E:\Python\DataSets\ECG\mitbih_train.csv"
    ECGDataOne = np.loadtxt(ECGDirectoryOne, delimiter=',')
    ECGDirectoryTwo = r"E:\Python\DataSets\ECG\mitbih_test.csv"
    ECGDataTwo = np.loadtxt(ECGDirectoryTwo, delimiter=',')

    ECGData = np.concatenate((ECGDataOne, ECGDataTwo))

    ECGLabels = ECGData[:,-1]
    ECGData = ECGData[:, :-1]





    ResizedECGData = []

    for ECG in ECGData:
        NumpyECG = np.expand_dims(ECG, -1)
        NumpyECG = np.flip(NumpyECG, 0)
        
        IndexToCutFrom = 0
        while NumpyECG[IndexToCutFrom] == 0.0:
            IndexToCutFrom += 1
        
        NumpyECG = NumpyECG[IndexToCutFrom:, :]
        NumpyECG = np.flip(NumpyECG, 0)
        PILECG = PIL.Image.fromarray(NumpyECG)
        PILECG = PILECG.resize((1, 187))
        NumpyECG = np.asarray(PILECG)
        NumpyECG = np.squeeze(NumpyECG, 1)
        ResizedECGData.append(NumpyECG)

    ResizedECGData = np.asarray(ResizedECGData)


    assert False


    SinCos = []
    for x in range(187): 
        SinCosSample = math.sin(2*math.pi*float(x)*(1.0/187.0)) + math.cos(2*math.pi*float(x)*(1.0/187.0))
        SinCosSample += 1.5
        SinCosSample /= 3.0
        # SinSample = math.sin(2*math.pi*float(x)*(1.0/187.0))
        # SinSample += 1.0
        # SinSample /= 2.0
        SinCos.append(SinCosSample)
    
    SinCos = np.asarray(SinCos)
    pyplot.plot(SinCos)
    pyplot.show()

    SinCosResiduals = []
    for ECG in ResizedECGData:
        SinCosResiduals.append(ECG - SinCos)
        
    SinCosResiduals = np.asarray(SinCosResiduals)
    CreateDictionaryAndExport(SinCosResiduals, ECGLabels, 187, "SinCosResiduals")
    
    
    
    
    
    results = seasonal_decompose(ResizedECGData.flatten(order="K"), period=187)
    ExtractedSeasonality = results.seasonal.reshape((int(len(results.seasonal)/187), 187))[0]
    
    pyplot.plot(ExtractedSeasonality)
    pyplot.show()
    
    SeasonalityResiduals = []
    for ECG in ResizedECGData:
        SeasonalityResiduals.append(ECG - ExtractedSeasonality)
        
    SeasonalityResiduals = np.asarray(SeasonalityResiduals)
    CreateDictionaryAndExport(SeasonalityResiduals, ECGLabels, 187, "SeasonalityResiduals")
    
    
    
    
    
    
    Average = []
    for x in range(187):
        Average.append(np.mean(ResizedECGData[:, x]))
    Average = np.asarray(Average)
    
    
    AverageResiduals = []
    for ECG in ResizedECGData:
        AverageResiduals.append(ECG - Average)
        
    AverageResiduals = np.asarray(AverageResiduals)
    CreateDictionaryAndExport(AverageResiduals, ECGLabels, 187, "AverageResiduals")
        