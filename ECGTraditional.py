# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:15:54 2022

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



def RemoveResiduals(Template):
    Residuals = []
    for ECG in ECGData:
        Residuals.append(ECG - Template)
        
    Residuals = np.asarray(Residuals)



    ZeroList = []
    OneList = []
    TwoList = []
    ThreeList = []
    FourList = []

    StartPoint = 0
    EndPoint = 187

    for x in zip(Residuals, ECGLabels):
        if x[1] == 0:
            ZeroList.append(x[0][StartPoint:EndPoint])
        elif x[1] == 1:
            OneList.append(x[0][StartPoint:EndPoint])
        elif x[1] == 2:
            TwoList.append(x[0][StartPoint:EndPoint])
        elif x[1] == 3:
            ThreeList.append(x[0][StartPoint:EndPoint])
        elif x[1] == 4:
            FourList.append(x[0][StartPoint:EndPoint])
        else:
            print("PISSFUCK")
            assert False

    ZeroList = np.asarray(ZeroList)
    OneList = np.asarray(OneList)
    TwoList = np.asarray(TwoList)
    ThreeList = np.asarray(ThreeList)
    FourList = np.asarray(FourList)

    # ArraysToPlot = np.array(
    #     [ZeroList[1:].flatten(),
    #      OneList.flatten(),
    #      TwoList.flatten(),
    #      ThreeList.flatten(),
    #      FourList[:-1].flatten()
    #      ])
    
    ArraysToPlot = np.array(
        [ZeroList[1:],
         OneList,
         TwoList,
         ThreeList,
         FourList[:-1]
         ])

    return ArraysToPlot






ECGDirectoryOne = r"E:\Python\DataSets\ECG\mitbih_train.csv"
ECGDataOne = np.loadtxt(ECGDirectoryOne, delimiter=',')
ECGDirectoryTwo = r"E:\Python\DataSets\ECG\mitbih_test.csv"
ECGDataTwo = np.loadtxt(ECGDirectoryTwo, delimiter=',')

ECGData = np.concatenate((ECGDataOne, ECGDataTwo))

ECGLabels = ECGData[:,-1]
ECGData = ECGData[:, :-1]

assert False



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



ECGData = ResizedECGData








Zero = 0
One = 0
Two = 0
Three = 0
Four = 0

for x in ECGLabels:
    if x == 0:
        Zero+=1
    elif x == 1:
        One += 1
    elif x == 2:
        Two += 1
    elif x == 3:
        Three += 1
    elif x == 4:
        Four += 1
    else:
        print("Cannot Find Number")
        assert False

print(Zero, One, Two, Three, Four)

assert False

ECGLabels = ECGLabels.astype(int)
print(ECGData.shape, ECGLabels.shape)
RandomIndex = random.randint(0, len(ECGData) - 1)
print(RandomIndex, " is rand index and length is ", len(ECGLabels))
SampleImage = ECGData[RandomIndex]
SampleLabel = ECGLabels[RandomIndex]
pyplot.plot(SampleImage)
pyplot.show()
print(SampleLabel)










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


Residuals = []
for ECG in ECGData:
    Residuals.append(ECG - SinCos)
    
Residuals = np.asarray(Residuals)
CreateDictionaryAndExport(Residuals, ECGLabels, 187, "SinCosResiduals")








ArraysToPlot = RemoveResiduals(SinCos)

for x in range(187):
    TheArraysToPlot = np.array(
        [ArraysToPlot[0][:, x],
         ArraysToPlot[1][:, x],
         ArraysToPlot[2][:, x],
         ArraysToPlot[3][:, x],
         ArraysToPlot[4][:, x]
         ])
    
    pyplot.boxplot(TheArraysToPlot)
    pyplot.show()


assert False

Average = []
for x in range(187):
    Average.append(np.mean(ECGData[:, x]))
Average = np.asarray(Average)

ArraysToPlot = RemoveResiduals(Average)

for x in range(187):
    TheArraysToPlot = np.array(
        [ArraysToPlot[0][:, x],
         ArraysToPlot[1][:, x],
         ArraysToPlot[2][:, x],
         ArraysToPlot[3][:, x],
         ArraysToPlot[4][:, x]
         ])
    
    pyplot.boxplot(TheArraysToPlot)
    pyplot.show()


assert False








results = seasonal_decompose(ECGData.flatten(order="K"), period=187)
#results.plot()

ExtractedSeasonality = results.seasonal.reshape((int(len(results.seasonal)/187), 187))[0]

ArraysToPlot = RemoveResiduals(ExtractedSeasonality)

for x in range(187):
    TheArraysToPlot = np.array(
        [ArraysToPlot[0][:, x],
         ArraysToPlot[1][:, x],
         ArraysToPlot[2][:, x],
         ArraysToPlot[3][:, x],
         ArraysToPlot[4][:, x]
         ])
    
    pyplot.boxplot(TheArraysToPlot)
    pyplot.show()

assert False



SinCosResiduals = []
for ECG in ECGData:
    SinCosResiduals.append(ECG - SinCos)
    
SinCosResiduals = np.asarray(SinCosResiduals)



results = seasonal_decompose(SinCosResiduals.flatten(order="K"), period=187)
#results.plot()

ExtractedSeasonality = results.seasonal.reshape((int(len(results.seasonal)/187), 187))[0]


Residuals = []
for ECG in ECGData:
    Residuals.append(ECG - ExtractedSeasonality)
    
Residuals = np.asarray(Residuals)
CreateDictionaryAndExport(Residuals, ECGLabels, 187, "SeasonalityResiduals")






ArraysToPlot = RemoveResiduals(ExtractedSeasonality)
# pyplot.boxplot(ArraysToPlot[0])
# pyplot.show()

for x in range(187):
    TheArraysToPlot = np.array(
        [ArraysToPlot[0][:, x],
         ArraysToPlot[1][:, x],
         ArraysToPlot[2][:, x],
         ArraysToPlot[3][:, x],
         ArraysToPlot[4][:, x]
         ])
    
    pyplot.boxplot(TheArraysToPlot)
    pyplot.show()


assert False


Residuals = []
for ECG in ResizedECGData:
    Residuals.append(ECG - ExtractedSeasonality)
    
Residuals = np.asarray(Residuals)

    










ZeroList = []
OneList = []
TwoList = []
ThreeList = []
FourList = []

StartPoint = 0
EndPoint = 187

for x in zip(Residuals, ECGLabels):
    if x[1] == 0:
        ZeroList.append(x[0][StartPoint:EndPoint])
    elif x[1] == 1:
        OneList.append(x[0][StartPoint:EndPoint])
    elif x[1] == 2:
        TwoList.append(x[0][StartPoint:EndPoint])
    elif x[1] == 3:
        ThreeList.append(x[0][StartPoint:EndPoint])
    elif x[1] == 4:
        FourList.append(x[0][StartPoint:EndPoint])
    else:
        print("PISSFUCK")
        assert False

ZeroList = np.asarray(ZeroList)
OneList = np.asarray(OneList)
TwoList = np.asarray(TwoList)
ThreeList = np.asarray(ThreeList)
FourList = np.asarray(FourList)

ArraysToPlot = np.array(
    [ZeroList[1:].flatten(),
     OneList.flatten(),
     TwoList.flatten(),
     ThreeList.flatten(),
     FourList[:-1].flatten()
     ])

pyplot.boxplot(ArraysToPlot)

pyplot.show()