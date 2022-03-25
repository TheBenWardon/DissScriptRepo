# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:22:12 2022

@author: james
"""

import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

def LoadECGData():    
    ECGDirectoryOne = r"E:\Python\DataSets\ECG\mitbih_train.csv"
    ECGDataOne = np.loadtxt(ECGDirectoryOne, delimiter=',')
    ECGDirectoryTwo = r"E:\Python\DataSets\ECG\mitbih_test.csv"
    ECGDataTwo = np.loadtxt(ECGDirectoryTwo, delimiter=',')
    
    ECGData = np.concatenate((ECGDataOne, ECGDataTwo))
    
    ECGLabels = ECGData[:,-1]
    ECGData = ECGData[:, :-1]
    
    
    
    ECGLabels = ECGLabels.astype(int)
    
    
    ECGData = np.expand_dims(ECGData, axis=-1)
    
    return ECGData, ECGLabels



def CreateDatasets(InputData, InputLabels):
    
    BufferSize = math.floor(len(InputData)/BatchSize)
    
    TheOverallDataset = (
        tf.data.Dataset.from_tensor_slices((InputData, InputLabels))
        .batch(BatchSize)
        .prefetch(tf.data.experimental.AUTOTUNE)
        )

    TheOverallDataset = TheOverallDataset.shuffle(BufferSize, reshuffle_each_iteration=True)
    
    TrainingSize = int(len(TheOverallDataset) * 0.7)
    ValidationSize = int(len(TheOverallDataset) * 0.15)
    TheTrainingDataset = TheOverallDataset.take(TrainingSize)
    TheValidationDataset = TheOverallDataset.skip(TrainingSize).take(ValidationSize)
    TheTestingDataset = TheOverallDataset.skip(TrainingSize + ValidationSize)
    
    return TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset



class estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred



if __name__ == "__main__":
    BatchSize = 32
    BufferSize = 300
    
    InputData, LabelData = LoadECGData()
    #TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateDatasets(InputData, LabelData)
    
    ModelDir = r"E:\Python\Diss\Models\ECGClassification_MLP_4096"
    MyModel = tf.keras.models.load_model(ModelDir)
    
    ModelPredictions = MyModel.predict(InputData)
    ModelPredictions = np.argmax(ModelPredictions, axis=1)
    
    # ConfusionMatrix = tf.math.confusion_matrix(labels=LabelData, predictions=ModelPredictions)
    # print(ConfusionMatrix)
    
    ClassNames = ["N", "P", "asd", "ddd", "8"]
    
    Classifier = estimator(MyModel, ClassNames)
    plot_confusion_matrix(Classifier, X=InputData, y_true=LabelData, normalize="true", cmap="cividis")
    plt.savefig("BiRNN_LSTM_Full_Matrix.png", dpi=300)
