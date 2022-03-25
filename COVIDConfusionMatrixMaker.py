# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:15:11 2022

@author: james
"""

import tensorflow as tf
import random
import numpy as np
import math
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import os


#Define Config Variables
ImageWidth = 500
ImageHeight = 100 #78 #1025

FRAME_SIZE = 2048
HOP_SIZE = 1024
CutOffSize = 50
MaxAudioFileLength = 500
MinAudioFileLength = 1

MelBinNumber = 80
BinsToTake = 1025 #78 #1025
MelLowerEdgeHertz = 100
MelUpperEdgeHertz = 24000

BatchSize = 32
BufferSize = 1000



# ModelName = "CovidClassification_Multi_Spectrogram-{}".format(int(time.time()))
# Dir = r"E:\Python\Diss\Logs\\"
# Dir += ModelName
# print(Dir)
# MyTensorBoard = TensorBoard(log_dir=Dir)



def load_py(a):
    # Arguments to py_function are eager tensors, so we can use `.numpy()` to get their string values.
    # tf.print(type(a))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # tf.print(a.numpy())  # b'a'
    
    a = a.numpy()
    
    #tf.print(a)
    
    #a = np.array([[100, 100], [100, 100]])
    
    ReturnArray = 0
    with open(a, "rb") as FILE:
        ReturnArray = np.load(FILE)
    
    ReturnArray = tf.convert_to_tensor(ReturnArray)
    #ReturnLabel = tf.strings.to_number(tf.strings.split(a, sep="_")[1], out_type=tf.dtypes.int32)
    
    ReturnLabel = tf.constant([1], dtype=tf.dtypes.int32)
    a = tf.strings.split(a, sep="_")[1]
    
    #tf.print(type(str(a.numpy())))
    
    if a.numpy() == b'2':
        ReturnLabel = tf.constant(2, dtype=tf.dtypes.int32)
    elif a.numpy() == b'1':
        ReturnLabel = tf.constant(1, dtype=tf.dtypes.int32)
    elif a.numpy() == b'0':
        ReturnLabel = tf.constant(0, dtype=tf.dtypes.int32)
    else:
        tf.print(a.numpy())
        tf.print(str(a.numpy()))
        tf.print(type(str(a.numpy())))
        assert False

    #tf.print(ReturnLabel.shape)

    ReturnArray.set_shape([ImageWidth, ImageHeight, 3])
    ReturnLabel.set_shape(())
    
    #tf.print(ReturnLabel.shape)
    
    return ReturnArray, ReturnLabel

def load(a):
    # `load` is executed in graph mode, so `a` and `b` are non-eager Tensors.
    # tf.print(type(a))  # <class 'tensorflow.python.framework.ops.Tensor'>
    
    Image, Label = tf.py_function(load_py, inp=[a], Tout=[tf.float32, tf.int32])
    Image.set_shape([ImageWidth, ImageHeight, 3])
    Label.set_shape(())
    return Image, Label
    
def ReShapeFix(a, b):
    return tf.reshape(a, shape=(ImageWidth, ImageHeight, 3)), b



def CreateFinalDataSets(): 
    HealthyFilePath = r"C:\CovidThreeLMkii\HealthyMFCC\*.npy"
    # NotHealthyFilePath = r"C:\CovidThree\NonHealthyMFCC\*.npy"
    
    # HealthyDataset = (
    #     tf.data.Dataset.list_files(HealthyFilePath)
    #     .map(load)
    #     )
    
    # NotHealthyDataset = (
    #     tf.data.Dataset.list_files(NotHealthyFilePath)
    #     .map(load)
    #     )
    
    TheOverallDataset = (
        tf.data.Dataset.list_files(HealthyFilePath)
        .map(load)
        )
    
    ShuffleAmount = int(len(TheOverallDataset) / BatchSize)
    
    TheOverallDataset = (
        TheOverallDataset
        .batch(BatchSize)
        #.shuffle(ShuffleAmount, reshuffle_each_iteration=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    
    TheOverallDataset = TheOverallDataset.shuffle(ShuffleAmount, seed=12)
    
    # TheOverallDataset = (
    #     tf.data.Dataset.from_tensor_slices(InputData)
    #     .batch(BatchSize)
    #     # .shuffle(BufferSize, reshuffle_each_iteration=True)
    #     # .prefetch(tf.data.experimental.AUTOTUNE)
    #     )
    
    TrainingSize = int(len(TheOverallDataset) * 0.7)
    ValidationSize = int(len(TheOverallDataset) * 0.15)
    TheTrainingDataset = TheOverallDataset.take(TrainingSize)
    TheValidationDataset = TheOverallDataset.skip(TrainingSize).take(ValidationSize)
    TheTestingDataset = TheOverallDataset.skip(TrainingSize + ValidationSize)
    
    return TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset

def LoadCOVID():
    HealthyFilePath = r"C:\CovidThreeLMkii\HealthyMFCC"
    
    DaData = []
    DaLablz = []
    for FileName in os.listdir(HealthyFilePath):
        Label = FileName.split(sep="_")[1]
        Label = int(Label)
        
        FilePath = os.path.join(HealthyFilePath, FileName)
        
        TheSpecOrMFCC = np.load(FilePath)
        DaData.append(TheSpecOrMFCC)
        DaLablz.append(Label)
    
    DaData = np.asarray(DaData)
    DaLablz = np.asarray(DaLablz)
    
    return DaData, DaLablz

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
        
    #TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()

    # tf.config.experimental.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    seed = 42


    InputData, LabelData = LoadCOVID()

    BatchSize = 32
    BufferSize = 300
    
    ModelDir = r"E:\Python\Diss\Models\CovidClassification_Multi_MFCC"
    MyModel = tf.keras.models.load_model(ModelDir)
    
    # T = random.sample(list(InputData), 1)
    # T = np.asarray(T)
    
    ModelPredictions = MyModel.predict(InputData[:1024])
    ModelPredictions = np.argmax(ModelPredictions, axis=1)
    
    Matrix = tf.math.confusion_matrix(LabelData[:1024], ModelPredictions)
    
    # ConfusionMatrix = tf.math.confusion_matrix(labels=LabelData, predictions=ModelPredictions)
    # print(ConfusionMatrix)
    
    # ModelCunt = tf.keras.Sequential()
    # ModelCunt.add(tf.keras.Input(shape=(500,100,3)))
    # ModelCunt.add(tf.keras.layers.Conv2D(32, 3))
    # ModelCunt.add(tf.keras.layers.Flatten())
    # ModelCunt.add(tf.keras.layers.Dense(3, activation="sigmoid"))
    
    # ClassNames = ["N", "P", "asd"]
    
    # Classifier = estimator(ModelCunt, ClassNames)
    # plot_confusion_matrix(Classifier, X=ModelPredictions, y_true=LabelData[:32], normalize="true", cmap="cividis")
    # plt.show()
    #plt.savefig("BiRNN_LSTM_Full_Matrix.png", dpi=300)
