# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:26:59 2022

@author: james
"""

import tensorflow as tf
import tensorflow_addons as tfa
import pandas
import numpy as np
import math
import time
import copy
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
import random

#Define Config Variables
ImageWidth = 500
ImageHeight = 78 #78 #200 #1025



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
    
    # TheOverallDataset = TheOverallDataset.shuffle(ShuffleAmount, seed=12)
    
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

















def LoadECGData():


    
    ECGDirectoryOne = r"E:\Python\DataSets\ECG\mitbih_train.csv"
    ECGDataOne = np.loadtxt(ECGDirectoryOne, delimiter=',')
    ECGDirectoryTwo = r"E:\Python\DataSets\ECG\mitbih_test.csv"
    ECGDataTwo = np.loadtxt(ECGDirectoryTwo, delimiter=',')
    
    ECGData = np.concatenate((ECGDataOne, ECGDataTwo))

    ECGData = ECGDataOne
    
    ECGLabels = ECGData[:,-1]
    ECGData = ECGData[:, :-1]
    
    ECGLabels = ECGLabels.astype(int)
    
    return ECGData, ECGLabels





# def LoadECGData():    
#     ECGDirectoryOne = r"E:\Python\DataSets\ECG\mitbih_train.csv"
#     ECGDataOne = np.loadtxt(ECGDirectoryOne, delimiter=',')
#     ECGDirectoryTwo = r"E:\Python\DataSets\ECG\mitbih_test.csv"
#     ECGDataTwo = np.loadtxt(ECGDirectoryTwo, delimiter=',')
    
#     ECGData = np.concatenate((ECGDataOne, ECGDataTwo))
    
#     ECGLabels = ECGData[:,-1]
#     ECGData = ECGData[:, :-1]
    
#     ECGLabels = ECGLabels.astype(int)
#     ECGData = np.expand_dims(ECGData, axis=-1)
    
    
    
#     WithFirstData = []
#     WithFirstLabels = []
    
#     WithoutFirstData = []
#     WithoutFirstLabels = []
    
#     for Data, Label in zip(ECGData, ECGLabels):
#         if(Label == 0):
#             WithFirstData.append(Data)
#             WithFirstLabels.append(Label)
#         else:
#             WithoutFirstData.append(Data)
#             WithoutFirstLabels.append(Label)
    
#     print(set(WithFirstLabels), len(WithFirstLabels))
#     print(set(WithoutFirstLabels), len(WithoutFirstLabels))
    
#     ListOfDataLabelTuples = random.sample(list(zip(WithFirstData, WithFirstLabels)), 9000)
    
#     NewFirstData = []
#     NewFirstLabels = []
    
#     for Tuple in ListOfDataLabelTuples:
#         NewFirstData.append(Tuple[0])
#         NewFirstLabels.append(Tuple[1])
        
    
#     FinalECGData = np.concatenate((NewFirstData, WithoutFirstData))
#     FinalECGLabels = np.concatenate((NewFirstLabels, WithoutFirstLabels))
    
#     return FinalECGData, FinalECGLabels






#https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
def make_square(im, min_size=100, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im



def LoadImages(BasePath):
    #BasePath = r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_0\all"
    FilesList1 = [f for f in listdir(BasePath) if isfile(join(BasePath, f))]
    FilesList = copy.copy(FilesList1)
    print(len(FilesList))
    ReturnImages = []
    for i in FilesList:
        PathFix = "\\"
        Directory = BasePath + PathFix + i
        ReadImage = Image.open(Directory)
        #oldsize = ReadImage.size
        ReadImage = ReadImage.crop(ReadImage.getbbox())
        #newsize = ReadImage.size
        #print(oldsize, newsize)
        PaddedImage = make_square(ReadImage)
        ReSizedImage = PaddedImage.resize((200,200))
        ReturnImages.append(ReSizedImage)
        
        # if(random.random() < 0.1):
        #     plt.figure()
        #     plt.imshow(ReSizedImage)
        #     plt.colorbar()
        #     plt.grid(False)
        #     plt.show()
        #     assert False

    return ReturnImages



def PrepImages():

    AllImages = []
    AllLabels = []
    
    # for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_0\all"):
    #     AllImages.append(i)
    #     AllLabels.append(1)
    # for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_1\all"):
    #     AllImages.append(i)
    #     AllLabels.append(1)
    for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_2\all"):
        AllImages.append(i)
        AllLabels.append(1)

    AmountALL = len(AllImages)
    # for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_0\hem"):
    #     AllImages.append(i)
    #     AllLabels.append(0)
    # for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_1\hem"):
    #     AllImages.append(i)
    #     AllLabels.append(0)
    for i in LoadImages(r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_2\hem"):
        AllImages.append(i)
        AllLabels.append(0)
        
    
    
    # for i in range(AmountALL):
    #     AllLabels.append(1)
    # for i in range(len(AllImages) - AmountALL):
    #     AllLabels.append(0)

    

    NumpyArrayData = []
    
    for i in AllImages:
        NumpyArrayData.append(tf.keras.preprocessing.image.img_to_array(i))

    print(AllLabels)
    
    AllImages = np.asarray(NumpyArrayData, order="K")
    AllLabels = np.asarray(AllLabels, order="K")
    
    AllImages = AllImages/255.0
    AllImages *= 3.0 #3.5
    
    return AllImages, AllLabels    



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

if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    BatchSize = 32
    
    # ImageNumpyArrays, LabelNumpyArrays, = PrepImages()
    
    ECGData, ECGLabels = LoadECGData()
    
    TRUEVALUES = []
    
    for Label in ECGLabels:
        if Label == 0:
            TRUEVALUES.append(np.array([1, 0, 0, 0, 0]))
        elif Label == 1:
            TRUEVALUES.append(np.array([0, 1, 0, 0, 0]))
        elif Label == 2:
            TRUEVALUES.append(np.array([0, 0, 1, 0, 0]))
        elif Label == 3:
            TRUEVALUES.append(np.array([0, 0, 0, 1, 0]))
        elif Label == 4:
            TRUEVALUES.append(np.array([0, 0, 0, 0, 1]))
        else:
            print(Label)
            assert False
    
    TRUEVALUES = np.asarray(TRUEVALUES)
    
    
    TheX, TheX2 = LoadCOVID()
    
    TRUEVALUES = []
    
    for Label in TheX2:
        if Label == 0:
            TRUEVALUES.append(np.array([1, 0, 0]))
        elif Label == 1:
            TRUEVALUES.append(np.array([0, 1, 0]))
        elif Label == 2:
            TRUEVALUES.append(np.array([0, 0, 1]))
        else:
            print(Label)
            assert False
    
    TRUEVALUES = np.asarray(TRUEVALUES)
    
    # assert False
    
    # TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateDatasets(ECGData, ECGLabels)
    
    
    
    # ModelDir = r"E:\Python\Diss\Models\ECGClassification_MLP"
    # MyModel = tf.keras.models.load_model(ModelDir)
    
    # MyModel.compile(optimizer="sgd",
    #           loss=tf.keras.losses.BinaryCrossentropy(),
    #           metrics=[tf.keras.metrics.AUC()])
    
    # OutputData = MyModel.evaluate(TheOverallDataset, batch_size=BatchSize, verbose = 2)


    
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    #TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateDatasets(ECGData, ECGLabels)
    # TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()
    
    ModelDir = r"E:\Python\Diss\Models\CovidClassification_Multi_MFCC"
    MyModel = tf.keras.models.load_model(ModelDir)
    

    
    # MyModel.compile(optimizer="sgd",
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #           metrics=Metric
    #           )
    # OutputData = MyModel.predict(ECGData, batch_size=BatchSize, verbose = 2)
    
    

    
    OutputData = MyModel.predict(TheX[:1024], batch_size=BatchSize, verbose = 2)
    
    # OutputData = np.argmax(OutputData, axis=1)
    # OutputData = OutputData.astype("int32")
    
    Metric = tfa.metrics.F1Score(num_classes=3)
    Metric.update_state(TRUEVALUES[:1024], OutputData)
    Result = Metric.result()
    print(np.mean(Result.numpy()), Result.numpy())
    
    assert False
    
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    
    Metric = tfa.metrics.F1Score(num_classes=5, average="weighted")
    Metric.update_state(TRUEVALUES, OutputData)
    Result = Metric.result()
    print(np.mean(Result.numpy()), Result.numpy())
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    
    Metric = tfa.metrics.F1Score(num_classes=5, average="micro")
    Metric.update_state(TRUEVALUES, OutputData)
    Result = Metric.result()
    print(np.mean(Result.numpy()), Result.numpy())
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    Metric = tfa.metrics.F1Score(num_classes=5, average="macro")
    Metric.update_state(TRUEVALUES, OutputData)
    Result = Metric.result()
    print(np.mean(Result.numpy()), Result.numpy())
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)    
    
    assert False
    
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()
    
    ModelDir = r"E:\Python\Diss\Models\CovidClassification_VGG16_MFCC"
    MyModel = tf.keras.models.load_model(ModelDir)
    
    MyModel.compile(optimizer="sgd",
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC()])
    
    OutputData = MyModel.evaluate(TheTestingDataset, batch_size=BatchSize, verbose = 2)
    
    
    
    
    
    
    
    
    assert False
    
    
    
    
    
    
    
    
    
    
    
    
    
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()
    
    ModelDir = r"E:\Python\Diss\Models\CovidClassification_Spectrogram"
    MyModel = tf.keras.models.load_model(ModelDir)
    
    MyModel.compile(optimizer="sgd",
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics="accuracy")
    
    OutputData = MyModel.evaluate(TheOverallDataset, batch_size=BatchSize, verbose = 2)
    
    
    
    
    
    
    
    
    
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()
    
    ModelDir = r"E:\Python\Diss\Models\CovidClassification_Spectrogram"
    MyModel = tf.keras.models.load_model(ModelDir)
    
    MyModel.compile(optimizer="sgd",
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC()])
    
    OutputData = MyModel.evaluate(TheTestingDataset, batch_size=BatchSize, verbose = 2)
    
    
    
    
    
    
    
    
