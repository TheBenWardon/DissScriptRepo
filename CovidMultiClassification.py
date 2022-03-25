# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:31:04 2022

@author: james
"""


from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
from os import listdir
from os.path import isfile, join
import openpyxl
import random
import pandas
import os
import math
import copy
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle



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



ModelName = "CovidClassification_Multi_Spectrogram-{}".format(int(time.time()))
Dir = r"E:\Python\Diss\Logs\\"
Dir += ModelName
print(Dir)
MyTensorBoard = TensorBoard(log_dir=Dir)



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

def CreateMFCC(Directory):
    ReadFile = tf.io.read_file(Directory)
    DecodedAudio, SampleRate = tf.audio.decode_wav(ReadFile)

    DecodedAudio = tf.squeeze(DecodedAudio, axis=-1)
    DecodedAudio = tf.cast(DecodedAudio, tf.float32)

    # StartIndex = int(StartTime * SampleRate.numpy())
    # EndIndex = int(EndTime * SampleRate.numpy())
    # DecodedAudio = DecodedAudio[StartIndex:EndIndex]
    
    SpecGram = tf.signal.stft(DecodedAudio, frame_length=FRAME_SIZE, frame_step=HOP_SIZE)
    SpecGram = tf.abs(SpecGram)
    
    if SpecGram.shape[0] > MaxAudioFileLength or SpecGram.shape[0] < MinAudioFileLength:
        return False
    
    # MelFilterBank = tf.signal.linear_to_mel_weight_matrix(
    #     num_mel_bins=MelBinNumber,
    #     num_spectrogram_bins=int((FRAME_SIZE/2)+1),
    #     sample_rate=SampleRate,
    #     lower_edge_hertz=MelLowerEdgeHertz,
    #     upper_edge_hertz=MelUpperEdgeHertz
    #     )

    # MelSpecGram = tf.tensordot(
    #     SpecGram,
    #     MelFilterBank,
    #     1
    #     )
    # MelSpecGram.set_shape(SpecGram.shape[:-1].concatenate(
    #     MelFilterBank.shape[-1:]))
    
    # LogMelSpecGram = tf.math.log(MelSpecGram + 1e-6)
    
    # mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
    #     LogMelSpecGram)[..., :BinsToTake]
    # mfcc = tf.expand_dims(mfcc, axis=-1)
    # mfcc = tf.image.grayscale_to_rgb(mfcc)
    
    ShapeString = str(SpecGram.shape) #mfcc #SpecGram
    ShapeString = ShapeString[1 : :]
    ShapeString = ShapeString[:-1:]
    ShapeString = ShapeString.split(", ")

    SmallShapeList = []
    for Val in ShapeString:
        Val = int(Val)
        SmallShapeList.append(Val)
    
    LengthToAdd = int(MaxAudioFileLength - SmallShapeList[0])   
    ZeroPadding = tf.zeros([LengthToAdd, BinsToTake, 3])
        
    # mfcc = tf.concat([mfcc, ZeroPadding], axis=0)
    # mfcc = tf.image.resize(mfcc, [ImageWidth,ImageHeight])
    
    SpecGram = tf.expand_dims(SpecGram, axis=-1)
    SpecGram = tf.image.grayscale_to_rgb(SpecGram)
    SpecGram = tf.concat([SpecGram, ZeroPadding], axis=0)
    SpecGram = tf.image.resize(SpecGram, [ImageWidth,ImageHeight])
    
    return SpecGram #SpecGram #mfcc



# def CreateFinalDataSets(InputData):
#     # TheInputData = []
#     # TheLabelsData = []
    
#     # for TheSpecGramAndLabel in InputData:
#     #     TheInputData.append(TheSpecGramAndLabel[0])
#     #     TheLabelsData.append(tf.convert_to_tensor(TheSpecGramAndLabel[1]))
    
#     # TheInputData = tf.convert_to_tensor(TheInputData)
#     # TheLabelsData = tf.convert_to_tensor(TheLabelsData)
    
#     BufferSize = math.floor((len(InputData))/BatchSize)
    
#     TheOverallDataset = (
#         tf.data.Dataset.from_tensor_slices(InputData)
#         .batch(BatchSize)
#         # .shuffle(BufferSize, reshuffle_each_iteration=True)
#         # .prefetch(tf.data.experimental.AUTOTUNE)
#         )
    
#     TrainingSize = int(len(TheOverallDataset) * 0.7)
#     ValidationSize = int(len(TheOverallDataset) * 0.15)
#     TheTrainingDataset = TheOverallDataset.take(TrainingSize)
#     TheValidationDataset = TheOverallDataset.skip(TrainingSize).take(ValidationSize)
#     TheTestingDataset = TheOverallDataset.skip(TrainingSize + ValidationSize)
    
#     return TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset
        


def CreateFinalDataSets(): 
    HealthyFilePath = r"C:\CovidThree\HealthyMFCC\*.npy"
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



def LoadPickle():
    
    yield


# def CreateNewModel(InputDataset): #ImageOnly
    
#     Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))

#     NormLayer = tf.keras.layers.Normalization()
#     NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    
#     x = NormLayer(Inputs)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.MaxPooling2D(1, 10)(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    
#     x = (lambda y: tf.expand_dims(tf.reduce_mean(y, axis=-1), -1))(x)
    
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(5000, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(5000, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1000, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(100, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation='softmax')(x)

#     NewModel = tf.keras.Model(inputs=Inputs, outputs=Outputs)
    
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



# def CreateNewModel(InputDataset):
    
#     Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))

#     NormLayer = tf.keras.layers.Normalization()
#     NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    
#     x = NormLayer(Inputs)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.MaxPooling2D(1, 10)(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
#     x = tf.keras.layers.Dense(384, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation='softmax')(x)

#     NewModel = tf.keras.Model(inputs=Inputs, outputs=Outputs)
    
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



# def CreateNewModel(InputDataset):
    
#     Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))

#     NormLayer = tf.keras.layers.Normalization()
#     NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    
#     x = NormLayer(Inputs)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.MaxPooling2D(1, 10)(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    
#     x = (lambda y: tf.expand_dims(tf.reduce_mean(y, axis=-1), -1))(x)
#     x = tf.keras.layers.Flatten()(x)
    
#     x = tf.keras.layers.Dense(384, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

#     NewModel = tf.keras.Model(inputs=Inputs, outputs=Outputs)
    
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



# def CreateNewModel(InputDataset):
    
#     Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))

#     NormLayer = tf.keras.layers.Normalization()
#     NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    
#     x = NormLayer(Inputs)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu")(x)
#     # x = tf.keras.layers.MaxPooling2D(1, 10)(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    
#     x = (lambda y: tf.expand_dims(tf.reduce_mean(y, axis=-1), -1))(x)
#     x = tf.keras.layers.Flatten()(x)
    
#     x = tf.keras.layers.Dense(384, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

#     NewModel = tf.keras.Model(inputs=Inputs, outputs=Outputs)
    
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



# def CreateNewModel(InputDataset): #"Resnet based"
#     Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))
    
#     NormLayer = tf.keras.layers.Normalization()
#     NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    
#     x = NormLayer(Inputs)    
    
#     ResidualExtract = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])
#     ResidualExtract = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])
    
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     ResidualExtract = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])   
#     ResidualExtract = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])   
    
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     ResidualExtract = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])   
#     ResidualExtract = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])
    
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     ResidualExtract = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])
#     ResidualExtract = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(ResidualExtract)
#     #x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.add([x, ResidualExtract])   
    
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(128, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) 
#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel







def CreateNewModel(InputDataset): #VGG16PlusNewFullyConnected
      
    BaseModel = tf.keras.applications.VGG16(include_top=True, weights="imagenet")

    Inputs = tf.keras.Input(shape=([ImageWidth, ImageHeight, 3]))
    NormLayer = tf.keras.layers.Normalization()
    NormLayer.adapt(data=InputDataset.map(map_func=lambda spec, label: spec))
    x = NormLayer(Inputs) 
    
    x = BaseModel.layers[1](x)
    for Layer in BaseModel.layers[2:-3]:
        x = Layer(x)

    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    Output = tf.keras.layers.Dense(3, activation="sigmoid")(x)

    NewModel = tf.keras.Model(inputs=Inputs, outputs = Output)

    tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
    return NewModel



if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    seed = 690
    PicklePath = r"E:\Python\MachineLearnin\PickleDumps\CovidDatasetThree.p"
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # D = r"E:\Python\DataSets\CovidDatasetThree\Processed\Penis.wav"
    # t = CreateMFCC(D)
    
    Directory = r"E:\Python\DataSets\CovidDatasetThree\metadata_compiled.xlsx"
    Data = pandas.read_excel(Directory)
    No = 0
    Yes = 0
    
    BasePath = r"E:\Python\DataSets\CovidDatasetThree\Processed"
    
    # Darta = []
    
    # InputDarta = []
    # LabelDarta = []
    
    # for index, row in Data.iterrows():
    #     if isinstance(row["status"], str):
            
    #         FileToGet = row["uuid"]
    #         #FileToGet = FileToGet[:-5]
    #         FileToGet += r".wav"
            
    #         Cunt = "\\"
    #         Directory = BasePath + Cunt + FileToGet
    #         #print(Directory)
            
    #         MFCC = CreateMFCC(Directory)
            
    #         if isinstance(MFCC, bool) == False:      
                
    #             Label = 0
                
    #             if row["status"] == "COVID-19":
    #                 Label = 0
    #             elif row["status"] == "symptomatic":
    #                 Label = 0
    #             elif row["status"] == "healthy":
    #                 Label = 1
    #             else:
    #                 print("urm. not good!!")
                
    #             Label = tf.convert_to_tensor(Label)
                
    #             InputDarta.append(MFCC)
    #             LabelDarta.append(Label)
            
    #         Yes += 1
    #     else:
    #         No += 1
    
    # print(Yes, No)
    
    # Darta = (InputDarta, LabelDarta)
    
    # pickle.dump(Darta, open(PicklePath, "wb"))
    # Darta = pickle.load(open(PicklePath, "rb"))
    
    # H = 0
    # S = 0
    # C = 0
    
    # for i in Darta:
    #     if i[1] == 0:
    #         C += 1
    #     elif i[1] == 1:
    #         S += 1
    #     elif i[1] == 2:
    #         H += 1
    
    # print(H, S, C)
    
    
    
    # FileName = 0
    # HealthyFilePath = r"C:\CovidThree\HealthyMFCC"
    # #NotHealthyFilePath = r"C:\CovidThree\NonHealthyMFCC"
    
    # for index, row in Data.iterrows():
    #     if isinstance(row["status"], str):
            
    #         FileToGet = row["uuid"]
    #         #FileToGet = FileToGet[:-5]
    #         FileToGet += r".wav"
            
    #         Cunt = "\\"
    #         Directory = BasePath + Cunt + FileToGet
    #         #print(Directory)
            
    #         MFCC = CreateMFCC(Directory)
            
    #         if isinstance(MFCC, bool) == False:      
            
    #             DirectoryToSave = ""
                
    #             if row["status"] == "COVID-19":
    #                 DirectoryToSave = HealthyFilePath + Cunt + str(FileName) +  "_2_"
    #             elif row["status"] == "symptomatic":
    #                 DirectoryToSave = HealthyFilePath + Cunt + str(FileName) +  "_1_"
    #             elif row["status"] == "healthy":
    #                 DirectoryToSave = HealthyFilePath + Cunt + str(FileName) +  "_0_"
    #             else:
    #                 print("urm. not good!!")
                
                
    #             np.save(DirectoryToSave, MFCC)
            
    #         Yes += 1
    #     else:
    #         No += 1
    
    #     FileName += 1
    
    # print(Yes, No)
    
    # assert False
    
    # TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets(Darta)
    
    
    CheckPointFilePath = r"E:\Python\Diss\ModelCheckpoints\\" + "CheckPoint_" + "CovidClassification_Multi_Spectrogram"
    MyCheckPoints = tf.keras.callbacks.ModelCheckpoint(
        filepath=CheckPointFilePath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
        )
    
    
    
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateFinalDataSets()

    MyModel = CreateNewModel(TheOverallDataset)
    MyModel.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #SparseCategoricalCrossentropy BinaryCrossentropy
        optimizer=tf.keras.optimizers.SGD(0.001),
        metrics="accuracy"
        )
    MyModel.fit(TheTrainingDataset, validation_data=TheValidationDataset, batch_size=BatchSize, epochs=64, callbacks=[MyTensorBoard, MyCheckPoints])
    
    TestLoss, TestAccuracy = MyModel.evaluate(TheTestingDataset, batch_size=BatchSize, verbose = 2)
    print(TestLoss, TestAccuracy)
    
    MyModel.save(r"E:\Python\Diss\Models\CovidClassification_Multi_Spectrogram")
    
    # LoadedModel = tf.keras.models.load_model(r"E:\Python\MachineLearnin\Diss\Covidiii")
    # TestLoss, TestAccuracy = LoadedModel.evaluate(TheTestingDataset, verbose = 2) #batch_size=BatchSize
    # print(TestLoss, TestAccuracy)
    
    # print("ShitFUck")
    # StatusData = np.asarray(Data["status"])
    # print(set(StatusData))
    # StatusDataAsList = StatusData.tolist()
    # print(StatusDataAsList.count("healthy"))
    # print(StatusDataAsList.count("symptomatic"))
    # print(StatusDataAsList.count("COVID-19"))
    
    
    
    # import tensorflow as tf
    # import numpy as np
    # import matplotlib.pyplot as plt
    # HealthyFilePath = r"C:\CovidThree\HealthyMFCC"
    # NotHealthyFilePath = r"C:\CovidThree\NonHealthyMFCC"
    
    # Cunt = "\\"
    
    # WAnk = "*.npy"
    
    # HealthyFilePath = HealthyFilePath + Cunt + WAnk
    # NotHealthyFilePath = NotHealthyFilePath + Cunt + WAnk
    
    # def load_py(a):
    #     # Arguments to py_function are eager tensors, so we can use `.numpy()` to get their string values.
    #     # tf.print(type(a))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    #     # tf.print(a.numpy())  # b'a'
        
    #     a = a.numpy()
        
    #     #tf.print(a)
        
    #     #a = np.array([[100, 100], [100, 100]])
        
    #     ReturnArray = 0
    #     with open(a, "rb") as FILE:
    #         ReturnArray = np.load(FILE)
        
    #     ReturnLabel = tf.strings.to_number(tf.strings.split(a, sep="_")[1], out_type=tf.dtypes.int32)
        
    #     return ReturnArray, ReturnLabel

    # def load(a):
    #     # `load` is executed in graph mode, so `a` and `b` are non-eager Tensors.
    #     # tf.print(type(a))  # <class 'tensorflow.python.framework.ops.Tensor'>
                
    #     return tf.py_function(load_py, inp=[a], Tout=[tf.float32, tf.int32])
    
    # CuntCuntOne = (
    #     tf.data.Dataset.list_files(HealthyFilePath)
    #     .map(load)
    #     )
    
    # CuntCuntTwo = (
    #     tf.data.Dataset.list_files(NotHealthyFilePath)
    #     .map(load)
    #     )
    
    # CuntCuntThree = CuntCuntOne.concatenate(CuntCuntTwo)
    
    # #CuntCunt = tf.data.Dataset.from_tensor_slices((CuntCunt, tf.zeros(65)))
    
    # for i, x in CuntCuntThree:
    #     print(x)
    #     #plt.imshow(i[0])
