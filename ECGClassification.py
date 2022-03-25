# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:13:13 2022

@author: james
"""

import matplotlib
import numpy as np
import tensorflow as tf
import time
import random
import math
from tensorflow.keras.callbacks import TensorBoard


GotZero = False
GotOne = False
GotTwo = False
GotThree = False
GotFour = False

ECGArray = []
ECGLabels = []

for i in x:
    print(i)
    if i[1] == 0 and GotZero == False:
        ECGArray.append(i[0])
        ECGLabels.append(0)
        GotZero = True
    if i[1] == 1 and GotOne == False:
        ECGArray.append(i[0])
        ECGLabels.append(1)
        GotOne = True
    if i[1] == 2 and GotTwo == False:
        ECGArray.append(i[0])
        ECGLabels.append(2)
        GotTwo = True
    if i[1] == 3 and GotThree == False:
        ECGArray.append(i[0])
        ECGLabels.append(3)
        GotThree = True
    if i[1] == 4 and GotFour == False:
        ECGArray.append(i[0])
        ECGLabels.append(4)
        GotFour = True









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
    
    
    
    WithFirstData = []
    WithFirstLabels = []
    
    WithoutFirstData = []
    WithoutFirstLabels = []
    
    for Data, Label in zip(ECGData, ECGLabels):
        if(Label == 0):
            WithFirstData.append(Data)
            WithFirstLabels.append(Label)
        else:
            WithoutFirstData.append(Data)
            WithoutFirstLabels.append(Label)
    
    print(set(WithFirstLabels), len(WithFirstLabels))
    print(set(WithoutFirstLabels), len(WithoutFirstLabels))
    
    ListOfDataLabelTuples = random.sample(list(zip(WithFirstData, WithFirstLabels)), 9000)
    
    NewFirstData = []
    NewFirstLabels = []
    
    for Tuple in ListOfDataLabelTuples:
        NewFirstData.append(Tuple[0])
        NewFirstLabels.append(Tuple[1])
        
    
    FinalECGData = np.concatenate((NewFirstData, WithoutFirstData))
    FinalECGLabels = np.concatenate((NewFirstLabels, WithoutFirstLabels))
    
    return FinalECGData, FinalECGLabels



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



def CreateNewModel():
    
    Inputs = tf.keras.Input(shape=([187, 1]))

    x = tf.keras.layers.Reshape((187,))(Inputs)
    x = tf.keras.layers.Dense(187, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(374, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    
    Outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
    NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
    tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
    return NewModel



# def CreateNewModel():
    
#     Inputs = tf.keras.Input(shape=([187, 1]))
    
#     x = tf.keras.layers.LSTM(187, return_sequences=True)(Inputs)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.LSTM(374, return_sequences=False)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(512, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



# def CreateNewModel():
    
#     Inputs = tf.keras.Input(shape=([187, 1]))
    
#     x = tf.keras.layers.GRU(187, return_sequences=True)(Inputs)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.GRU(374, return_sequences=False)(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(512, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



def CreateNewModel():
    
    Inputs = tf.keras.Input(shape=([187, 1]))
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(187, return_sequences=True))(Inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(374, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)

    Outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
    NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
    tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
    return NewModel


# def CreateNewModel():
    
#     Inputs = tf.keras.Input(shape=([187, 1]))
    
#     x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(187, return_sequences=True))(Inputs)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(374, return_sequences=False))(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(512, activation="relu")(x)

#     Outputs = tf.keras.layers.Dense(5, activation="sigmoid")(x)
#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    BatchSize = 32
    
    ModelName = "ECGClassification_MLP_4096-{}".format(int(time.time()))
    Dir = r"E:\Python\Diss\Logs\\"
    Dir += ModelName
    print(Dir)
    MyTensorBoard = TensorBoard(log_dir=Dir)
    
    CheckPointFilePath = r"E:\Python\Diss\ModelCheckpoints\\" + "CheckPoint_" + "ECGClassification_MLP_4096"
    MyCheckPoints = tf.keras.callbacks.ModelCheckpoint(
        filepath=CheckPointFilePath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
        )

    InputData, LabelData = LoadECGData()
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateDatasets(InputData, LabelData)

    MyModel = CreateNewModel()
    MyModel.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.SGD(0.001), #RMSprop(1e-3) Adam(0.001) SGD(0.001)
        metrics=["accuracy"],
        )
    
    MyModel.fit(TheTrainingDataset, validation_data=TheValidationDataset, batch_size=BatchSize, epochs=4096, callbacks=[MyTensorBoard, MyCheckPoints])
    
    TestLoss, TestAccuracy = MyModel.evaluate(TheTestingDataset, batch_size=BatchSize, verbose = 2)
    print(TestLoss, TestAccuracy)
    
    MyModel.save(r"E:\Python\Diss\Models\ECGClassification_MLP_4096")
