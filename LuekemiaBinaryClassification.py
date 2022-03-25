# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:15:56 2022

@author: james
"""

import tensorflow as tf
import pandas
import numpy as np
import math
import time
import copy
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard





#https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
def make_square(im, min_size=100, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im



def PrepImagesSmall():
    AllImages = []
    AllLabels = []
    
    TheCSV = TheCSV = pandas.read_csv(
        r"E:\Python\DataSets\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data_labels.csv",
        )
    AllLabels = TheCSV["labels"].tolist()
    ImageNames = TheCSV["new_names"].tolist()
    
    for i in ImageNames:   
        Fix = "\\"
        Directory = r"E:\Python\DataSets\archive\C-NMC_Leukemia\validation_data\C-NMC_test_prelim_phase_data" + Fix + i
        ReadImage = Image.open(Directory)

        ReadImage = ReadImage.crop(ReadImage.getbbox())

        PaddedImage = make_square(ReadImage)
        ReSizedImage = PaddedImage.resize((200,200))
        AllImages.append(ReSizedImage)
    
    NumpyArrayData = []
    
    for i in AllImages:
        NumpyArrayData.append(tf.keras.preprocessing.image.img_to_array(i))

    print(AllLabels)
    
    AllImages = np.asarray(NumpyArrayData, order="K")
    AllLabels = np.asarray(AllLabels, order="K")
    
    AllImages = AllImages/255.0
    AllImages *= 3.0 #3.5
    
    return AllImages, AllLabels
    


def PrepImagesLarge():

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
    # AllImages *= 3.0 #3.5
    
    
    AllImages = tf.convert_to_tensor(AllImages)
    AllLabels = tf.convert_to_tensor(AllLabels)
    
    return AllImages, AllLabels    

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





def CreateDatasets(InputData, InputLabels):
    
    BufferSize = math.floor(len(InputData)/BatchSize)
    
    TheOverallDataset = (
        tf.data.Dataset.from_tensor_slices((InputData, InputLabels))
        .batch(BatchSize)
        .prefetch(tf.data.experimental.AUTOTUNE)
        )

    TheOverallDataset = TheOverallDataset.shuffle(BufferSize)  #reshuffle_each_iteration=True
    
    TrainingSize = int(len(TheOverallDataset) * 0.7)
    ValidationSize = int(len(TheOverallDataset) * 0.15)
    TheTrainingDataset = TheOverallDataset.take(TrainingSize)
    TheValidationDataset = TheOverallDataset.skip(TrainingSize).take(ValidationSize)
    TheTestingDataset = TheOverallDataset.skip(TrainingSize + ValidationSize)
    
    return TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset
    


# def CreateNewModel(): #"Resnet based"
#     Inputs = tf.keras.Input(shape=([200, 200, 3]))
    
#     ResidualExtract = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(Inputs)
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


# def CreateNewModel(): #SimpleCNN
#     Inputs = tf.keras.Input(shape=([200, 200, 3]))
    
#     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(Inputs)
#     x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(Inputs)
#     x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(Inputs)
#     x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(Inputs)
#     x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(384, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(192, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(1, activation="relu")(x)
    
#     Outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Outputs)
#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



def CreateNewModel(): #VGG16PlusNewFullyConnected
      
    BaseModel = tf.keras.applications.VGG16(include_top=True, weights="imagenet")

    Inputs = tf.keras.Input(shape=([200, 200, 3]))
    x = BaseModel.layers[1](Inputs)
    for Layer in BaseModel.layers[2:-3]:
        x = Layer(x)

    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    Output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    NewModel = tf.keras.Model(inputs=Inputs, outputs = Output)

    tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
    return NewModel



# def CreateNewModel(): #VGG16PlusNewFullyConnectedNOWEIGHTS
      
#     BaseModel = tf.keras.applications.VGG16(include_top=True, weights=None)

#     Inputs = tf.keras.Input(shape=([200, 200, 3]))
#     x = BaseModel.layers[1](Inputs)
#     for Layer in BaseModel.layers[2:-3]:
#         x = Layer(x)

#     x = tf.keras.layers.Dense(4096, activation="relu")(x)
#     x = tf.keras.layers.Dense(4096, activation="relu")(x)
#     Output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

#     NewModel = tf.keras.Model(inputs=Inputs, outputs = Output)

#     tf.keras.utils.plot_model(NewModel, show_shapes=True)
    
#     return NewModel



if __name__ == "__main__":
    
    PrepImagesSmall()
    
    # tf.config.experimental.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    BatchSize = 32 #24 for Simple, 32 for others
    
    ModelName = "LeukemiaBinaryClassification_VGG16_FOLD2_FINAL_NOINTENSE_NORESHUFFLE-{}".format(int(time.time())) #LeukemiaBinaryClassification_ResnetBased-{}
    Dir = r"E:\Python\Diss\Logs\\"
    Dir += ModelName
    print(Dir)
    MyTensorBoard = TensorBoard(log_dir=Dir)
    
    CheckPointFilePath = r"E:\Python\Diss\ModelCheckpoints\\" + "CheckPoint_" + "LeukemiaBinaryClassification_VGG16_FOLD2_FINAL_NOINTENSE_NORESHUFFLE"
    MyCheckPoints = tf.keras.callbacks.ModelCheckpoint(
        filepath=CheckPointFilePath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
        )

    # ImageNumpyArraysSmall, LabelNumpyArraysSmall = PrepImagesSmall()
    ImageNumpyArraysLarge, LabelNumpyArraysLarge, = PrepImagesLarge()
    
    # ImageNumpyArrays = np.concatenate((ImageNumpyArraysSmall, ImageNumpyArraysLarge))
    # LabelNumpyArrays = np.concatenate((LabelNumpyArraysSmall, LabelNumpyArraysLarge))
    
    TheTrainingDataset, TheValidationDataset, TheTestingDataset, TheOverallDataset = CreateDatasets(ImageNumpyArraysLarge, LabelNumpyArraysLarge)
    
    MyModel = CreateNewModel()
    MyModel.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(0.001), #RMSprop(1e-3) Adam(0.001) SGD(0.001)
        metrics=["accuracy"],
        )
    
    MyModel.fit(TheTrainingDataset, validation_data=TheValidationDataset, batch_size=BatchSize, epochs=128, callbacks=[MyTensorBoard, MyCheckPoints])
    
    TestLoss, TestAccuracy = MyModel.evaluate(TheTestingDataset, batch_size=BatchSize, verbose = 2)
    print(TestLoss, TestAccuracy)
    
    MyModel.save(r"E:\Python\Diss\Models\LeukemiaBinaryClassification_VGG16_FOLD2_FINAL_NOINTENSE_NORESHUFFLE")
    
    
    
    

    
    
    
    
    
    
    
    
    