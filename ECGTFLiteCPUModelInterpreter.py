# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:27:02 2022

@author: james
"""

import random
import os
import pathlib
import numpy as np
import time
import tensorflow as tf


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
    ECGData = ECGData.astype("float32")
    
    
    return ECGData, ECGLabels







# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = r"E:\Python\Diss\TFLiteModels\ECGClassification_MLP_TFLite.tflite"

# Initialize the TF interpreter
interpreter = tf.lite.Interpreter(model_file)
interpreter.allocate_tensors()


# Load and Select the ECG Data
TheECGData, TheECGLabels = LoadECGData()

ECGsToRun = random.sample(list(TheECGData), 30)


# Run an inference

FinalInferenceTimes = []

for ECG in ECGsToRun:
    LastRun = 0
    for runs in range(4):
      ECGToUse = random.choice(TheECGData)
      ECGToUse = np.expand_dims(ECGToUse, axis=0)
      
      TheInput = interpreter.get_input_details()[0]
      interpreter.set_tensor(TheInput["index"], ECGToUse)
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      print('%.3fms' % (inference_time * 1000))
      # classes = classify.get_classes(interpreter, top_k=1)
      # for TheClass in classes:
      #   print(TheClass)

        
      LastRun = inference_time
      
    print("========")
    FinalInferenceTimes.append(LastRun)

print(FinalInferenceTimes)



