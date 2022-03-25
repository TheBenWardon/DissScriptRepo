# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:07:44 2022

@author: james
"""

import tensorflow as tf



ModelDir = r"E:\Python\Diss\Models\CovidClassification_VGG16_MFCC"

Converter = tf.lite.TFLiteConverter.from_saved_model(ModelDir)
TFLiteModel = Converter.convert()



ModelName = "CovidClassification_VGG16_MFCC_TFLite"
TFLiteModelDir = r"E:\Python\Diss\TFLiteModels" + "\\" + ModelName + ".tflite"

with open(TFLiteModelDir, 'wb') as File:
  File.write(TFLiteModel)