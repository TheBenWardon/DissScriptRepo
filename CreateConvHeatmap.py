# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:33:30 2022

@author: james
"""


#Mostly nabbed from here;
#https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759



import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



#https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
def make_square(im, min_size=100, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im



ModelDir = r"E:\Python\Diss\Models\LeukemiaBinaryClassification_VGG16_FOLD1"
MyModel = tf.keras.models.load_model(ModelDir)

ImageDir = r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_2\hem\UID_H1_7_1_hem.bmp"
ImageDir =r"E:\Python\DataSets\archive\C-NMC_Leukemia\training_data\fold_2\all\UID_21_5_2_all.bmp"
ReadImage = Image.open(ImageDir).convert("RGB")
ReadImage = ReadImage.crop(ReadImage.getbbox())
PaddedImage = make_square(ReadImage)
ReSizedImage = PaddedImage.resize((200,200))
ReSizedImage = np.asarray(ReSizedImage)
ReSizedImage = ReSizedImage/255.0
ReSizedImage *= 3.0
ExpandedImage = np.expand_dims(ReSizedImage, axis=0)

prediction = MyModel.predict(ExpandedImage)


with tf.GradientTape() as tape:
  last_conv_layer = MyModel.get_layer('block5_pool')
  iterate = tf.keras.models.Model([MyModel.inputs], [MyModel.output, last_conv_layer.output])
  model_out, last_conv_layer = iterate(ExpandedImage)
  class_out = model_out[:, np.argmax(model_out[0])]
  grads = tape.gradient(class_out, last_conv_layer)
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
heatmap = np.maximum(heatmap, 0)
assert False
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((6, 6))
plt.matshow(heatmap)
plt.show()





#img = cv2.imread(ReSizedImage)

INTENSITY = 0.005

heatmap = cv2.resize(heatmap, (ReSizedImage.shape[1], ReSizedImage.shape[0]))

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

HeatMapImage = heatmap * INTENSITY + ReSizedImage

#plt.imshow(ReSizedImage)

cv2.imshow("HeatMapWindow", HeatMapImage)
cv2.imshow("ImageWindow", ReSizedImage)
cv2.waitKey()