#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:08:18 2021
 * Copyright (c) 2021 Den Zemtcov <dlya.v5ego@yandex.com>

 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
@author: Den Zemtcov
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from IPython.display import Image
#%matplotlib imline

physical_devices_cpu = tf.config.experimental.list_physical_devices("CPU")
physical_devices_gpu = tf.config.experimental.list_physical_devices("GPU")

#print("CPU available: ", len(physical_devices_cpu))

mobile_v1 = tf.keras.applications.mobilenet.MobileNet()
mobile_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2()

def prepare_image_v1(file):
    img_path = os.path.join(os.getcwd(), "MobileNet-sample")
    os.chdir(img_path)
    img = image.load_img(file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def prepare_image_v2(file):
    img_path = os.path.join(os.getcwd(), "MobileNet-sample")
    os.chdir(img_path)
    img = image.load_img(img_path + file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

#Image(filename=('MobileNet-sample/1.PNG'), width=300, height=200)

preprocessing_img = prepare_image_v1('24.jpg')
predictions_v1 = mobile_v1.predict(preprocessing_img)
results_v1 = imagenet_utils.decode_predictions(predictions_v1)
results_v1

predictions_v2 = mobile_v2.predict(preprocessing_img)
results_v2 = imagenet_utils.decode_predictions(predictions_v2)
results_v2
