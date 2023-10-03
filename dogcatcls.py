import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('DOG_CAT',
 
target_size=(224,224),
 
color_mode='rgb',
 
batch_size=32,
 
class_mode='categorical',
 
shuffle=True)

train_generator.class_indices.values()
NO_CLASSES = len(train_generator.class_indices.values())
train_generator.class_indices.values()

model = VGG16(include_top=False, input_shape=(224, 224, 3))
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
# final layer with softmax activation
preds = Dense(NO_CLASSES, activation='softmax')(x)
model = Model(model.input, preds)
# don't train first 19 layers
for layer in model.layers[:19]:
 layer.trainable = False
# train rest of the layers - 19 onwards
for layer in model.layers[19:]:
 layer.trainable = True

 model.compile(optimizer='Adam',
 loss='categorical_crossentropy',
 metrics=['accuracy'])

 model.fit(x=train_generator,batch_size= 1,verbose= 1,epochs= 20)

 class_dictionary = {0:'Cat', 1:'Dog'}
imgtest = cv2.imread(' dog.481.jpg',cv2.IMREAD_UNCHANGED)
size = (224, 224)
imgtest = cv2.resize(imgtest, size)
# prepare the image for prediction
x = imgtest
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.imagenet_utils.preprocess_input(x)
# making prediction
cv2_imshow(imgtest)
predicted_prob = model.predict(x)
print(predicted_prob)
print(predicted_prob[0].argmax())
print("Predicted : " + class_dictionary[predicted_prob[0].argmax()])
print("============================\n")
