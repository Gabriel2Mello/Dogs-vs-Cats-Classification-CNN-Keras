# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import random
import os

## Datasets paths
train_dir = './dogs-vs-cats/dataset_treino/'
test_dir = './dogs-vs-cats/dataset_teste/'
validation_dir = './dogs-vs-cats/dataset_validation/'

## Define image properties:
# Largura imagem
Image_Width = 128
# Altura imagem
Image_Height = 128
# Tamanho da imagem
Image_Size = (Image_Width, Image_Height)
# Canais de imagem
Image_Channels = 3

## Prepare dataset for training model:
filenames = os.listdir(train_dir)
categories = []
for file in filenames:
    category = file.split('.')[0]    
    if category == 'dog':
        categories.append(1)
    else:     
        categories.append(0)

df = pd.DataFrame({
    'filename' : filenames,
    'category' : categories
    })

## Create the neutral net model: 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Dense#, Activation
from tensorflow.keras.layers import BatchNormalization

model = Sequential()

## Conv_1
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=
                (128, 128, 3))
          )
model.add(keras.layers.Conv2D(32,
                              (3,3),
                              input_shape=(Image_Width, Image_Height, Image_Channels),
                              activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

## Conv_2
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

## Conv_3
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

## Flatten
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

## Optimizer and loss
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',              
              metrics = ['accuracy']
              )

## Analyzing model
model.summary()

## Define callbacks and learning rate:
from tensorflow.keras.callbacks import EarlyStopping    
from tensorflow.keras.callbacks import ReduceLROnPlateau

earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience = 2,
                                            verbose = 1,
                                            factor = 0.5,
                                            min_lr = 0.00001
                                            )
callbacks = [earlystop, learning_rate_reduction]

## Manage data:
df["category"] = df["category"].replace({ 0 : 'cat', 1 : 'dog' })    
train_df, validate_df = train_test_split(df, test_size = 0.30,
                                         random_state = 42)

train_df = train_df.reset_index(drop = True)
validate_df = validate_df.reset_index(drop = True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

## Training and validation data generator:
train_datagen = ImageDataGenerator(rotation_range = 15,
                                   rescale = 1./255,
                                   shear_range = 0.1,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   )
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    train_dir,
                                                    x_col = 'filename',
                                                    y_col = 'category',
                                                    target_size = Image_Size,
                                                    class_mode = 'categorical',
                                                    batch_size = batch_size
                                                    )

validation_datagen  = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    train_dir,
    x_col = 'filename',
    y_col = 'category',
    target_size = Image_Size,
    class_mode = 'categorical',
    batch_size = batch_size
    )

## Test data preparation:
test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({
    'filename' : test_filenames
    })
nb_samples = test_df.shape[0]

test_datagen = ImageDataGenerator(rotation_range = 15,
                                  rescale = 1./255,                                  
                                  shear_range = 0.1,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1
                                  )
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  test_dir,
                                                  x_col = 'filename',
                                                  y_col = None,
                                                  target_size = Image_Size,
                                                  class_mode = None,
                                                  batch_size = batch_size
                                                  )

## Model training:
epochs = 3
history = model.fit(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = total_validate//batch_size,
    steps_per_epoch = total_validate//batch_size,
    callbacks = callbacks    
    )        

## Save the model:
model.save("model1_cats_dogs_10epoch.h5")

## Make categorical prediction
predict = model.predict(test_generator,
                        steps = np.ceil(nb_samples/batch_size))

## Convert labels to categories
test_df['category'] = np.argmax(predict, axis = -1)

label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog' : 1, 'cat' : 0})

## Visualize the prediction results
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize = (12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(test_dir + filename,
                   target_size = Image_Size)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')

plt.tight_layout()    
plt.show()




'''
## Test model performance on custom data:
results = { 0 : 'cat', 1 : 'dog'}
from PIL import Image
im = Image.open("./dogs-vs-cats/1234.jpg")
im = im.resize(Image_Size)
im = np.expand_dims(im, axis = 0)
im = np.array(im)
im = im/255
pred = np.argmax(model.predict((im), axis = -1))
print(pred, results[pred])
'''

























