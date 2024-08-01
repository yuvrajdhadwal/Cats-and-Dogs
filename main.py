import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing - Image Augmentation to Prevent Overfitting
train_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size=32, class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1/255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Building CNN
cnn = tf.keras.models.Sequential()
# First Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# First Layer Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Second Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Flatten
cnn.add(tf.keras.layers.Flatten())
# Build Artificial Neural Network part of CNN
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the Model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Save model
cnn.save(filepath='catsVsDogs.keras')