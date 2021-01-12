import numpy as np
import os
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw
from imutils.contours import sort_contours
import argparse
import imutils
import cv2

import fct_utiles as fct


def load_dataset():
	
	datasetPath = fct.DATASET_PATH
	
	# initialize the list of data and labels
	data = []
	labels = []

    # loop over the rows of the dataset
	for row in open(datasetPath):
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")
		data.append(image)
		labels.append(label)


	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")

	# Set depth of image to 3
	data = data.reshape(data.shape[0], 32, 32, 3)


	# return a 2-tuple of the A-Z data and labels
	return (data, labels)


def train_model (modelName, liste):

	modelPath = fct.MODELS_LOCAL_PATH + modelName + '.h5'

	# Split to train and test
	x, y = load_dataset()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

	# Change type from int to float and normalize to [0, 1]
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# Optionally check the number of samples
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# Convert class vectors to binary class matrices (transform the problem to multi-class classification)
	num_classes = len(liste)
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# Check if there is a pre-trained model
	if not os.path.exists(modelPath):
		# Create a neural network with 2 convolutional layers and 2 dense layers
		model = Sequential()

		model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(AveragePooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		# Train the model
		training = model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=1, validation_data=(x_test, y_test))

		# Save the model
		model.save(modelPath)

	else:
		# Load the model from disk
		model = load_model(modelPath)

	# Get loss and accuracy on validation set
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return modelPath, training


def plot(training):

	plt.plot(training.history['loss'], color='red', label='Training loss')
	plt.plot(training.history['val_loss'],  color='green', label='Validation loss')

	plt.xlabel('Epochs')
	plt.ylabel('Loss')

	plt.show()


def computer_metrics (modelPath, liste):

	# Load the model
	model = load_model(modelPath)

	# Split to train and test
	x, y = load_dataset()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

	# Process the images as in training
	x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
	x_test = x_test.astype('float32')
	x_test /= 255

	# Make predictions
	# predictions = model.predict_classes(x_test, verbose=0)
	predictions = np.argmax(model.predict(x_test), axis=-1)
	correct_indices = np.nonzero(predictions == y_test)[0]
	incorrect_indices = np.nonzero(predictions != y_test)[0]

	# Optionally plot some images
	print("Correct: %d" %len(correct_indices))
	plt.figure()
	for i, correct in enumerate(correct_indices[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(x_test[correct].reshape(32,32,3), cmap='gray', interpolation='none')
		plt.title(liste[predictions[correct]])
	
	print("Incorrect: %d" %len(incorrect_indices))
	plt.figure()
	for i, incorrect in enumerate(incorrect_indices[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(x_test[incorrect].reshape(32,32,3), cmap='gray', interpolation='none')
		plt.title(liste[predictions[incorrect]])


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*13.5,col_size*2.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='plasma')
            activation_index += 1